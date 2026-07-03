from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.adapters.persistence.filesystem.markdown_store import MarkdownStore
from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.domain.actions.action_schema import action_need, canonical_action
from memoryos.domain.feedback.reward_result import compute_rewards
from memoryos.domain.memory.memory_item import MemoryItem
from memoryos.domain.memory.storage_plan import all_memory_storage_plans, memory_storage_plan
from memoryos.domain.memory.update_policy import normalize_operation_for_policy, update_policy
from memoryos.domain.scene.observation import ObservationContext
from memoryos.domain.scene.scene_features import SceneFeatures
from memoryos.domain.scene.scene_signature import stable_scene_signature
from memoryos.interfaces.api.app import handle
from memoryos.interfaces.hooks.memory_digest_hook import MemoryHook
from memoryos.observability.audit_log import AuditLogger
from memoryos.services.learning.behavior_feedback import BehaviorStats
from memoryos.services.learning.behavior_patterns import BehaviorPatternStore
from memoryos.services.learning.rl_calibrator import ReinforcementPolicyLedger
from memoryos.services.memory.extractor import JsonLLMMemoryExtractor, MemoryOperation
from memoryos.services.memory.merge_ops import merge_op_factory
from memoryos.services.memory.schema import memory_type_spec
from memoryos.services.memory.update_service import MemoryUpdateContext, MemoryUpdateService
from memoryos.services.memory.weights import score_memory_weight
from memoryos.services.policy.policy_gate import PermissionPolicyEngine
from memoryos.services.prediction.candidate_generator import Candidate
from memoryos.services.prediction.candidate_ranker import CandidateRanker
from memoryos.services.retrieval.memory_context_builder import MemoryContextBuilder
from memoryos.usecases.episode.episode_state_machine import CLOSED, FEEDBACK_PENDING
from memoryos.usecases.episode.process_observation import EpisodeProcessor
from memoryos.usecases.intervention.select_intervention import InterventionSelector
from memoryos.usecases.session.commit_session import SessionManager
from memoryos.workers.feedback_worker import FeedbackWorker
from memoryos.workers.reindex_worker import ReindexWorker
from memoryos.workers.replay_worker import ReplayWorker


class FakeProvider:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(self.response, ensure_ascii=False)


class FakeEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if any(term in lowered for term in ("cooling", "air conditioning", "hot room")):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in ("tea", "drink")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FakeRerankProvider:
    def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        scores = []
        for document in documents:
            lowered = document.lower()
            if "cooling memory" in lowered or "actual=open_ac" in lowered:
                scores.append(0.95)
            elif "tea memory" in lowered:
                scores.append(0.05)
            else:
                scores.append(0.2)
        return scores


class FakeObservationExtractor:
    def extract(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        return [
            MemoryOperation(
                action="add",
                memory_type="event",
                title="raw observation draft",
                text="rawhot123 should remain an episode draft until feedback consolidation.",
                tags=["event", "observation_draft"],
            )
        ]


class MemoryStoreTest(unittest.TestCase):
    def test_add_search_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="preference",
                    title="ac preference",
                    text="User prefers air conditioning at 25 degrees when the room is hot.",
                    tags=["temperature"],
                )
            )
            results = store.search("air conditioning hot", user_id="gulf")
            self.assertEqual(len(results), 1)
            self.assertIn("25 degrees", results[0]["content"])
            digest = MemoryHook(store).build_digest("gulf", "room hot")
            self.assertIn("<personal-memory", digest)
            self.assertIn("ac preference", digest)

    def test_session_commit_extracts_remember_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            sessions = SessionManager(store)
            sessions.add_message("gulf", "demo", "user", "记住：我怕热，空调一般开 25 度。")
            diff = sessions.commit("gulf", "demo")
            self.assertEqual(diff["summary"]["total_adds"], 1)
            results = store.search("空调", user_id="gulf")
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["type"], "event")

    def test_update_delete_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            first = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="temperature preference",
                text="User prefers 25 degrees.",
            )
            second = MemoryItem(
                user_id="gulf",
                memory_type="event",
                title="hot room event",
                text="User accepted cooling after sweating.",
            )
            store.add_memory(first)
            store.add_memory(second)
            updated = store.update_memory(first.path or "", "gulf", text="User prefers 24-26 degrees.")
            self.assertIn("24-26", updated["after"]["content"])
            self.assertNotEqual(updated["before"]["metadata"]["updated_at"], updated["after"]["metadata"]["updated_at"])
            merged = store.merge_memory(first.path or "", second.path or "", "gulf")
            self.assertEqual(merged["target_uri"], first.path)
            self.assertNotIn("#", merged["update"]["after"]["metadata"]["abstract"])
            results = store.search("sweating", user_id="gulf")
            self.assertEqual(len(results), 1)
            deleted = store.delete_memory(first.path or "", "gulf")
            self.assertEqual(deleted["uri"], first.path)
            self.assertEqual(store.search("degrees", user_id="gulf"), [])

    def test_injected_digest_is_not_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            sessions = SessionManager(store)
            sessions.add_message(
                "gulf",
                "demo",
                "user",
                '<personal-memory source="memoryos" format="digest">记住：不要重复写入。</personal-memory>',
            )
            diff = sessions.commit("gulf", "demo")
            self.assertEqual(diff["summary"]["total_adds"], 0)

    def test_profile_daily_and_event_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            profile = store.upsert_profile("gulf", "用户怕热，工作时不喜欢频繁打扰。", mode="replace")
            self.assertEqual(profile["operation"], "create")
            profile_update = store.upsert_profile("gulf", "用户允许低风险环境调节。", mode="append")
            self.assertEqual(profile_update["operation"], "update")
            self.assertIn("低风险", profile_update["after"]["content"])

            daily = store.update_daily_behavior("gulf", "上午在电脑前工作，室温较高。", day="2026-07-01")
            self.assertEqual(daily["operation"], "create")
            event = store.record_event(
                "gulf",
                event_type="ac_acceptance",
                text="用户出汗后接受了开空调。",
                day="2026-07-01",
            )
            self.assertTrue((Path(tmp) / event["daily_log"]).exists())
            self.assertNotIn("# Daily Behavior", event["daily_behavior_update"]["after"]["content"])
            results = store.search("出汗 空调", user_id="gulf")
            self.assertGreaterEqual(len(results), 1)

    def test_search_updates_lifecycle_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="habit",
                title="smoking trigger",
                text="User often wants to smoke after sitting at the computer for a long time.",
                tags=["prediction", "smoking"],
            )
            store.add_memory(item)
            before = store.resolve_memory(item.path or "", "gulf")
            self.assertEqual(before["active_count"], 0)
            self.assertEqual(before["lifecycle_state"], "warm")

            results = store.search("computer smoke", user_id="gulf", touch=True)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["active_count"], 1)
            self.assertIsNotNone(results[0]["last_accessed_at"])
            self.assertGreater(results[0]["hotness"], 0)

            metadata, _ = store.read_memory(item.path or "")
            self.assertEqual(metadata["active_count"], 1)
            report = store.lifecycle_report("gulf", limit=1)
            self.assertEqual(report[0]["id"], item.memory_id)

    def test_json_llm_extractor_commit_updates_and_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="temperature preference",
                text="User prefers 25 degrees.",
                tags=["temperature"],
            )
            store.add_memory(item)
            provider = FakeProvider(
                {
                    "operations": [
                        {
                            "action": "update",
                            "memory_type": "preference",
                            "title": "temperature preference",
                            "text": "User prefers air conditioning between 24 and 26 degrees.",
                            "tags": ["temperature"],
                            "confidence": 0.9,
                            "target": item.path,
                            "rationale": "The new statement refines the old range.",
                        },
                        {
                            "action": "ignore",
                            "memory_type": "event",
                            "title": "small talk",
                            "text": "transient greeting",
                            "tags": ["ignore"],
                            "confidence": 0.5,
                            "rationale": "Not durable memory.",
                        },
                    ]
                }
            )
            sessions = SessionManager(store, extractor=JsonLLMMemoryExtractor(provider))
            sessions.add_message(
                "gulf",
                "llm-demo",
                "user",
                '<personal-memory source="memoryos">不要写入这段</personal-memory>空调改成 24 到 26 度。',
            )
            diff = sessions.commit("gulf", "llm-demo")

            self.assertEqual(diff["summary"]["total_updates"], 1)
            self.assertEqual(diff["summary"]["total_ignores"], 1)
            updated = store.resolve_memory(item.path or "", "gulf")
            self.assertIn("24 and 26", updated["content"])
            self.assertNotIn("不要写入这段", provider.prompt)

    def test_json_llm_extractor_accepts_top_level_list_payload(self) -> None:
        provider = FakeProvider(
            [
                {
                    "action": "add",
                    "memory_type": "preference",
                    "title": "quiet reminders",
                    "text": "User prefers quiet reminders.",
                    "tags": ["preference"],
                    "confidence": 0.9,
                }
            ]
        )

        operations = JsonLLMMemoryExtractor(provider).extract([{"role": "user", "text": "remember this"}])

        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].memory_type, "preference")

    def test_delete_memory_operation_removes_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="obsolete preference",
                text="User used to prefer loud reminders.",
                tags=["preference"],
            )
            store.add_memory(item)
            service = MemoryUpdateService(store)

            diff = service.apply(
                [
                    MemoryOperation(
                        action="delete",
                        memory_type="preference",
                        title="forget obsolete preference",
                        text="User asked to forget this preference.",
                        tags=["preference", "explicit_user_intent"],
                        target=item.path,
                    )
                ],
                MemoryUpdateContext(user_id="gulf", source="test", diff_id="delete-test", explicit_user_intent=True),
            )

            self.assertEqual(diff["summary"]["total_deletes"], 1)
            with self.assertRaises(FileNotFoundError):
                store.resolve_memory(item.path or "", "gulf")

    def test_rule_based_policy_marker_sets_explicit_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operation = SessionManager(MemoryStore(Path(tmp))).extractor.extract(
                [{"role": "user", "text": "记住：以后不要自动开空调"}]
            )[0]

        self.assertEqual(operation.memory_type, "policy")
        self.assertIn("explicit_user_intent", operation.tags)

    def test_session_commit_is_idempotent_and_uses_message_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            sessions = SessionManager(store)
            sessions.add_message("gulf", "demo", "user", "记住：alpha123 是一次性事实。")

            first = sessions.commit("gulf", "demo")
            second = sessions.commit("gulf", "demo")
            sessions.add_message("gulf", "demo", "user", "记住：beta456 是一次性事实。")
            third = sessions.commit("gulf", "demo")

            self.assertEqual(first["summary"]["total_adds"], 1)
            self.assertTrue(second["idempotent"])
            self.assertEqual(third["committed_message_count"], 1)
            alpha_events = [
                row for row in store.search("alpha123", user_id="gulf", memory_type="event")
                if "/daily/" not in row["path"]
            ]
            beta_events = [
                row for row in store.search("beta456", user_id="gulf", memory_type="event")
                if "/daily/" not in row["path"]
            ]
            self.assertEqual(len(alpha_events), 1)
            self.assertEqual(len(beta_events), 1)

    def test_path_traversal_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))

            with self.assertRaises(ValueError):
                store.init("../gulf")
            with self.assertRaises(ValueError):
                SessionManager(store).add_message("gulf", "../demo", "user", "hello")
            with self.assertRaises(ValueError):
                EpisodeProcessor(store).process(user_id="gulf", episode_id="../ep", scene="hello")

    def test_search_does_not_touch_hotness_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="quiet preference",
                text="User prefers quiet reminders.",
                tags=["preference"],
            )
            store.add_memory(item)

            results = store.search("quiet reminders", user_id="gulf")
            resolved = store.resolve_memory(item.path or "", "gulf")

            self.assertEqual(results[0]["active_count"], 0)
            self.assertEqual(resolved["active_count"], 0)

    def test_hybrid_search_can_use_embedding_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp), embedding_provider=FakeEmbeddingProvider())
            store.init("gulf")
            ac = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="temperature preference",
                text="User prefers air conditioning at 25 degrees.",
                tags=["temperature"],
            )
            tea = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="tea preference",
                text="User likes green tea in the afternoon.",
                tags=["drink"],
            )
            store.add_memory(ac)
            store.add_memory(tea)

            keyword_results = store.search("cooling", user_id="gulf")
            self.assertEqual(keyword_results, [])
            hybrid_results = store.hybrid_search("cooling", user_id="gulf", limit=1)
            self.assertEqual(hybrid_results[0]["id"], ac.memory_id)
            self.assertGreater(hybrid_results[0]["embedding_score"], 0)

    def test_hybrid_search_uses_rerank_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(
                Path(tmp),
                embedding_provider=FakeEmbeddingProvider(),
                rerank_provider=FakeRerankProvider(),
            )
            store.init("gulf")
            tea = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="tea memory",
                text="User likes green tea in the afternoon.",
                tags=["drink"],
            )
            cooling = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="cooling memory",
                text="User prefers cooling when the room is hot.",
                tags=["temperature"],
            )
            store.add_memory(tea)
            store.add_memory(cooling)

            results = store.hybrid_search("tea hot room", user_id="gulf", limit=2)

            self.assertEqual(results[0]["id"], cooling.memory_id)
            self.assertGreater(results[0]["rerank_score"], results[1]["rerank_score"])

    def test_memory_merge_ops_follow_schema_semantics(self) -> None:
        patch = merge_op_factory("patch")
        self.assertEqual(
            patch.apply("User prefers quiet reminders.", "User prefers brief reminders."),
            "User prefers quiet reminders.\n\nUser prefers brief reminders.\n",
        )
        self.assertEqual(
            patch.apply(
                "User prefers quiet reminders.",
                {"blocks": [{"search": "quiet", "replace": "brief"}]},
            ),
            "User prefers brief reminders.",
        )
        immutable = merge_op_factory("immutable")
        self.assertEqual(immutable.apply("ask_before_action", "auto_action"), "ask_before_action")

    def test_memory_schema_loads_from_yaml_templates(self) -> None:
        habit = memory_type_spec("habit")
        self.assertEqual(habit.directory, "habits")
        self.assertEqual(habit.operation_mode, "evidence_then_aggregate")
        self.assertIn("{{ strongest_memories }}", habit.overview_template)

    def test_memory_operation_updater_preserves_page_id_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            profile = store.upsert_profile("gulf", "用户偏好安静的提醒方式。", mode="replace")
            service = MemoryUpdateService(store)
            operation = MemoryOperation(
                action="add",
                memory_type="preference",
                title="quiet reminder preference",
                text="User prefers quiet reminders.",
                tags=["preference"],
                page_id=101,
                links=[
                    {
                        "to": profile["uri"],
                        "link_type": "related_to",
                        "description": "Preference belongs to the user profile.",
                    }
                ],
            )
            diff = service.apply(
                [operation],
                MemoryUpdateContext(user_id="gulf", source="test", diff_id="link-test"),
            )
            created_uri = diff["operations"]["adds"][0]["uri"]
            created = store.resolve_memory(created_uri, "gulf")
            target = store.resolve_memory(profile["uri"], "gulf")
            self.assertEqual(created["page_id"], 101)
            self.assertEqual(created["links"][0]["to"], profile["uri"])
            self.assertEqual(target["backlinks"][0]["from"], profile["uri"])
            self.assertEqual(target["backlinks"][0]["to"], created_uri)

    def test_reinforcement_policy_ledger_updates_predicted_and_actual_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user" / "gulf" / "rl" / "policy_ledger.json"
            ledger = ReinforcementPolicyLedger(path)
            state = ledger.build_state(
                scene="user is sweating in a hot room",
                context_tags=["room", "temperature_hot"],
                memories=[],
                behavior_patterns=[],
            )
            scores = ledger.action_scores(state, ["smoke", "open_ac"])
            ledger.record_prediction(
                user_id="gulf",
                episode_id="ep-rl",
                state=state,
                candidates=[],
                selected_action="smoke",
                intervention_action="warn_no_smoking",
                action_scores=scores,
            )

            update = ledger.record_feedback(
                episode_id="ep-rl",
                predicted_action="smoke",
                actual_action="open_ac",
                reward=-1.0,
            )

            self.assertTrue(update["updated"])
            self.assertFalse(update["predicted_success"])
            self.assertLess(update["predicted_action_value"]["normalized_value"], 0.5)
            self.assertGreater(update["actual_action_value"]["normalized_value"], 0.5)

    def test_reinforcement_policy_state_ignores_retrieval_volatility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ReinforcementPolicyLedger(Path(tmp) / "policy_ledger.json")
            first = ledger.build_state(
                scene="用户回到房间，室温 30 度，出汗，说热。",
                context_tags=["room", "arrive_home", "very_hot", "sweating", "says_hot"],
                memories=[{"type": "habit", "path": "user/gulf/habits/a.md"}],
                behavior_patterns=[{"group_id": "hot-room-a", "action": "open_ac"}],
            )
            second = ledger.build_state(
                scene="用户回到房间，室温 30 度，出汗，说热。",
                context_tags=["room", "arrive_home", "very_hot", "sweating", "says_hot"],
                memories=[{"type": "case", "path": "user/gulf/cases/b.md"}],
                behavior_patterns=[{"group_id": "hot-room-b", "action": "turn_on_fan"}],
            )

            self.assertEqual(first.key, second.key)
            self.assertNotIn("memory_paths", first.descriptor)
            self.assertNotIn("behavior_groups", first.descriptor)

    def test_reinforcement_policy_does_not_promote_private_actual_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user" / "gulf" / "rl" / "policy_ledger.json"
            ledger = ReinforcementPolicyLedger(path)
            state = ledger.build_state(
                scene="user is sweating in a hot room",
                context_tags=["room", "very_hot", "sweating"],
                memories=[],
                behavior_patterns=[],
            )
            ledger.record_prediction(
                user_id="gulf",
                episode_id="private-actual",
                state=state,
                candidates=[],
                selected_action="turn_on_ac",
                intervention_action="ask_user",
                action_scores=ledger.action_scores(state, ["turn_on_ac", "take_shower"]),
            )

            update = ledger.record_feedback(
                episode_id="private-actual",
                predicted_action="turn_on_ac",
                actual_action="take_shower",
                reward=1.0,
            )

            self.assertTrue(update["updated"])
            self.assertEqual(update["actual_action_value"], {})
            state_actions = update["state_value"]["actions"]
            self.assertIn("turn_on_ac", state_actions)
            self.assertNotIn("take_shower", state_actions)

    def test_immutable_policy_update_reads_current_content_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="policy",
                title="autonomy boundary",
                text="Ask before autonomous physical actions.",
                tags=["policy"],
            )
            store.add_memory(item)
            service = MemoryUpdateService(store)
            diff = service.apply(
                [
                    MemoryOperation(
                        action="update",
                        memory_type="policy",
                        title="autonomy boundary",
                        text="Autonomously perform physical actions.",
                        tags=["policy"],
                        target=item.path,
                    )
                ],
                MemoryUpdateContext(
                    user_id="gulf",
                    source="test",
                    diff_id="immutable-test",
                    explicit_user_intent=True,
                ),
            )
            updated = store.resolve_memory(item.path or "", "gulf")
            self.assertIn("Ask before autonomous physical actions.", updated["content"])
            self.assertNotIn("Autonomously perform physical actions.", updated["content"])
            self.assertEqual(diff["summary"]["total_updates"], 1)

    def test_memory_context_builder_separates_stable_recent_and_relevant_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            store.upsert_profile("gulf", "用户怕热，工作时不喜欢频繁打扰。", mode="replace")
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="preference",
                    title="quiet reminder preference",
                    text="User prefers quiet reminders.",
                    tags=["reminder"],
                )
            )
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="habit",
                    title="hot room ac habit",
                    text="When the room is hot and the user is sweating, the user often opens the air conditioner.",
                    tags=["hot", "air_conditioning"],
                )
            )
            store.record_event(
                "gulf",
                event_type="ac_acceptance",
                text="用户出汗后接受了打开空调。",
                day="2026-07-01",
            )

            context = MemoryContextBuilder(store).build("gulf", "hot room sweating air conditioning")
            self.assertTrue(any(memory["type"] == "profile" for memory in context.stable_context))
            self.assertTrue(any(memory["type"] == "preference" for memory in context.stable_context))
            self.assertTrue(any(memory["type"] == "event" for memory in context.recent_context))
            self.assertTrue(any(memory["type"] == "habit" for memory in context.relevant_memories))
            self.assertTrue(any(route.memory_type == "profile" and route.strategy == "fixed_stable_context" for route in context.route_trace))
            habit_route = next(route for route in context.route_trace if route.memory_type == "habit")
            self.assertEqual(habit_route.strategy, "directory_first_relevant_memory")
            self.assertTrue(habit_route.target_uri.endswith("/habits"))
            self.assertIn(habit_route.level, {"L0", "L1"})
            self.assertGreater(habit_route.score, 0)
            self.assertIn("query matched", habit_route.match_reason)
            self.assertIn("habit", context.source_summary())
            self.assertIn("Stable context:", context.digest)
            self.assertIn("Relevant memories:", context.digest)
            self.assertIn("Memory route trace:", context.digest)
            directory_hits = store.rank_directory_layers(
                "hot room sweating air conditioning",
                user_id="gulf",
                memory_types={"habit"},
                limit=1,
            )
            self.assertEqual(directory_hits[0]["type"], "habit")
            self.assertIn(directory_hits[0]["level"], {"L0", "L1"})

    def test_episode_process_stores_memory_predicts_and_records_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="habit",
                    title="computer desk habit trigger",
                    text="用户坐在电脑前很久时，可能进入某个下一步行为。",
                    tags=["action:smoke", "need:habit_trigger", "computer_desk"],
                )
            )
            episode = EpisodeProcessor(store)
            result = episode.process(
                user_id="gulf",
                episode_id="ep-1",
                scene="用户坐在电脑前很久，手边有烟盒。",
                messages=[
                    {
                        "role": "observation",
                        "text": "记住：用户在电脑前久坐后可能想抽烟。",
                    }
                ],
                available_actions=["remind_no_smoking", "ask_user", "do_nothing"],
            )

            self.assertEqual(result["memory_diff"]["summary"]["total_adds"], 1)
            self.assertEqual(result["prediction"]["predicted_action"], "smoke")
            self.assertEqual(result["prediction"]["recommended_intervention"], "remind_no_smoking")
            self.assertEqual(result["intervention"]["action"], "remind_no_smoking")
            self.assertGreaterEqual(len(result["ranked_candidates"]), 2)
            self.assertEqual(result["ranked_candidates"][0]["action"], "smoke")
            self.assertNotIn("recommended_intervention", result["ranked_candidates"][0])
            self.assertGreater(result["ranked_candidates"][0]["prior"], 0)
            self.assertIn("memory_habit", result["ranked_candidates"][0]["sources"])
            self.assertGreaterEqual(len(result["ranked_candidates"][0]["evidence"]), 1)
            self.assertIn("memory_support", result["ranked_candidates"][0]["features"])
            self.assertIn("memory_context", result)
            self.assertIn("retrieval", result)
            self.assertIn("source_summary", result["retrieval"])
            self.assertEqual(result["retrieval"]["query_plan"]["mode"], "directory_first_memory_then_behavior")
            self.assertIn("behavior_routes", result["retrieval"]["query_plan"])
            self.assertIn("behavior_context", result["retrieval"])
            self.assertTrue(
                any(
                    route["strategy"] == "hierarchical_behavior_pattern"
                    for route in result["retrieval"]["behavior_context"]["route_trace"]
                )
            )
            self.assertIn("stable_context", result["memory_context"])
            self.assertIn("relevant_memories", result["memory_context"])
            self.assertIn("route_trace", result["memory_context"])
            self.assertIn("behavior_patterns", result["retrieval"])
            self.assertTrue((Path(tmp) / "user" / "gulf" / "episodes" / "ep-1" / "episode_result.json").exists())

            feedback = episode.record_feedback(
                user_id="gulf",
                episode_id="ep-1",
                feedback="prediction_wrong",
                reward=-1,
                actual_action="organize_desk",
                correction="用户只是整理桌面，不是准备抽烟。",
            )
            self.assertEqual(feedback["reward"], -1.0)
            self.assertEqual(feedback["actual_action"], "organize_desk")
            self.assertEqual(feedback["learning_status"], "queued")
            self.assertFalse(feedback["corrects_memory"])
            self.assertTrue((Path(tmp) / "user" / "gulf" / "episodes" / "ep-1" / "feedback.jsonl").exists())
            worker_result = FeedbackWorker(store).process_pending("gulf")
            learning = worker_result["results"][0]["learning_result"]
            self.assertIn("behavior_update", learning)
            self.assertLess(learning["behavior_update"]["behavior_reward"], 0)
            self.assertIn("policy_update", learning)
            self.assertTrue((Path(tmp) / "user" / "gulf" / "behavior_stats.json").exists())
            self.assertTrue((Path(tmp) / "user" / "gulf" / "policy_stats.json").exists())
            self.assertEqual(store.search("整理桌面", user_id="gulf"), [])

            follow_up = episode.process(
                user_id="gulf",
                episode_id="ep-2",
                scene="用户坐在电脑前很久，手边有烟盒。",
                available_actions=["remind_no_smoking", "ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )
            smoke_candidate = next(candidate for candidate in follow_up["ranked_candidates"] if candidate["action"] == "smoke")
            self.assertNotIn("policy_reward", smoke_candidate["features"])
            self.assertEqual(smoke_candidate["features"]["behavior_reward"], 0.0)
            self.assertTrue(any(item["action"] == "organize_desk" for item in follow_up["behavior_distribution"]))
            self.assertIn("behavior_feedback", follow_up["retrieval"]["source_summary"])
            organize_candidate = next(candidate for candidate in follow_up["ranked_candidates"] if candidate["action"] == "organize_desk")
            self.assertIn("memory_case", organize_candidate["sources"])
            self.assertFalse(any("behavior_feedback" in candidate["sources"] for candidate in follow_up["ranked_candidates"]))
            self.assertNotEqual(follow_up["intervention"]["action"], "remind_no_smoking")
            self.assertGreaterEqual(len(smoke_candidate["memory_evidence"]), 1)
            first_evidence = smoke_candidate["memory_evidence"][0]
            self.assertIn("retrieval_weight", first_evidence)
            self.assertIn("usage_weight", first_evidence)
            self.assertIn("combined_weight", first_evidence)
            self.assertGreater(first_evidence["usage_weight"], 0)

    def test_episode_observation_uses_two_stage_memory_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store, extractor=FakeObservationExtractor())

            result = episode.process(
                user_id="gulf",
                episode_id="ep-two-stage",
                scene="用户回家后出汗，说有点热。",
                available_actions=["ask_user", "do_nothing"],
            )

            self.assertEqual(result["episode_status"], "predicted")
            self.assertEqual(result["episode_log_timing"], "before_prediction")
            self.assertEqual(result["memory_commit_timing"], "after_feedback")
            self.assertEqual(result["memory_diff"]["summary"]["total_adds"], 0)
            self.assertEqual(len(result["pending_memory_operations"]), 1)
            self.assertEqual(store.search("rawhot123", user_id="gulf"), [])
            self.assertTrue((Path(tmp) / "user" / "gulf" / "episodes" / "ep-two-stage" / "episode_log.json").exists())

            episode.record_feedback(
                user_id="gulf",
                episode_id="ep-two-stage",
                feedback="accepted",
                reward=1,
                actual_action="open_ac",
                action_params={"target_temperature": 25, "mode": "cool"},
                spontaneity="after_prompt",
            )
            queued = json.loads(
                (Path(tmp) / "user" / "gulf" / "episodes" / "ep-two-stage" / "episode_result.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(queued["episode_status"], "feedback_queued")
            worker_result = FeedbackWorker(store).process_pending("gulf")
            learning = worker_result["results"][0]["learning_result"]
            closed = json.loads(
                (Path(tmp) / "user" / "gulf" / "episodes" / "ep-two-stage" / "episode_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(closed["episode_status"], "closed_with_feedback")
            self.assertEqual(closed["actual_action"], "open_ac")
            self.assertEqual(closed["action_params"]["target_temperature"], 25)
            self.assertIn("case_memory", learning)
            self.assertEqual(store.search("rawhot123", user_id="gulf"), [])
            cases = store.search("Actual action: open_ac", user_id="gulf", memory_type="case")
            self.assertGreaterEqual(len(cases), 1)
            self.assertIn("Predicted candidates:", cases[0]["content"])
            self.assertIn('"target_temperature": 25', cases[0]["content"])

    def test_repeated_feedback_promotes_behavior_pattern_to_habit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            for index in range(3):
                observed_at = (datetime(2026, 7, 1, 19, 0, tzinfo=timezone.utc) + timedelta(days=index)).isoformat()
                episode_id = f"hot-room-promote-{index}"
                episode.process(
                    user_id="gulf",
                    episode_id=episode_id,
                    observation={
                        "raw_text": "用户回到房间，出汗，说有点热。",
                        "location": "room",
                        "activity": "arrive_home",
                        "observed_at": observed_at,
                        "signals": ["hot", "sweating"],
                        "environment": {"temperature": 30},
                    },
                    available_actions=["ask_user", "do_nothing"],
                )
                episode.record_feedback(
                    user_id="gulf",
                    episode_id=episode_id,
                    feedback="accepted",
                    reward=1,
                    actual_action="open_ac",
                )

            worker_result = FeedbackWorker(store).process_pending("gulf")
            learning = worker_result["results"][-1]["learning_result"]
            self.assertTrue(learning["memory_consolidation"]["promoted"])
            promoted = store.search("open_ac promoted_behavior_pattern", user_id="gulf", memory_type="habit")
            self.assertTrue(any("promoted_behavior_pattern" in " ".join(row["tags"]) for row in promoted))

    def test_memory_weight_temporal_scopes_decay_differently(self) -> None:
        now = datetime(2026, 7, 1, tzinfo=timezone.utc)
        recent_hot_weather_trigger = {
            "type": "trigger",
            "temporal_scope": "rolling_7d",
            "base_weight": 0.82,
            "confidence": 0.9,
            "evidence_count": 6,
            "positive_count": 5,
            "negative_count": 1,
            "updated_at": (now - timedelta(days=2)).isoformat(),
        }
        stale_hot_weather_trigger = {
            **recent_hot_weather_trigger,
            "updated_at": (now - timedelta(days=30)).isoformat(),
        }
        stable_profile = {
            "type": "profile",
            "temporal_scope": "stable",
            "base_weight": 0.95,
            "confidence": 0.9,
            "evidence_count": 1,
            "positive_count": 1,
            "negative_count": 0,
            "updated_at": (now - timedelta(days=365)).isoformat(),
        }

        recent = score_memory_weight(recent_hot_weather_trigger, now=now)
        stale = score_memory_weight(stale_hot_weather_trigger, now=now)
        profile = score_memory_weight(stable_profile, now=now)

        self.assertGreater(recent.effective_weight, stale.effective_weight)
        self.assertEqual(profile.temporal_weight, 1.0)

    def test_behavior_stats_generalizes_across_semantic_scene_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = BehaviorStats(Path(tmp) / "behavior_stats.json")
            stats.record(
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"],
                predicted_action="seek_cooling",
                actual_action="open_ac",
                reward=1,
            )

            distribution = stats.distribution_for_scene(
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_31 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_31", "hot_environment"],
            )

            open_ac = next(item for item in distribution if item["action"] == "open_ac")
            self.assertEqual(open_ac["match_level"], "semantic")
            self.assertEqual(open_ac["match_weight"], 0.7)
            self.assertGreater(open_ac["weighted_behavior_reward"], 0)
            self.assertLess(open_ac["weighted_behavior_reward"], open_ac["behavior_reward_score"])

    def test_behavior_pattern_retrieval_generalizes_across_semantic_scene_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BehaviorPatternStore(Path(tmp))
            created_at = datetime(2026, 7, 1, 19, 0, tzinfo=timezone.utc).isoformat()
            store.record(
                user_id="gulf",
                episode_id="hot-room-episode",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"],
                predicted_action="seek_cooling",
                actual_action="open_ac",
                reward=1,
                created_at=created_at,
                predicted_candidates=[{"action": "open_ac", "score": 0.7}, {"action": "turn_on_fan", "score": 0.2}],
                action_params={"target_temperature": 25, "mode": "cool"},
                scene_features={"location": "room", "time_bucket": "evening", "thermal_level": "very_hot"},
                spontaneity="after_prompt",
            )

            distribution = store.distribution_for_scene(
                user_id="gulf",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_31 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_31", "hot_environment"],
            )

            self.assertEqual(distribution[0]["action"], "open_ac")
            self.assertEqual(distribution[0]["source"], "behavior_pattern")
            self.assertEqual(distribution[0]["match_level"], "semantic")
            self.assertEqual(distribution[0]["match_weight"], 0.7)
            self.assertLess(distribution[0]["prediction_coefficient"], distribution[0]["evidence_confidence"])
            self.assertEqual(distribution[0]["action_distribution"][0]["action"], "open_ac")
            self.assertEqual(distribution[0]["action_distribution"][0]["param_distribution"]["target_temperature"]["25"], 1.0)

    def test_behavior_pattern_group_distribution_keeps_multiple_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BehaviorPatternStore(Path(tmp))
            scene = "用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment"
            tags = ["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"]
            actions = [
                ("open_ac", 1.0, {"target_temperature": 25}),
                ("open_ac", 1.0, {"target_temperature": 26}),
                ("turn_on_fan", 0.6, {"fan_speed": "auto"}),
                ("drink_water", -0.2, {}),
            ]
            for index, (action, reward, params) in enumerate(actions):
                store.record(
                    user_id="gulf",
                    episode_id=f"multi-action-{index}",
                    retrieval_query=scene,
                    context_tags=tags,
                    predicted_action="seek_cooling",
                    actual_action=action,
                    reward=reward,
                    created_at=(datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                    action_params=params,
                    predicted_candidates=[{"action": "open_ac", "score": 0.6}, {"action": "turn_on_fan", "score": 0.3}],
                )

            distribution = store.distribution_for_scene(
                user_id="gulf",
                retrieval_query=scene,
                context_tags=tags,
            )
            top = distribution[0]
            actions_by_name = {item["action"]: item for item in top["action_distribution"]}

            self.assertIn("open_ac", actions_by_name)
            self.assertIn("turn_on_fan", actions_by_name)
            self.assertIn("drink_water", actions_by_name)
            self.assertAlmostEqual(actions_by_name["open_ac"]["probability"], 0.5)
            self.assertEqual(actions_by_name["open_ac"]["param_distribution"]["target_temperature"]["25"], 0.5)
            self.assertEqual(actions_by_name["drink_water"]["negative_count"], 1)

            candidates = EpisodeProcessor(MemoryStore(Path(tmp))).predictor.generator.generate(
                scene,
                memories=[],
                behavior_patterns=distribution,
            )
            candidate_actions = {candidate.action for candidate in candidates}
            self.assertIn("turn_on_ac", candidate_actions)
            self.assertIn("turn_on_fan", candidate_actions)

    def test_ranker_uses_behavior_param_distribution(self) -> None:
        ranker = CandidateRanker()
        candidate = Candidate(action="turn_on_ac", need="cool_down", prior=0.5, sources=["behavior_pattern"])
        with_params = ranker.rank(
            scene="hot room",
            memories=[],
            candidates=[candidate],
            behavior_distribution=[
                {
                    "action": "turn_on_ac",
                    "probability": 0.7,
                    "avg_reward": 0.8,
                    "confidence": 0.8,
                    "recency_weight": 0.8,
                    "param_distribution": {"target_temperature": {"25": 0.8, "26": 0.2}},
                }
            ],
        )[0]
        without_params = ranker.rank(
            scene="hot room",
            memories=[],
            candidates=[Candidate(action="turn_on_ac", need="cool_down", prior=0.5, sources=["behavior_pattern"])],
            behavior_distribution=[
                {
                    "action": "turn_on_ac",
                    "probability": 0.7,
                    "avg_reward": 0.8,
                    "confidence": 0.8,
                    "recency_weight": 0.8,
                }
            ],
        )[0]

        self.assertGreater(with_params.features["behavior_reward"], without_params.features["behavior_reward"])
        self.assertGreater(with_params.score, without_params.score)

    def test_storage_plan_matches_memory_type_semantics(self) -> None:
        plans = {plan.memory_type: plan for plan in all_memory_storage_plans()}
        self.assertIn("profile", plans)
        self.assertIn("event", plans)
        self.assertEqual(plans["profile"].directory, "profile")
        self.assertEqual(plans["profile"].default_temporal_scope, "stable")
        self.assertEqual(plans["event"].update_mode, "append_only")
        self.assertEqual(plans["trigger"].default_temporal_scope, "rolling_7d")
        self.assertEqual(plans["trigger"].update_policy, "aggregate_from_evidence")
        self.assertGreater(plans["profile"].default_base_weight, plans["event"].default_base_weight)
        self.assertEqual(memory_storage_plan("habit").prediction_role, "candidate_behavior_prior")

    def test_update_policy_guards_memory_type_updates(self) -> None:
        self.assertTrue(update_policy("event").append_only)
        action, reason = normalize_operation_for_policy("update", "event")
        self.assertEqual(action, "add")
        self.assertIn("append-only", reason or "")

        action, reason = normalize_operation_for_policy("add", "policy")
        self.assertEqual(action, "ignore")
        self.assertIn("explicit user intent", reason or "")

        action, reason = normalize_operation_for_policy("add", "policy", explicit_user_intent=True)
        self.assertEqual(action, "add")
        self.assertIsNone(reason)

        action, reason = normalize_operation_for_policy("add", "trigger")
        self.assertEqual(action, "add")
        self.assertIn("evidence-first", reason or "")

    def test_memory_update_service_aggregates_habit_after_cross_day_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            service = MemoryUpdateService(store, min_evidence_days=3)
            operation = MemoryOperation(
                action="add",
                memory_type="habit",
                title="hot room cooling habit",
                text="When the room is hot and the user is sweating, the user tends to open the air conditioner.",
                tags=["hot", "cooling"],
                confidence=0.8,
            )

            for day in ["2026-07-01", "2026-07-02"]:
                diff = service.apply(
                    [operation],
                    MemoryUpdateContext(user_id="gulf", source=f"test:{day}", diff_id=day, day=day),
                )
                self.assertEqual(diff["summary"]["total_adds"], 1)
                self.assertEqual(store.list_by_type("gulf", "habit"), [])

            diff = service.apply(
                [operation],
                MemoryUpdateContext(user_id="gulf", source="test:2026-07-03", diff_id="2026-07-03", day="2026-07-03"),
            )
            self.assertEqual(diff["summary"]["total_adds"], 2)
            habits = store.list_by_type("gulf", "habit")
            self.assertEqual(len(habits), 1)
            self.assertEqual(habits[0]["evidence_count"], 3)
            self.assertIn("Distinct days: 3", habits[0]["content"])

    def test_episode_observed_at_day_drives_habit_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            for index, day in enumerate(["2026-07-01", "2026-07-02", "2026-07-03"]):
                result = episode.process(
                    user_id="gulf",
                    episode_id=f"habit-day-{index}",
                    observation=ObservationContext(
                        raw_text="用户回到房间，说热并出汗。",
                        location="room",
                        observed_at=f"{day}T18:00:00+00:00",
                        signals=["hot", "sweating"],
                    ),
                    messages=[
                        {
                            "role": "observation",
                            "text": "记住：用户一般在回房间说热并出汗后打开空调。",
                        }
                    ],
                    available_actions=["turn_on_ac", "do_nothing"],
                )
            self.assertEqual(result["memory_diff"]["summary"]["total_adds"], 2)
            habits = store.list_by_type("gulf", "habit")
            self.assertEqual(len(habits), 1)
            self.assertEqual(habits[0]["evidence_count"], 3)

    def test_memory_update_service_patches_existing_preference_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            service = MemoryUpdateService(store)
            first = MemoryOperation(
                action="add",
                memory_type="preference",
                title="temperature preference",
                text="User prefers air conditioning at 25 degrees.",
                tags=["temperature", "air_conditioning"],
            )
            second = MemoryOperation(
                action="add",
                memory_type="preference",
                title="ac temperature preference",
                text="User prefers air conditioning between 24 and 26 degrees.",
                tags=["temperature", "air_conditioning"],
            )
            service.apply([first], MemoryUpdateContext(user_id="gulf", source="test", diff_id="pref-1"))
            diff = service.apply([second], MemoryUpdateContext(user_id="gulf", source="test", diff_id="pref-2"))
            self.assertEqual(diff["summary"]["total_updates"], 1)
            preferences = store.list_by_type("gulf", "preference")
            self.assertEqual(len(preferences), 1)
            self.assertIn("24 and 26", preferences[0]["content"])
            self.assertEqual(preferences[0]["evidence_count"], 2)

    def test_memory_update_service_aggregates_feedback_without_rewriting_raw_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            service = MemoryUpdateService(store)
            operation = MemoryOperation(
                action="add",
                memory_type="feedback",
                title="cooling feedback",
                text="User accepted cooling suggestion.",
                tags=["cooling", "accepted"],
            )
            for index in range(3):
                diff = service.apply(
                    [operation],
                    MemoryUpdateContext(user_id="gulf", source=f"feedback:{index}", diff_id=f"feedback-{index}"),
                )
            self.assertEqual(diff["summary"]["total_adds"], 2)
            feedback = store.list_by_type("gulf", "feedback", limit=10)
            raw = [item for item in feedback if "aggregated" not in item["tags"]]
            aggregated = [item for item in feedback if "aggregated" in item["tags"]]
            self.assertEqual(len(raw), 3)
            self.assertEqual(len(aggregated), 1)
            self.assertEqual(aggregated[0]["evidence_count"], 3)

    def test_feedback_creates_case_memory_for_actual_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            episode.process(
                user_id="gulf",
                episode_id="case-episode",
                scene="用户回到房间，说热并出汗。",
                available_actions=["turn_on_ac", "do_nothing"],
                memory_write_timing="deferred",
            )
            feedback = episode.record_feedback(
                user_id="gulf",
                episode_id="case-episode",
                feedback="actual_action_observed",
                reward=1,
                actual_action="open_ac",
            )
            self.assertEqual(feedback["learning_status"], "queued")
            learning = FeedbackWorker(store).process_pending("gulf")["results"][0]["learning_result"]
            self.assertIn("case_memory", learning)
            cases = store.list_by_type("gulf", "case")
            self.assertEqual(len(cases), 1)
            self.assertIn("actual_action:open_ac", cases[0]["tags"])

    def test_archive_cold_memories_moves_low_hotness_events_to_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
            item = MemoryItem(
                user_id="gulf",
                memory_type="event",
                title="old event",
                text="Old low value event.",
                tags=["old"],
                created_at=old,
                updated_at=old,
            )
            store.init("gulf")
            store.add_memory(item)
            result = store.archive_cold_memories("gulf", limit=10, max_hotness=0.12)
            self.assertEqual(result["summary"]["total_archived"], 1)
            self.assertFalse((Path(tmp) / (item.path or "")).exists())
            self.assertTrue((Path(tmp) / result["archive_path"]).exists())

    def test_episode_can_defer_memory_writes_until_after_prediction_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            store.add_memory(
                MemoryItem(
                    user_id="gulf",
                    memory_type="habit",
                    title="computer desk habit trigger",
                    text="用户坐在电脑前很久时，可能进入某个下一步行为。",
                    tags=["action:smoke", "need:habit_trigger", "computer_desk"],
                )
            )
            episode = EpisodeProcessor(store)
            result = episode.process(
                user_id="gulf",
                episode_id="ep-deferred",
                scene="用户坐在电脑前，手边有烟盒。",
                messages=[
                    {
                        "role": "observation",
                        "text": "记住：用户在电脑前久坐后可能想抽烟。",
                    }
                ],
                available_actions=["remind_no_smoking", "do_nothing"],
                memory_write_timing="deferred",
            )

            self.assertEqual(result["prediction"]["predicted_action"], "smoke")
            self.assertEqual(result["memory_diff"]["summary"]["total_adds"], 0)
            self.assertEqual(len(result["pending_memory_operations"]), 1)
            self.assertEqual(store.search("久坐 抽烟", user_id="gulf"), [])

            diff = episode.commit_pending_memory("gulf", "ep-deferred")
            self.assertEqual(diff["summary"]["total_adds"], 1)
            self.assertGreaterEqual(len(store.search("久坐 抽烟", user_id="gulf")), 1)

    def test_candidate_generation_prefers_behavior_pattern_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            pattern_store = BehaviorPatternStore(Path(tmp))
            scene = "用户回到房间，说感觉很热，并且在出汗。"
            for index in range(3):
                created_at = (datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat()
                pattern_store.record(
                    user_id="gulf",
                    episode_id=f"hot-room-{index}",
                    retrieval_query=scene,
                    context_tags=["room", "hot", "sweating"],
                    predicted_action="seek_cooling",
                    actual_action="open_ac",
                    reward=1,
                    created_at=created_at,
                )

            current = episode.process(
                user_id="gulf",
                episode_id="hot-room-current",
                scene="用户回到房间，说热，额头出汗。",
                available_actions=["turn_on_ac", "ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )

            self.assertEqual(current["ranked_candidates"][0]["action"], "turn_on_ac")
            self.assertIn("behavior_pattern", current["ranked_candidates"][0]["sources"])
            self.assertGreaterEqual(current["behavior_patterns"][0]["distinct_days"], 3)
            self.assertIn("evidence_confidence", current["behavior_patterns"][0])
            self.assertGreater(current["behavior_patterns"][0]["prediction_coefficient"], 0)

    def test_behavior_pattern_store_builds_l1_pattern_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = BehaviorPatternStore(root)
            query = "用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment"
            tags = ["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"]
            for index in range(3):
                store.record(
                    user_id="gulf",
                    episode_id=f"pattern-ep-{index}",
                    retrieval_query=query,
                    context_tags=tags,
                    predicted_action="unknown",
                    actual_action="open_ac",
                    reward=1,
                    created_at=(datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                )

            distribution = store.distribution_for_scene(
                user_id="gulf",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_31 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_31", "hot_environment"],
            )

            self.assertEqual(distribution[0]["source"], "behavior_pattern")
            self.assertEqual(distribution[0]["action"], "open_ac")
            self.assertGreater(distribution[0]["evidence_confidence"], 0.45)
            self.assertTrue((root / "user" / "gulf" / "behavior" / ".overview.md").exists())
            self.assertTrue((root / distribution[0]["pattern_uri"]).exists())

    def test_behavior_pattern_group_tracks_competing_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = BehaviorPatternStore(root)
            query = "用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment"
            tags = ["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"]
            for index in range(3):
                store.record(
                    user_id="gulf",
                    episode_id=f"group-ac-{index}",
                    retrieval_query=query,
                    context_tags=tags,
                    predicted_action="unknown",
                    actual_action="open_ac",
                    reward=1,
                    created_at=(datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                )
            for index in range(2):
                store.record(
                    user_id="gulf",
                    episode_id=f"group-window-{index}",
                    retrieval_query=query,
                    context_tags=tags,
                    predicted_action="unknown",
                    actual_action="open_window",
                    reward=1,
                    created_at=(datetime(2026, 7, 4, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                )

            distribution = store.distribution_for_scene(
                user_id="gulf",
                retrieval_query=query,
                context_tags=tags,
            )
            group_path = root / distribution[0]["group_uri"]
            group = json.loads(group_path.read_text(encoding="utf-8"))
            actions = {item["action"]: item for item in group["action_distribution"]}

            self.assertEqual(group["total_samples"], 5)
            self.assertEqual(group["top_action"], "open_ac")
            self.assertLess(group["top_action_margin"], 1.0)
            self.assertGreater(group["group_entropy"], 0.0)
            self.assertAlmostEqual(actions["open_ac"]["ratio"], 0.6)
            self.assertAlmostEqual(actions["open_window"]["ratio"], 0.4)
            self.assertLess(next(item for item in distribution if item["action"] == "open_window")["evidence_confidence"], 0.45)

    def test_behavior_pattern_approx_merge_prevents_scene_fragmentation_and_writes_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = BehaviorPatternStore(root)
            base_tags = ["room", "arrive_home", "hot", "sweating", "temperature_30", "hot_environment"]
            store.record(
                user_id="gulf",
                episode_id="merge-30",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_30 hot_environment",
                context_tags=base_tags,
                predicted_action="unknown",
                actual_action="open_ac",
                reward=1,
                created_at="2026-07-01T00:00:00+00:00",
            )
            store.record(
                user_id="gulf",
                episode_id="merge-31",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_31 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_31", "hot_environment"],
                predicted_action="unknown",
                actual_action="open_ac",
                reward=1,
                created_at="2026-07-02T00:00:00+00:00",
            )

            pattern_files = list((root / "user" / "gulf" / "behavior" / "room" / "patterns").glob("*.json"))
            distribution = store.distribution_for_scene(
                user_id="gulf",
                retrieval_query="用户回到房间 说热 出汗 room arrive_home hot sweating temperature_32 hot_environment",
                context_tags=["room", "arrive_home", "hot", "sweating", "temperature_32", "hot_environment"],
            )

            self.assertEqual(len(pattern_files), 1)
            self.assertTrue((root / "user" / "gulf" / "behavior" / ".pattern_index.sqlite").exists())
            self.assertEqual(distribution[0]["sample_count"], 2)
            self.assertEqual(distribution[0]["source"], "behavior_pattern")

    def test_behavior_pattern_compacts_old_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = BehaviorPatternStore(root, active_evidence_limit=3)
            query = "用户回到房间 说热 出汗 room hot sweating"
            tags = ["room", "hot", "sweating"]
            for index in range(5):
                store.record(
                    user_id="gulf",
                    episode_id=f"compact-{index}",
                    retrieval_query=query,
                    context_tags=tags,
                    predicted_action="unknown",
                    actual_action="open_ac",
                    reward=1,
                    created_at=(datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                )

            distribution = store.distribution_for_scene("gulf", query, tags)
            pattern = json.loads((root / distribution[0]["pattern_uri"]).read_text(encoding="utf-8"))

            self.assertEqual(pattern["sample_count"], 5)
            self.assertEqual(len(pattern["episodes"]), 3)
            self.assertEqual(pattern["old_evidence_summary"]["sample_count"], 2)
            self.assertIn("recent_30d_count", pattern)

    def test_same_day_repeated_observations_do_not_create_history_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            pattern_store = BehaviorPatternStore(Path(tmp))
            scene = "用户回到房间，说感觉很热，并且在出汗。"
            for index in range(5):
                created_at = datetime(2026, 7, 1, 10, index, tzinfo=timezone.utc).isoformat()
                pattern_store.record(
                    user_id="gulf",
                    episode_id=f"same-day-hot-room-{index}",
                    retrieval_query=scene,
                    context_tags=["room", "hot", "sweating"],
                    predicted_action="seek_cooling",
                    actual_action="open_ac",
                    reward=1,
                    created_at=created_at,
                )

            current = episode.process(
                user_id="gulf",
                episode_id="same-day-current",
                scene="用户回到房间，说热，额头出汗。",
                available_actions=["turn_on_ac", "ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )

            history_candidate = next(
                (candidate for candidate in current["ranked_candidates"] if "behavior_pattern" in candidate["sources"]),
                None,
            )
            self.assertIsNone(history_candidate)
            self.assertEqual(current["behavior_patterns"][0]["distinct_days"], 1)
            self.assertLess(current["behavior_patterns"][0]["evidence_confidence"], 0.45)

    def test_observation_context_builds_retrieval_query(self) -> None:
        observation = ObservationContext(
            raw_text="User is sitting at the desk and sweating.",
            location="computer_desk",
            activity="computer_work",
            started_at="2026-07-01T18:45:00+00:00",
            observed_at="2026-07-01T19:30:00+00:00",
            signals=["sweating", "says_hot"],
            environment={"temperature": 30, "humidity": 72},
        )

        self.assertEqual(observation.computed_duration_minutes(), 45)
        self.assertEqual(observation.time_of_day(), "evening")
        tags = observation.context_tags()
        self.assertIn("computer_desk", tags)
        self.assertIn("computer_work", tags)
        self.assertIn("duration_30m_plus", tags)
        self.assertIn("hot_environment", tags)
        self.assertIn("humid_environment", tags)
        query = observation.to_retrieval_query()
        self.assertIn("computer_desk", query)
        self.assertIn("sweating", query)

    def test_episode_process_accepts_observation_context_for_behavior_pattern_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            episode = EpisodeProcessor(store)
            pattern_store = BehaviorPatternStore(Path(tmp))
            base_observation = {
                "raw_text": "用户回到房间，说热，额头出汗。",
                "location": "room",
                "activity": "arrive_home",
                "signals": ["hot", "sweating", "says_hot"],
                "environment": {"temperature": 30},
            }
            for index in range(3):
                episode_id = f"structured-hot-room-{index}"
                created_at = (datetime(2026, 7, 1, 19, 0, tzinfo=timezone.utc) + timedelta(days=index)).isoformat()
                stored_observation = ObservationContext(
                    **base_observation,
                    started_at=(datetime(2026, 7, 1, 18, 30, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                    observed_at=created_at,
                )
                pattern_store.record(
                    user_id="gulf",
                    episode_id=episode_id,
                    retrieval_query=stored_observation.to_retrieval_query(),
                    context_tags=stored_observation.context_tags(),
                    predicted_action="seek_cooling",
                    actual_action="open_ac",
                    reward=1,
                    created_at=created_at,
                )

            current = episode.process(
                user_id="gulf",
                episode_id="structured-hot-room-current",
                observation=ObservationContext(
                    **base_observation,
                    started_at="2026-07-04T18:25:00+00:00",
                    observed_at="2026-07-04T19:05:00+00:00",
                ),
                available_actions=["turn_on_ac", "ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )

            self.assertIn("retrieval_query", current)
            self.assertIn("hot_environment", current["context_tags"])
            self.assertEqual(current["observation"]["computed_duration_minutes"], 40)
            self.assertEqual(current["ranked_candidates"][0]["action"], "turn_on_ac")
            self.assertIn("behavior_pattern", current["ranked_candidates"][0]["sources"])

    def test_action_schema_and_reward_split_normalize_aliases(self) -> None:
        self.assertEqual(canonical_action("open_ac"), "turn_on_ac")
        self.assertEqual(action_need("seek_cooling"), "cool_down")

        reward = compute_rewards(
            predicted_action="turn_on_ac",
            actual_action="turn_on_fan",
            user_reward=0.2,
            intervention_action="ask_user",
            intervention_result="accepted",
        )

        self.assertTrue(reward.need_match)
        self.assertFalse(reward.action_match)
        self.assertGreater(reward.behavior_reward, 0)
        self.assertEqual(reward.param_reward, 0.0)
        self.assertEqual(reward.memory_update_signal, "similar_need_different_action")
        self.assertGreaterEqual(reward.intervention_reward, 0.3)

        parameter_reward = compute_rewards(
            predicted_action="open_ac",
            actual_action="turn_on_ac",
            user_reward=0.8,
            intervention_action="ask_user",
            intervention_result="accepted",
            predicted_params={"target_temperature": 25},
            actual_params={"target_temperature": 26},
        )
        self.assertTrue(parameter_reward.action_match)
        self.assertFalse(parameter_reward.param_match)
        self.assertEqual(parameter_reward.param_reward, 0.4)
        self.assertEqual(parameter_reward.memory_update_signal, "parameter_correction")

    def test_permission_engine_blocks_private_behavior_intervention(self) -> None:
        decision = PermissionPolicyEngine().authorize("take_shower", prediction_confidence=0.9)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.max_allowed_action, "do_nothing")

        unknown = PermissionPolicyEngine().authorize("unknown_robot_action", prediction_confidence=0.9)
        self.assertFalse(unknown.allowed)
        self.assertEqual(unknown.max_allowed_action, "do_nothing")

        selected = InterventionSelector().select(
            Candidate(action="take_shower", need="comfort", prior=0.9, score=0.9),
            available_actions=["ask_user", "do_nothing"],
            policy_stats={},
        )

        self.assertEqual(selected.action, "do_nothing")
        self.assertEqual(selected.features["policy_allowed"], 0.0)

    def test_feedback_event_outbox_and_episode_state_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            episode = EpisodeProcessor(store)

            result = episode.process(
                user_id="gulf",
                episode_id="event-flow",
                scene="用户回到房间，说热并出汗。",
                available_actions=["ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )
            self.assertEqual(result["episode_state"], FEEDBACK_PENDING)
            self.assertEqual(result["versions"]["action_schema_version"], "action_schema_v1")

            feedback = episode.record_feedback(
                user_id="gulf",
                episode_id="event-flow",
                feedback="accepted",
                reward=1,
                actual_action="open_ac",
                intervention_result="accepted",
            )

            self.assertEqual(feedback["learning_status"], "queued")
            self.assertEqual(feedback["feedback_event"]["event_type"], "FeedbackRecorded")
            self.assertIn("reward_breakdown", feedback)
            self.assertTrue((root / "user" / "gulf" / "events" / "feedback_events.jsonl").exists())
            self.assertTrue((root / "user" / "gulf" / "events" / "outbox_events.jsonl").exists())

            worker_result = FeedbackWorker(store).process_pending("gulf")
            self.assertEqual(worker_result["processed"], 1)
            self.assertIn("case_memory", worker_result["results"][0]["learning_result"])
            closed = json.loads((root / "user" / "gulf" / "episodes" / "event-flow" / "episode_result.json").read_text())
            self.assertEqual(closed["episode_state"], CLOSED)
            self.assertIn("reward_model_version", closed["versions"])
            self.assertIn(CLOSED, {item["state"] for item in closed["state_history"]})
            audit_events = AuditLogger(root).list_events("gulf")
            audit_types = {event["event_type"] for event in audit_events}
            self.assertIn("episode_predicted", audit_types)
            self.assertIn("feedback_queued", audit_types)
            self.assertIn("feedback_learning_applied", audit_types)

            retry = episode.record_feedback(
                user_id="gulf",
                episode_id="event-flow",
                feedback="accepted",
                reward=1,
                actual_action="open_ac",
                intervention_result="accepted",
            )
            self.assertEqual(retry["learning_status"], "queued")
            retry_worker_result = FeedbackWorker(store).process_pending("gulf")
            self.assertEqual(retry_worker_result["processed"], 0)
            self.assertEqual(len(store.list_by_type("gulf", "case")), 1)

    def test_memory_operation_log_tombstone_and_verify_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            store.init("gulf")
            item = MemoryItem(
                user_id="gulf",
                memory_type="preference",
                title="temporary preference",
                text="User temporarily likes loud alerts.",
                tags=["preference"],
            )
            store.add_memory(item)
            healthy = store.verify_index("gulf")
            self.assertTrue(healthy["healthy"])

            deletion = store.delete_memory(item.path or "", "gulf")

            self.assertIn("tombstone", deletion)
            self.assertTrue((root / deletion["tombstone"]).exists())
            operation_log = root / "user" / "gulf" / "events" / "memory_operations.jsonl"
            self.assertTrue(operation_log.exists())
            operation_text = operation_log.read_text(encoding="utf-8")
            self.assertIn('"operation": "add_memory"', operation_text)
            self.assertIn('"operation": "delete_memory"', operation_text)
            self.assertIn('"status": "committed"', operation_text)
            self.assertTrue(store.verify_index("gulf")["healthy"])

    def test_scene_features_and_stable_signature_are_separate_from_raw_text(self) -> None:
        first = ObservationContext(
            raw_text="用户说太热了。",
            location="room",
            activity="arrive_home",
            signals=["sweating"],
            environment={"temperature": 30.2},
        )
        second = ObservationContext(
            raw_text="换一种说法：房间很热。",
            location="room",
            activity="arrive_home",
            signals=["sweating"],
            environment={"temperature": 30.4},
        )

        first_signature = stable_scene_signature(SceneFeatures.from_observation(first))
        second_signature = stable_scene_signature(SceneFeatures.from_observation(second))

        self.assertEqual(first_signature, second_signature)

    def test_api_facade_worker_replay_reindex_and_markdown_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            health = handle("GET /health", store)
            self.assertEqual(health["status"], "ok")

            episode = handle(
                "POST /episodes",
                store,
                {
                    "user_id": "gulf",
                    "episode_id": "api-flow",
                    "scene": "用户回到房间，说热并出汗。",
                    "available_actions": ["ask_user", "do_nothing"],
                    "memory_write_timing": "deferred",
                },
            )
            self.assertEqual(episode["episode_status"], "predicted")

            feedback = handle(
                "POST /episodes/feedback",
                store,
                {
                    "user_id": "gulf",
                    "episode_id": "api-flow",
                    "feedback": "accepted",
                    "reward": 1,
                    "actual_action": "open_ac",
                    "intervention_result": "accepted",
                },
            )
            self.assertEqual(feedback["learning_status"], "queued")

            processed = handle("POST /workers/feedback", store, {"user_id": "gulf"})
            self.assertEqual(processed["processed"], 1)
            replay = ReplayWorker(store).replay_feedback("gulf")
            self.assertEqual(replay["replayed"], 1)
            self.assertEqual(replay["idempotent"], 1)

            reindexed = ReindexWorker(store).reindex("gulf")
            self.assertEqual(reindexed["status"], "reindexed")
            digest = handle("GET /memory/digest", store, {"user_id": "gulf", "query": "hot room"})
            self.assertIn("<personal-memory", digest["digest"])

            file_path = root / "atomic.md"
            MarkdownStore().write_text_atomic(file_path, "hello")
            self.assertEqual(MarkdownStore().read_text(file_path), "hello")


if __name__ == "__main__":
    unittest.main()
