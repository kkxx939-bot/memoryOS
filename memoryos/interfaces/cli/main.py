from __future__ import annotations

import argparse
import json
from pathlib import Path

from memoryos.domain.memory.memory_item import MemoryItem
from memoryos.application.memory.extractor import JsonLLMMemoryExtractor, RuleBasedExtractor
from memoryos.infrastructure.providers.openai_compatible import (
    build_chat_provider_from_env,
    build_embedding_provider_from_env,
    build_rerank_provider_from_env,
)
from memoryos.infrastructure.providers.embedding_provider import HashingEmbeddingProvider
from memoryos.interfaces.hooks.memory_digest_hook import MemoryHook
from memoryos.application.session.session_manager import SessionManager
from memoryos.infrastructure.repositories.memory_repository import MemoryStore, validate_memory_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Memory OS")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root_user(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--root", default="./memory-root")
        cmd.add_argument("--user", default="gulf")
        cmd.add_argument("--embedding-provider", choices=["auto", "api", "local"], default="auto")
        cmd.add_argument("--rerank-provider", choices=["auto", "api", "off"], default="auto")

    init = sub.add_parser("init")
    add_root_user(init)

    add = sub.add_parser("add-memory")
    add_root_user(add)
    add.add_argument("--type", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--text", required=True)
    add.add_argument("--tags", default="")

    update = sub.add_parser("update-memory")
    add_root_user(update)
    update.add_argument("--id", required=True, help="Memory id or relative path")
    update.add_argument("--title")
    update.add_argument("--text")
    update.add_argument("--tags")

    delete = sub.add_parser("delete-memory")
    add_root_user(delete)
    delete.add_argument("--id", required=True, help="Memory id or relative path")

    merge = sub.add_parser("merge-memory")
    add_root_user(merge)
    merge.add_argument("--target", required=True, help="Target memory id or relative path")
    merge.add_argument("--source", required=True, help="Source memory id or relative path")

    profile = sub.add_parser("update-profile")
    add_root_user(profile)
    profile.add_argument("--text", required=True)
    profile.add_argument("--mode", choices=["append", "replace"], default="append")

    daily = sub.add_parser("update-daily")
    add_root_user(daily)
    daily.add_argument("--text", required=True)
    daily.add_argument("--date")
    daily.add_argument("--mode", choices=["append", "replace"], default="append")

    event = sub.add_parser("record-event")
    add_root_user(event)
    event.add_argument("--event-type", required=True)
    event.add_argument("--text", required=True)
    event.add_argument("--date")
    event.add_argument("--tags", default="")

    search = sub.add_parser("search")
    add_root_user(search)
    search.add_argument("--query", required=True)
    search.add_argument("--type")
    search.add_argument("--limit", type=int, default=8)

    hybrid_search = sub.add_parser("hybrid-search")
    add_root_user(hybrid_search)
    hybrid_search.add_argument("--query", required=True)
    hybrid_search.add_argument("--type")
    hybrid_search.add_argument("--limit", type=int, default=8)

    digest = sub.add_parser("digest")
    add_root_user(digest)
    digest.add_argument("--query", required=True)
    digest.add_argument("--limit", type=int, default=6)

    message = sub.add_parser("add-message")
    add_root_user(message)
    message.add_argument("--session", required=True)
    message.add_argument("--role", choices=["user", "assistant", "system"], required=True)
    message.add_argument("--text", required=True)

    commit = sub.add_parser("commit-session")
    add_root_user(commit)
    commit.add_argument("--session", required=True)
    commit.add_argument("--extractor", choices=["auto", "api", "rule"], default="auto")

    reindex = sub.add_parser("reindex")
    add_root_user(reindex)

    lifecycle = sub.add_parser("lifecycle-report")
    add_root_user(lifecycle)
    lifecycle.add_argument("--limit", type=int, default=20)

    archive = sub.add_parser("archive-cold")
    add_root_user(archive)
    archive.add_argument("--limit", type=int, default=20)
    archive.add_argument("--max-hotness", type=float, default=0.12)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = MemoryStore(
        Path(args.root),
        embedding_provider=_build_embedding_provider(args.embedding_provider),
        rerank_provider=_build_rerank_provider(args.rerank_provider),
    )

    if args.command == "init":
        store.init(args.user)
        print(f"initialized memory root: {store.root}")
        return

    if args.command == "add-memory":
        store.init(args.user)
        memory_type = validate_memory_type(args.type)
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        path = store.add_memory(
            MemoryItem(
                user_id=args.user,
                memory_type=memory_type,
                title=args.title,
                text=args.text,
                tags=tags,
            )
        )
        print(path)
        return

    if args.command == "update-memory":
        store.init(args.user)
        tags = None if args.tags is None else [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        result = store.update_memory(args.id, user_id=args.user, title=args.title, text=args.text, tags=tags)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "delete-memory":
        store.init(args.user)
        result = store.delete_memory(args.id, user_id=args.user)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "merge-memory":
        store.init(args.user)
        result = store.merge_memory(args.target, args.source, user_id=args.user)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "update-profile":
        store.init(args.user)
        result = store.upsert_profile(args.user, text=args.text, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "update-daily":
        store.init(args.user)
        result = store.update_daily_behavior(args.user, text=args.text, day=args.date, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "record-event":
        store.init(args.user)
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        result = store.record_event(args.user, event_type=args.event_type, text=args.text, day=args.date, tags=tags)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        rows = store.search(args.query, user_id=args.user, memory_type=args.type, limit=args.limit)
        for row in rows:
            print(f"[{row['type']}] {row['title']} :: {row['abstract']} :: {row['path']}")
        return

    if args.command == "hybrid-search":
        rows = store.hybrid_search(args.query, user_id=args.user, memory_type=args.type, limit=args.limit)
        for row in rows:
            print(
                f"[{row['type']}] {row['title']} :: final={row.get('final_score', 0.0):.3f} "
                f"keyword={row.get('keyword_score', 0.0):.3f} embedding={row.get('embedding_score', 0.0):.3f} "
                f"rerank={row.get('rerank_score', 0.0):.3f} hot={row.get('hotness', 0.0):.3f} :: {row['path']}"
            )
        return

    if args.command == "digest":
        print(MemoryHook(store).build_digest(args.user, args.query, limit=args.limit))
        return

    if args.command == "add-message":
        store.init(args.user)
        path = SessionManager(store).add_message(args.user, args.session, args.role, args.text)
        print(path)
        return

    if args.command == "commit-session":
        store.init(args.user)
        diff = SessionManager(store, extractor=_build_extractor(args.extractor)).commit(args.user, args.session)
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        return

    if args.command == "reindex":
        store.reindex(user_id=args.user)
        print("reindexed")
        return

    if args.command == "lifecycle-report":
        rows = store.lifecycle_report(user_id=args.user, limit=args.limit)
        for row in rows:
            print(
                f"[{row['lifecycle_state']}] {row['hotness']:.3f} "
                f"active={row['active_count']} :: {row['type']} :: {row['title']} :: {row['path']}"
            )
        return

    if args.command == "archive-cold":
        result = store.archive_cold_memories(user_id=args.user, limit=args.limit, max_hotness=args.max_hotness)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


def _build_embedding_provider(mode: str):
    if mode == "local":
        return HashingEmbeddingProvider()
    provider = build_embedding_provider_from_env()
    if provider:
        return provider
    if mode == "api":
        raise SystemExit("MEMORYOS_EMBEDDING_MODEL is required when --embedding-provider api is used")
    return HashingEmbeddingProvider()


def _build_rerank_provider(mode: str):
    if mode == "off":
        return None
    provider = build_rerank_provider_from_env()
    if provider:
        return provider
    if mode == "api":
        raise SystemExit("MEMORYOS_RERANK_MODEL is required when --rerank-provider api is used")
    return None


def _build_extractor(mode: str):
    if mode == "rule":
        return RuleBasedExtractor()
    provider = build_chat_provider_from_env()
    if provider:
        return JsonLLMMemoryExtractor(provider)
    if mode == "api":
        raise SystemExit("MEMORYOS_LLM_MODEL is required when --extractor api is used")
    return RuleBasedExtractor()
