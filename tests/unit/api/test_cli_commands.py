"""验证 CLI 命令协议及其应用入口声明。"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from pre.connect import ConnectMetadata

cli_main = import_module("openApi.cli.commands")


class FakeCLIClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, root: str) -> None:
        self.root = root
        self.last_recall_trace_id = ""

    def predict(self, request, policies=None):  # noqa: ANN001, ANN201
        self.calls.append({"root": self.root, "request": request, "policies": policies})

        class Result:
            def to_dict(self) -> dict[str, Any]:
                return {"episode_id": request.episode_id, "ok": True}

        return Result()

    def search_context(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_recall_trace_id = "trace-cli"
        self.calls.append({"operation": "search", "query": query, **kwargs})
        return [{"uri": "memoryos://user/u1/resources/result", "text": "result"}]

    def assemble_context(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"operation": "assemble", "query": query, **kwargs})
        return {"contexts": [], "packed_context": "assembled", "trace_id": "trace-assemble"}

    def read(self, uri: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"operation": "read", "uri": uri, **kwargs})
        return {"object": {"uri": uri}, "layer": kwargs["layer"], "content": "safe"}

    def recall_trace(self, trace_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"operation": "trace", "trace_id": trace_id, **kwargs})
        return {"trace_id": trace_id}

    def archive_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"operation": "archive", "query": query, **kwargs})
        return [{"archive_uri": "memoryos://user/u1/sessions/history/s1"}]


def test_cli_version_and_inspect_architecture(capsys) -> None:  # noqa: ANN001
    assert cli_main.run(["version"]) == 0
    assert "0.1.0" in capsys.readouterr().out

    assert cli_main.run(["inspect-architecture"]) == 0
    assert "MemoryOS" in capsys.readouterr().out


def test_cli_predict_rejects_missing_metadata_before_client(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)

    exit_code = cli_main.run(["predict", "--user", "u1", "--episode", "s1", "--observation", "hot"])

    assert exit_code == 2
    assert "requires explicit embodied/action_capable" in capsys.readouterr().err
    assert FakeCLIClient.calls == []


def test_cli_predict_rejects_agent_metadata_before_client(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    metadata = json.dumps(ConnectMetadata.default_agent("codex").to_dict())

    exit_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            metadata,
        ]
    )

    assert exit_code == 2
    assert "can_predict_behavior=True" in capsys.readouterr().err
    assert FakeCLIClient.calls == []


def test_cli_predict_rejects_string_false_behavior_capability(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()
    metadata["capabilities"]["can_predict_behavior"] = "false"

    exit_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            json.dumps(metadata),
        ]
    )

    assert exit_code == 2
    assert "capability field must be boolean" in capsys.readouterr().err
    assert FakeCLIClient.calls == []


def test_cli_predict_stable_errors_for_bad_metadata_and_policies(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    missing = tmp_path / "missing.json"
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad json", encoding="utf-8")
    valid_metadata = json.dumps(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())

    missing_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-file",
            str(missing),
        ]
    )
    missing_err = capsys.readouterr().err
    bad_file_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-file",
            str(bad_file),
        ]
    )
    bad_file_err = capsys.readouterr().err
    bad_json_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            "{bad json",
        ]
    )
    bad_json_err = capsys.readouterr().err
    bad_policies_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            valid_metadata,
            "--policies-json",
            "{}",
        ]
    )
    bad_policies_err = capsys.readouterr().err
    bad_policies_json_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            valid_metadata,
            "--policies-json",
            "{bad json",
        ]
    )
    bad_policies_json_err = capsys.readouterr().err
    bad_policies_item_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            valid_metadata,
            "--policies-json",
            '["not-object"]',
        ]
    )
    bad_policies_item_err = capsys.readouterr().err

    assert (
        missing_code
        == bad_file_code
        == bad_json_code
        == bad_policies_code
        == bad_policies_json_code
        == bad_policies_item_code
        == 2
    )
    assert "failed to read connect metadata file" in missing_err
    assert str(missing) not in missing_err
    assert "valid JSON" in bad_file_err
    assert "valid JSON" in bad_json_err
    assert "policies JSON must be an array" in bad_policies_err
    assert "policies JSON must be valid JSON" in bad_policies_json_err
    assert "policies JSON entries must be objects" in bad_policies_item_err
    assert (
        "Traceback"
        not in missing_err
        + bad_file_err
        + bad_json_err
        + bad_policies_err
        + bad_policies_json_err
        + bad_policies_item_err
    )
    assert FakeCLIClient.calls == []


def test_cli_predict_error_output_redacts_paths_and_secrets(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    metadata = ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()
    metadata["connect_type"] = "bad /Users/gulf/private password=secret api_key=sk-test token=abc"

    exit_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            json.dumps(metadata),
        ]
    )
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "/Users/gulf" not in err
    assert "secret" not in err
    assert "sk-test" not in err
    assert "abc" not in err
    assert "Traceback" not in err
    assert FakeCLIClient.calls == []


def test_cli_predict_allows_action_capable_embodied_metadata_json(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    metadata = json.dumps(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())

    exit_code = cli_main.run(
        [
            "predict",
            "--root",
            "/tmp/memory",
            "--user",
            "u1",
            "--episode",
            "s1",
            "--observation",
            "hot",
            "--connect-metadata-json",
            metadata,
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["episode_id"] == "s1"
    assert FakeCLIClient.calls[0]["request"].connect_metadata["connect_type"] == "embodied"


def test_cli_predict_allows_action_capable_embodied_metadata_file(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()),
        encoding="utf-8",
    )

    exit_code = cli_main.run(
        [
            "predict",
            "--user",
            "u1",
            "--episode",
            "s2",
            "--observation",
            "hot",
            "--connect-metadata-file",
            str(metadata_file),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["episode_id"] == "s2"


def test_cli_exposes_context_search_and_assembly(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)

    search_code = cli_main.run(["context-search", "--root", "/tmp/memory", "--user", "u1", "--query", "preference"])
    search_payload = json.loads(capsys.readouterr().out)
    assemble_code = cli_main.run(
        [
            "context-assemble",
            "--root",
            "/tmp/memory",
            "--user",
            "u1",
            "--query",
            "preference",
            "--context-type",
            "memory",
        ]
    )
    assemble_payload = json.loads(capsys.readouterr().out)

    assert search_code == assemble_code == 0
    assert search_payload["trace_id"] == "trace-cli"
    assert assemble_payload["packed_context"] == "assembled"
    assert [item["operation"] for item in FakeCLIClient.calls] == ["search", "assemble"]
    assert FakeCLIClient.calls[0]["caller"].user_id == "u1"


def test_cli_exposes_context_read_trace_and_archive_search(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "_client", FakeCLIClient)
    uri = "memoryos://user/u1/resources/report"

    assert cli_main.run(["context-read", "--user", "u1", "--uri", uri, "--layer", "L1"]) == 0
    assert json.loads(capsys.readouterr().out)["layer"] == "L1"
    assert cli_main.run(["recall-trace", "--user", "u1", "--trace-id", "trace-1"]) == 0
    assert json.loads(capsys.readouterr().out)["trace_id"] == "trace-1"
    assert cli_main.run(["archive-search", "--user", "u1", "--query", "yesterday"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["archive_uri"].endswith("/s1")
    assert [item["operation"] for item in FakeCLIClient.calls] == ["read", "trace", "archive"]


def test_console_script_entrypoints_are_declared() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'memoryos = "runtime.entry:run"' in pyproject
    assert 'memoryos-mcp-server = "openApi.mcp.stdio:run"' in pyproject
    assert 'memoryos-agent-hook = "openApi.cli.agent_hooks:run"' in pyproject
    assert 'memoryos-http-server = "openApi.http.app:run"' in pyproject


def test_console_script_entrypoint_targets_import() -> None:
    targets = [
        ("runtime.entry", "run"),
        ("openApi.mcp.stdio", "run"),
        ("openApi.cli.agent_hooks", "run"),
        ("openApi.http.app", "run"),
    ]

    for target, callable_name in targets:
        module = import_module(target)
        assert callable(getattr(module, callable_name))
