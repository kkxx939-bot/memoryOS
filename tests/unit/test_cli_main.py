from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from memoryos.connect import ConnectMetadata

cli_main = import_module("memoryos.api.cli.main")


class FakeCLIClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, root: str) -> None:
        self.root = root

    def predict(self, request, policies=None):  # noqa: ANN001, ANN201
        self.calls.append({"root": self.root, "request": request, "policies": policies})

        class Result:
            def to_dict(self) -> dict[str, Any]:
                return {"episode_id": request.episode_id, "ok": True}

        return Result()


def test_cli_version_and_inspect_architecture(capsys) -> None:  # noqa: ANN001
    assert cli_main.main(["version"]) == 0
    assert "0.1.0" in capsys.readouterr().out

    assert cli_main.main(["inspect-architecture"]) == 0
    assert "MemoryOS" in capsys.readouterr().out


def test_cli_predict_rejects_missing_metadata_before_client(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "MemoryOSClient", FakeCLIClient)

    exit_code = cli_main.main(["predict", "--user", "u1", "--episode", "s1", "--observation", "hot"])

    assert exit_code == 2
    assert "requires explicit embodied/action_capable" in capsys.readouterr().err
    assert FakeCLIClient.calls == []


def test_cli_predict_rejects_agent_metadata_before_client(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "MemoryOSClient", FakeCLIClient)
    metadata = json.dumps(ConnectMetadata.default_agent("codex").to_dict())

    exit_code = cli_main.main(
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


def test_cli_predict_allows_action_capable_embodied_metadata_json(monkeypatch, capsys) -> None:  # noqa: ANN001
    FakeCLIClient.calls = []
    monkeypatch.setattr(cli_main, "MemoryOSClient", FakeCLIClient)
    metadata = json.dumps(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict())

    exit_code = cli_main.main(
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
    monkeypatch.setattr(cli_main, "MemoryOSClient", FakeCLIClient)
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(
        json.dumps(ConnectMetadata.action_capable_embodied("reachy_mini").to_dict()),
        encoding="utf-8",
    )

    exit_code = cli_main.main(
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


def test_console_script_entrypoints_are_declared() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'memoryos = "memoryos.api.cli.main:main"' in pyproject
    assert 'memoryos-mcp-server = "memoryos.api.mcp.stdio:main"' in pyproject
    assert 'memoryos-agent-hook = "memoryos.adapters.agent_hooks.cli:main"' in pyproject
