from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from infrastructure.store.filesystem.durable_io import ImmutableArtifactConflictError, atomic_create_json
from foundation.integrity import canonical_digest, canonical_json


def test_integrity_json_bytes_and_digest_are_deterministic() -> None:
    payload = {
        "z": {3, 1, 2},
        "unicode": "记忆",
        "at": datetime(2026, 7, 17, 1, 2, 3, 456789, tzinfo=timezone.utc),
    }

    encoded = '{"at":"2026-07-17T01:02:03.456789Z","unicode":"记忆","z":[1,2,3]}'
    assert canonical_json(payload) == encoded
    assert canonical_digest(payload) == "ca8e9a4141cfcf6ad8c2d40eeab813f707ced4c1b6bb841045c6d59450031cb4"


def test_atomic_json_create_only_identity(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    payload = {"status": "committed", "revision": 1}

    assert atomic_create_json(path, payload, artifact_root=tmp_path) is True
    assert atomic_create_json(path, payload, artifact_root=tmp_path) is False
    with pytest.raises(ImmutableArtifactConflictError):
        atomic_create_json(path, {**payload, "revision": 2}, artifact_root=tmp_path)
