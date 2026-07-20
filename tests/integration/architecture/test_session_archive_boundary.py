from __future__ import annotations

import json
import subprocess
import sys


def test_filesystem_session_archive_import_does_not_load_or_register_memory() -> None:
    code = """
import json
import sys
from infrastructure.store.filesystem.session_archive import SessionArchiveStore

print(json.dumps({
    "evidence_modules": sorted(name for name in sys.modules if name == "pre.evidence" or name.startswith("pre.evidence.")),
    "legacy_encoder_loaded": "memory.commit.evidence.encoder" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "evidence_modules": [],
        "legacy_encoder_loaded": False,
    }
