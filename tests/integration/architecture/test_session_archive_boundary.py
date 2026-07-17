from __future__ import annotations

import json
import subprocess
import sys


def test_filesystem_session_archive_import_does_not_load_or_register_memory() -> None:
    code = """
import json
import sys
from memoryos.adapters.persistence.filesystem.session_archive import SessionArchiveStore
from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder

try:
    session_evidence_encoder()
except RuntimeError as exc:
    registration_error = str(exc)
else:
    registration_error = ""

print(json.dumps({
    "memory_modules": sorted(name for name in sys.modules if name.startswith("memoryos.memory")),
    "registration_error": registration_error,
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
        "memory_modules": [],
        "registration_error": "Session evidence encoder is not registered",
    }
