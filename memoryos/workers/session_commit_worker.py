from __future__ import annotations

from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive


class SessionCommitWorker:
    def __init__(self, service: SessionCommitService) -> None:
        self.service = service

    def process_archive(self, archive: SessionArchive) -> dict:
        result = self.service.async_commit(archive)
        return {"task_id": result.task_id, "status": result.status, "done": result.done}
