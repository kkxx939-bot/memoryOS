"""Runtime readiness guards shared by production worker entry points."""

from __future__ import annotations

from typing import Any


def readiness_for_source_store(source_store: Any) -> Any | None:
    """Return a readiness gate without importing the runtime package.

    Worker modules are imported while ``memoryos.runtime.container`` is being
    initialized.  Importing ``memoryos.runtime.readiness`` here would first
    execute ``memoryos.runtime.__init__`` and recurse into that container.  A
    readiness gate is intentionally a tiny duck-typed boundary instead.
    """

    readiness = getattr(source_store, "readiness", None)
    return readiness if callable(getattr(readiness, "require_ready", None)) else None


def readiness_for_session_service(service: Any) -> Any | None:
    committer = getattr(service, "committer", None)
    committer = getattr(committer, "delegate", committer)
    source_store = getattr(committer, "source_store", None)
    readiness = readiness_for_source_store(source_store)
    if readiness is not None:
        return readiness
    planner = getattr(service, "memory_planner", None)
    return readiness_for_source_store(getattr(planner, "source_store", None))


def require_source_store_ready(source_store: Any) -> None:
    readiness = readiness_for_source_store(source_store)
    if readiness is not None:
        readiness.require_ready()


def require_session_service_ready(service: Any) -> None:
    readiness = readiness_for_session_service(service)
    if readiness is not None:
        readiness.require_ready()


def session_service_is_ready(service: Any) -> bool:
    """Return whether an attached runtime still permits ordinary work.

    Standalone services without a runtime readiness binding retain their
    historical behavior.  A bound service is ready only in the exact READY
    state; DEGRADED/RECOVERING/NOT_READY must all stop a leased batch.
    """

    readiness = readiness_for_session_service(service)
    if readiness is None:
        return True
    state_obj = getattr(readiness, "state", None)
    return str(getattr(state_obj, "value", state_obj or "")) == "READY"


def require_source_store_recovering(source_store: Any) -> None:
    """Authorize only the runtime builder's projection-recovery entry point."""

    readiness = readiness_for_source_store(source_store)
    state_obj = getattr(readiness, "state", None)
    state = str(getattr(state_obj, "value", state_obj or "UNBOUND"))
    if readiness is None or state != "RECOVERING":
        raise RuntimeError(f"projection startup entry requires RECOVERING runtime, got {state}")
