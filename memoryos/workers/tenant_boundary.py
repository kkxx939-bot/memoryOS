"""Fail-closed tenant binding for durable worker jobs."""

from __future__ import annotations

from collections.abc import Mapping


class WorkerTenantBoundaryError(ValueError):
    """A queued job is missing or contradicts its tenant-bound worker."""


def require_bound_job_tenant(
    payload: Mapping[str, object],
    *,
    bound_tenant_id: str,
) -> str:
    """Return the declared tenant only when it matches the worker's store.

    Queue payloads are durable, externally mutable artifacts and therefore are
    not authority for choosing an archive namespace.  In particular, absence
    must never mean ``default`` for a worker already bound to another tenant.
    """

    declared = payload.get("tenant_id")
    if not isinstance(declared, str) or not declared:
        raise WorkerTenantBoundaryError("queued memory job has no explicit tenant identity")
    if not isinstance(bound_tenant_id, str) or not bound_tenant_id:
        raise WorkerTenantBoundaryError("memory worker has no bound tenant identity")
    if declared != bound_tenant_id:
        raise WorkerTenantBoundaryError("queued memory job crosses the worker tenant boundary")
    return declared
