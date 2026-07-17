from __future__ import annotations

import pytest

from tests.support.session_archive import compose_domain_runtime_bindings


@pytest.fixture(autouse=True)
def _compose_domain_runtime_boundaries() -> None:
    """Give each test the same explicit domain bindings as runtime startup."""

    compose_domain_runtime_bindings()
