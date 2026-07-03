from __future__ import annotations

from typing import Any

# The concrete SQLite adapter is still local-first and dynamic. The port keeps
# usecases/services from importing adapter modules directly while the repository
# is being split into smaller typed ports.
MemoryRepository = Any
