# Backward compatibility shim. Do not add new logic here.
from memoryos.memory.lifecycle.memory_cooling import MemoryCoolingPolicy
from memoryos.memory.service.memory_updater import MemoryUpdater

__all__ = ["MemoryCoolingPolicy", "MemoryUpdater"]
