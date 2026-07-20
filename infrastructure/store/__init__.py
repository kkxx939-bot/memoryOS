"""MemoryOS 的存储协议、数据模型和具体持久化实现。

上层只依赖 ``infrastructure.store.contracts``；运行时组合根从这里选择文件、
SQLite、向量和锁实现，并注入对应协议。
"""
