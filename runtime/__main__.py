"""支持通过 ``python -m runtime`` 启动 MemoryOS。"""

from runtime.entry import run

if __name__ == "__main__":
    raise SystemExit(run())
