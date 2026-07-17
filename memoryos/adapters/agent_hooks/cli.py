"""Compatibility entrypoint for the application-owned agent-hook CLI."""

from memoryos.api.cli.agent_hooks import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
