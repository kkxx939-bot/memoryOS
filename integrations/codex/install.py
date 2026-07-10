from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    target = codex_home / "hooks.json"
    current = json.loads(target.read_text()) if target.exists() else {"hooks": {}}
    desired = json.loads(Path(__file__).with_name("hooks.json").read_text())
    hooks = current.setdefault("hooks", {})
    for event_name, entries in desired["hooks"].items():
        existing = hooks.setdefault(event_name, [])
        if args.uninstall:
            hooks[event_name] = [entry for entry in existing if "memoryos-agent-hook" not in json.dumps(entry)]
            if not hooks[event_name]:
                hooks.pop(event_name)
        else:
            for entry in entries:
                if entry not in existing:
                    existing.append(entry)
    print(json.dumps({"target": str(target), "dry_run": args.dry_run, "uninstall": args.uninstall}))
    if args.dry_run:
        return 0
    codex_home.mkdir(parents=True, exist_ok=True)
    backup = target.with_suffix(".json.memoryos.bak")
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(json.dumps(current, indent=2) + "\n")
    if args.uninstall:
        subprocess.run(["codex", "mcp", "remove", "memoryos"], check=False)
    elif subprocess.run(["codex", "mcp", "get", "memoryos"], check=False, capture_output=True).returncode != 0:
        subprocess.run(["codex", "mcp", "add", "memoryos", "--", "memoryos-mcp-server"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
