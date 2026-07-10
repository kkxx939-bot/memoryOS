from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--settings", default=str(Path.home() / ".claude" / "settings.json"))
    args = parser.parse_args()
    target = Path(args.settings)
    template = Path(__file__).with_name("settings.template.json")
    current = json.loads(target.read_text()) if target.exists() else {}
    backup = target.with_suffix(".json.memoryos.bak")
    if args.uninstall:
        current.get("mcpServers", {}).pop("memoryos", None)
        hooks = current.get("hooks", {})
        for name in list(hooks):
            hooks[name] = [entry for entry in hooks[name] if "memoryos-agent-hook" not in json.dumps(entry)]
            if not hooks[name]:
                hooks.pop(name)
    else:
        desired = json.loads(template.read_text())
        current.setdefault("mcpServers", {}).setdefault("memoryos", desired["mcpServers"]["memoryos"])
        hooks = current.setdefault("hooks", {})
        for name, entries in desired["hooks"].items():
            existing = hooks.setdefault(name, [])
            for entry in entries:
                if entry not in existing:
                    existing.append(entry)
    print(json.dumps({"target": str(target), "dry_run": args.dry_run, "uninstall": args.uninstall}))
    if args.dry_run:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
