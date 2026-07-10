#!/bin/sh
set -eu
memoryos doctor "$@"
codex mcp get memoryos
python3 -m json.tool "${CODEX_HOME:-$HOME/.codex}/hooks.json" >/dev/null
