#!/bin/sh
set -eu
memoryos doctor "${@}"
python3 -m json.tool "${HOME}/.claude/settings.json" >/dev/null
