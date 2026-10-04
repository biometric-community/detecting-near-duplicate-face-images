#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
"$PY" -m pip install -q -r "$ROOT/requirements.txt"
echo "OK env=$PY"
