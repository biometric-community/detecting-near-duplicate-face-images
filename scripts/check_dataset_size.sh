#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
"$PY" - <<'PY' "$@"
import json, sys
from pathlib import Path
paths=[Path(p) for p in sys.argv[1:]]
total=0
for p in paths:
    if not p.exists():
        print(f"missing {p}", file=sys.stderr); sys.exit(2)
    for f in (p.rglob("*") if p.is_dir() else [p]):
        if f.is_file(): total += f.stat().st_size
gib=total/(1024**3)
out=Path("outputs/logs"); out.mkdir(parents=True, exist_ok=True)
decision="skip_full_train" if gib>=5 else "full_train"
(out/"dataset_size.json").write_text(json.dumps({"total_bytes":total,"total_gib":gib,"decision":decision},indent=2))
print(f"Wrote {out/'dataset_size.json'}")
print(f"total_bytes={total} total_gib={gib:.4f} decision={decision}")
sys.exit(3 if decision=="skip_full_train" else 0)
PY
