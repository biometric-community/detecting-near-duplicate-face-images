#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
LFW="${1:-../../datasets/lfw}"
set +e
bash scripts/check_dataset_size.sh "$LFW"
rc=$?
set -e
if [[ "$rc" -eq 3 ]]; then echo "Dataset >= 5 GiB — skip full train." >&2; exit 3; fi
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
mkdir -p outputs/logs outputs/checkpoints/full
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
LOG=outputs/logs/train_full.log
PIDFILE=outputs/logs/train_full.pid
echo "Starting FULL NearDup ChebNet CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "$PY" -u -m neardup.train --config configs/full.yaml "${@:2}" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "PID=$(cat $PIDFILE) log=$LOG"
