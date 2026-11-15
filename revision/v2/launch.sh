#!/usr/bin/env bash
# Start (or resume) a benchmark queue in the background, from WSL.
# The first launch freezes the code into <bench>/code; later launches reuse it and skip finished jobs.
# usage: launch.sh [<bench dir> [run_queue.py options]]   (default bench dir: /home/jack/ageas_work/v2/bench)
set -euo pipefail
BENCH=${1:-/home/jack/ageas_work/v2/bench}
shift || true
SHARE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)  # the folder holding this Ageas checkout
source /home/jack/anaconda3/etc/profile.d/conda.sh
conda activate ageas
export PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$BENCH"
if [ -f "$BENCH/queue.pid" ] && kill -0 "$(cat "$BENCH/queue.pid")" 2>/dev/null; then
  echo "queue already running (pid $(cat "$BENCH/queue.pid"))" >&2
  exit 1
fi
[ -d "$BENCH/code" ] || python -I "$SHARE/Ageas/revision/v2/snapshot.py" --out "$BENCH"
rm -f "$BENCH/STOP"
cd "$BENCH"
setsid nohup python -I "$BENCH/code/Ageas/revision/v2/run_queue.py" --out "$BENCH" "$@" \
  >> "$BENCH/queue.log" 2>&1 < /dev/null &
echo $! > "$BENCH/queue.pid"
echo "queue pid $(cat "$BENCH/queue.pid"); log: $BENCH/queue.log"
