#!/usr/bin/env bash
set -u
OUT="${CHESS_OUTPUT:-$HOME/chess_outputs}"
PIDFILE="$OUT/train_full.pid"
LOG="$OUT/train_full.log"
echo "=== pidfile ==="
if [ -f "$PIDFILE" ]; then
  cat "$PIDFILE"
  echo
  pid=$(cat "$PIDFILE")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "launcher: alive"
    ps -p "$pid" -o pid,etime,pcpu,pmem,cmd
  else
    echo "launcher: dead"
  fi
else
  echo "missing pidfile"
fi
echo "=== python train ==="
pgrep -af "chess_llm.train" || echo "no chess_llm.train process"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo "=== log bytes ==="
wc -c "$LOG" 2>/dev/null || true
echo "=== log tail ==="
# strip NULs from tqdm/unsloth progress
tr -d '\000' <"$LOG" | tail -n 30
