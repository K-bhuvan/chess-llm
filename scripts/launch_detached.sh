#!/usr/bin/env bash
# Start train_full.sh in its own session so Cursor/PowerShell cannot kill it.
set -eu
OUT="${CHESS_OUTPUT:-$HOME/chess_outputs}"
mkdir -p "$OUT"
LOG="$OUT/train_full.log"
PIDFILE="$OUT/train_full.pid"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_full.sh"

setsid nohup bash "$SCRIPT" >"$LOG" 2>&1 </dev/null &
echo $! >"$PIDFILE"
echo "STARTED:$(cat "$PIDFILE")"
echo "LOG:$LOG"
sleep 1
if ps -p "$(cat "$PIDFILE")" >/dev/null 2>&1; then
  echo "alive"
else
  echo "died immediately — see $LOG"
  tail -n 40 "$LOG" || true
  exit 1
fi
