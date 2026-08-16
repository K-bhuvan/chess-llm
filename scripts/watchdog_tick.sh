#!/usr/bin/env bash
# One-shot status line for the Windows watchdog. Prints KEY=value.
set -u
LOG="${CHESS_OUTPUT:-$HOME/chess_outputs}/train_full.log"
SFT="${CHESS_OUTPUT:-$HOME/chess_outputs}/sft"
echo "RUNNING=$(pgrep -f 'chess_llm.train.(sft|grpo|dpo)' >/dev/null && echo 1 || echo 0)"
if [ -f "$LOG" ]; then
  echo "LOG_AGE_SEC=$(( $(date +%s) - $(stat -c %Y "$LOG") ))"
  echo "STEP=$(tr -d '\000' <"$LOG" | grep -oE '[0-9]+/(4000|1000|26664)' | tail -n 1)"
  echo "COMPLETE=$(grep -c '=== done ===' "$LOG" 2>/dev/null || echo 0)"
else
  echo "LOG_AGE_SEC=99999"
  echo "STEP="
  echo "COMPLETE=0"
fi
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,""); print "GPU="$1; print "POWER="$2}'
echo "CKPT=$(ls -d "$SFT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -n 1)"
