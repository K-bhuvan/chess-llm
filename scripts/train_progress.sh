#!/usr/bin/env bash
set -u
OUT="${CHESS_OUTPUT:-$HOME/chess_outputs}"
LOG="$OUT/train_full.log"
echo "=== python train ==="
pgrep -af "chess_llm.train" || echo "no chess_llm.train process"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo "=== latest progress ==="
tr -d '\000' <"$LOG" | grep -oE 'loss[^|]*|[0-9]+/26664[^]]*\]|Total steps = [0-9]+|saved adapters|GRPO|Error|Traceback' | tail -n 25
echo "=== checkpoints ==="
ls -lt "$OUT/sft" 2>/dev/null | head -n 15
