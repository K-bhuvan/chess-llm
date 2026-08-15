#!/usr/bin/env bash
# SFT then a short GRPO pass. Run inside WSL (conda env CONDA_ENV, default llm_gpu).
# Checkpoints default to $HOME/chess_outputs so a full Windows C: drive is not required.
set -eu
# shellcheck source=/dev/null
source "$(dirname "$0")/wsl_env.sh"

OUT="${CHESS_OUTPUT:-$HOME/chess_outputs}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/train_full.log") 2>&1

echo "=== SFT ==="
python -m chess_llm.train.sft --config configs/sft.yaml --output-dir "$OUT/sft" --no-eval

echo "=== sync SFT adapters (playable now) ==="
mkdir -p outputs
rm -rf outputs/sft
cp -a "$OUT/sft" outputs/sft

echo "=== GRPO ==="
python -m chess_llm.train.grpo --config configs/grpo.yaml \
  --max-samples 20000 --max-steps 1000 \
  --adapter-path "$OUT/sft" --output-dir "$OUT/grpo"

echo "=== sync GRPO adapters ==="
rm -rf outputs/grpo
cp -a "$OUT/grpo" outputs/grpo

echo "=== done ==="
python -m chess_llm.eval.metrics --data data/test.parquet --limit 2000 --adapter outputs/grpo --out outputs/eval.json
