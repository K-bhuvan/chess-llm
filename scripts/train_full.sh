#!/usr/bin/env bash
# SFT then a short GRPO pass. Run inside WSL (conda env CONDA_ENV, default llm_gpu).
# Checkpoints default to $HOME/chess_outputs so a full Windows C: drive is not required.
set -eu
# shellcheck source=/dev/null
source "$(dirname "$0")/wsl_env.sh"

OUT="${CHESS_OUTPUT:-$HOME/chess_outputs}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/train_full.log") 2>&1

ADAPTER="${SFT_ADAPTER:-}"
if [ -z "$ADAPTER" ]; then
  for d in $(ls -d "$OUT/sft"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    if [ -f "$d/adapter_model.safetensors" ]; then
      ADAPTER="$d"
    fi
  done
fi
if [ -z "$ADAPTER" ] && [ -f "$OUT/sft/adapter_model.safetensors" ]; then
  ADAPTER="$OUT/sft"
fi

if [ -f "$OUT/sft_done" ]; then
  echo "=== skip SFT (found $OUT/sft_done); adapter=$ADAPTER ==="
else
  echo "=== SFT ==="
  python -m chess_llm.train.sft --config configs/sft.yaml --output-dir "$OUT/sft" --no-eval
  touch "$OUT/sft_done"
  for d in $(ls -d "$OUT/sft"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    if [ -f "$d/adapter_model.safetensors" ]; then
      ADAPTER="$d"
    fi
  done
fi

if [ -z "$ADAPTER" ]; then
  echo "no complete SFT adapter under $OUT/sft" >&2
  exit 1
fi

echo "=== sync SFT adapters (playable now) ==="
mkdir -p outputs
rm -rf outputs/sft
cp -a "$ADAPTER" outputs/sft

echo "=== GRPO (adapter=$ADAPTER) ==="
python -m chess_llm.train.grpo --config configs/grpo.yaml \
  --max-samples 20000 --max-steps 1000 \
  --adapter-path "$ADAPTER" --output-dir "$OUT/grpo"

echo "=== sync GRPO adapters ==="
rm -rf outputs/grpo
cp -a "$OUT/grpo" outputs/grpo

echo "=== done ==="
python -m chess_llm.eval.metrics --data data/test.parquet --limit 2000 --adapter outputs/grpo --out outputs/eval.json
