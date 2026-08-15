#!/usr/bin/env bash
set -eu
# shellcheck source=/dev/null
source "$(dirname "$0")/wsl_env.sh"
python -m chess_llm.train.grpo --config configs/grpo.yaml --max-samples 8 --max-steps 1
