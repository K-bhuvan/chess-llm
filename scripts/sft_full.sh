#!/usr/bin/env bash
set -eu
# shellcheck source=/dev/null
source "$(dirname "$0")/wsl_env.sh"
python -m chess_llm.train.sft --config configs/sft.yaml
