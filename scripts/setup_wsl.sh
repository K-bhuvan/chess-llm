#!/usr/bin/env bash
# WSL2 Ubuntu + Blackwell (RTX 50-series) setup for chess-llm.
# Do not install a Linux NVIDIA driver inside WSL. Use the Windows Game Ready / Studio driver.
set -euo pipefail

ENV_NAME="${ENV_NAME:-chess_llm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi missing in WSL. Install/update the Windows NVIDIA driver, then: wsl --shutdown"
  exit 1
fi

if [[ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  echo "Expected miniforge at ~/miniforge3. Install miniforge or edit this script."
  exit 1
fi

# shellcheck source=/dev/null
source "$HOME/miniforge3/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\\s"; then
  echo "conda env ${ENV_NAME} already exists"
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"

python -m pip install -U pip
# Blackwell sm_120 needs the cu128 (or newer) PyTorch index — not the default cu126 wheels.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install unsloth unsloth_zoo bitsandbytes trl peft transformers accelerate
python -m pip install -e ".[dev]"

python scripts/gpu_smoke.py
