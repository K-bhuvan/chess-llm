#!/usr/bin/env bash
# Install Unsloth next to an existing Blackwell (cu128) torch. Do not let pip replace torch.
set -eu
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate llm_gpu
pip install "unsloth==2026.8.18" unsloth_zoo --no-deps
pip install \
  "transformers>=4.51.3,<=5.5.0" \
  "trl>=0.18.2,<=0.24.0" \
  "datasets>=3.4.1,<4.4.0" \
  peft accelerate bitsandbytes sentencepiece tyro structlog diffusers \
  hf_transfer msgspec
pip uninstall -y wandb || true
export UNSLOTH_SKIP_TORCHVISION_CHECK=1
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); import unsloth; print('unsloth ok')"
