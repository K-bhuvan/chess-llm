#!/usr/bin/env python3
"""Print GPU / CUDA facts. Run inside WSL with the chess_llm conda env."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    print(f"python {sys.version.split()[0]}  {sys.executable}")
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        print("nvidia-smi: MISSING")
        return 1
    out = subprocess.check_output(
        [
            nvsmi,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    print(f"nvidia-smi: {out}")
    try:
        import torch
    except ImportError:
        print("torch: not installed (activate chess_llm / llm_gpu env)")
        return 1
    print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA not visible to PyTorch. Use WSL2 + cu128 wheels.")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"capability: sm_{cap[0]}{cap[1]}")
    x = torch.zeros(1, device="cuda")
    print(f"alloc_ok: {x.device}")
    try:
        import unsloth  # noqa: F401

        print("unsloth: import ok")
    except ImportError:
        print("unsloth: not installed (pip install unsloth unsloth_zoo bitsandbytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
