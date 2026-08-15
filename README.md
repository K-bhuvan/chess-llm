# chess-llm

Fine-tune Qwen3-4B on Stockfish-labeled chess (QLoRA SFT, optional GRPO) and play it in a local Matrix-style UI.

The 4B model runs on your GPU. Do **not** host it on Vercel.

This repo is **code only**. Parquet labels, LoRA adapters, and train logs are gitignored.

## Setup

Linux or WSL, Python 3.11+, CUDA GPU. Install the package, then Unsloth against a **cu128** (or newer) PyTorch — unconstrained `pip install unsloth` can pin an old torch and break RTX 50-series.

```bash
pip install -e ".[dev]"
bash scripts/install_unsloth_blackwell.sh   # safe Unsloth install next to existing torch
python scripts/gpu_smoke.py
```

New env from scratch: `bash scripts/setup_wsl.sh` (name is historical; it is a conda+CUDA setup).

## Data

Stream [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) (CC0). This does **not** download the full dump, and **does not** commit parquet.

```bash
python -m chess_llm.data.sample --out-dir data --min-depth 20 --sample-mod 1
```

Writes `train` (2M), `val`/`test` (20k each), `rl` (50k). Lichess releases that eval export as **CC0 1.0**. Stockfish itself is GPL; this repo does not ship the engine. Qwen3-4B-Instruct is Apache-2.0; adapters are not in git.

## Train

Default SFT is **4,000 QLoRA steps** (~5–10h on a 16GB card), then a short GRPO pass.

```bash
python -m chess_llm.train.sft --config configs/sft.yaml --max-samples 32   # smoke
bash scripts/train_full.sh
```

OOM: lower `per_device_train_batch_size`, `lora_r`, or GRPO `num_generations` in `configs/`. If GRPO still OOMs: `python -m chess_llm.train.dpo --config configs/dpo.yaml`.

## Play

```bash
python -m chess_llm.serve    # stub moves until adapters exist
cd web && cp .env.example .env.local && npm install && npm run dev
```

http://localhost:3000

```bash
python -m chess_llm.eval.metrics --data data/test.parquet --limit 2000 --adapter outputs/grpo
python -m chess_llm.infer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
pytest
```

## Windows

Training was tested on **Windows + WSL2**, not native Windows (Unsloth/CUDA live in Linux). Give WSL enough RAM in `%USERPROFILE%\.wslconfig`, and do not install a Linux NVIDIA driver. Unattended run helper: `scripts/watchdog.ps1`.
