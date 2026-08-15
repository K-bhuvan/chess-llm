"""DPO fallback when GRPO does not fit in 16GB: Stockfish move vs random legal."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from chess_llm.data.format import (
    build_assistant_reply,
    build_user_prompt,
    is_legal_uci,
    random_other_legal_uci,
)
from chess_llm.train.config import load_yaml


def _random_rejected_uci(fen: str, chosen: str, rng: random.Random) -> str | None:
    return random_other_legal_uci(fen, chosen, rng)


def parquet_to_dpo_dataset(path: Path, max_samples: int | None, seed: int):
    from datasets import Dataset

    ds = Dataset.from_parquet(str(path))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    rng = random.Random(seed)
    prompts, chosen, rejected = [], [], []
    for row in ds:
        fen, uci, cp, mate = row["fen"], row["uci"], row.get("cp"), row.get("mate")
        if not is_legal_uci(fen, uci):
            continue
        other = _random_rejected_uci(fen, uci, rng)
        if other is None:
            continue
        prompt = build_user_prompt(fen)
        prompts.append(prompt)
        chosen.append(build_assistant_reply(uci, cp, mate, fen))
        rejected.append(build_assistant_reply(other, 0, None, fen))
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DPO fallback for chess Qwen-4B")
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    import unsloth  # noqa: F401
    from chess_llm.train.sft import load_unsloth_model
    from trl import DPOConfig, DPOTrainer

    adapter = Path(cfg["adapter_path"]) if cfg.get("adapter_path") else None
    load_cfg = {
        **cfg,
        "target_modules": cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        "lora_dropout": 0,
        "gradient_checkpointing": "unsloth",
    }
    if adapter is not None and adapter.exists():
        load_cfg["model_name"] = str(adapter)
        model, tokenizer = load_unsloth_model(load_cfg, apply_lora=False)
    else:
        model, tokenizer = load_unsloth_model(load_cfg, apply_lora=True)
    ds = parquet_to_dpo_dataset(Path(cfg["rl_path"]), cfg.get("max_samples"), int(cfg.get("seed", 42)))

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=DPOConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            num_train_epochs=float(cfg.get("num_train_epochs", 1)),
            max_steps=int(cfg.get("max_steps", 1000)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 100)),
            bf16=True,
            seed=int(cfg.get("seed", 42)),
            report_to="none",
            max_length=int(cfg.get("max_seq_length", 256)),
        ),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"saved DPO adapters to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
