"""Short GRPO pass: legality + Stockfish PV match + eval proximity."""

from __future__ import annotations

import argparse
from pathlib import Path

from chess_llm.data.format import build_user_prompt
from chess_llm.data.rewards import grpo_reward
from chess_llm.train.config import load_yaml


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def parquet_to_grpo_dataset(path: Path, tokenizer, max_samples: int | None):
    from datasets import Dataset

    ds = Dataset.from_parquet(str(path))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_prompt(batch):
        prompts = []
        for fen in batch["fen"]:
            messages = [{"role": "user", "content": build_user_prompt(fen)}]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        return {"prompt": prompts}

    return ds.map(to_prompt, batched=True)


def make_reward_fn():
    def reward_func(completions, fen, uci, cp, mate, **kwargs):
        rewards = []
        for completion, f, u, c, m in zip(completions, fen, uci, cp, mate):
            rewards.append(
                grpo_reward(
                    _completion_text(completion),
                    fen=f,
                    target_uci=u,
                    target_cp=c,
                    target_mate=m,
                )
            )
        return rewards

    return reward_func


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GRPO sharpening for chess Qwen-4B")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.adapter_path:
        cfg["adapter_path"] = args.adapter_path

    import unsloth  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    from chess_llm.train.sft import load_unsloth_model

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

    ds = parquet_to_grpo_dataset(Path(cfg["rl_path"]), tokenizer, cfg.get("max_samples"))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(),
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            num_train_epochs=float(cfg.get("num_train_epochs", 1)),
            max_steps=int(cfg.get("max_steps", 1000)),
            logging_steps=int(cfg.get("logging_steps", 5)),
            save_steps=int(cfg.get("save_steps", 100)),
            max_completion_length=int(cfg.get("max_new_tokens", 24)),
            num_generations=int(cfg.get("num_generations", 4)),
            bf16=True,
            seed=int(cfg.get("seed", 42)),
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"saved GRPO adapters to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
