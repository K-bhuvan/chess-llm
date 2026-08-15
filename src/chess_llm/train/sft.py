"""Unsloth QLoRA SFT on Stockfish-labeled (FEN → move + eval) data."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from chess_llm.data.format import messages_for
from chess_llm.train.config import load_yaml


def parquet_to_sft_dataset(path: Path, tokenizer, max_samples: int | None = None) -> Dataset:
    ds = Dataset.from_parquet(str(path))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_text(batch):
        texts = []
        for fen, uci, cp, mate in zip(batch["fen"], batch["uci"], batch["cp"], batch["mate"]):
            messages = messages_for(fen, uci, cp if cp is not None else None, mate)
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
        return {"text": texts}

    return ds.map(to_text, batched=True, remove_columns=ds.column_names)


def latest_complete_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    complete = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.split("-")[-1]
        if not suffix.isdigit():
            continue
        if (path / "adapter_model.safetensors").exists() or (path / "model.safetensors").exists():
            complete.append((int(suffix), path))
    if not complete:
        return None
    complete.sort()
    return complete[-1][1]


def ensure_special_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token is None or tokenizer.eos_token not in vocab:
        for candidate in ("<|im_end|>", "<|endoftext|>"):
            if candidate in vocab:
                tokenizer.eos_token = candidate
                break
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_unsloth_model(cfg: dict, apply_lora: bool = True):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
    )
    if apply_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg["lora_r"]),
            target_modules=list(cfg["target_modules"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg.get("lora_dropout", 0)),
            bias="none",
            use_gradient_checkpointing=cfg.get("gradient_checkpointing", "unsloth"),
            random_state=int(cfg.get("seed", 42)),
        )
    ensure_special_tokens(tokenizer)
    return model, tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for chess Qwen-4B")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    import unsloth  # noqa: F401  # must precede trl/transformers
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_unsloth_model(cfg)
    train_ds = parquet_to_sft_dataset(
        Path(cfg["train_path"]), tokenizer, cfg.get("max_samples")
    )
    val_path = Path(cfg["val_path"])
    eval_ds = None
    if not args.no_eval and val_path.exists():
        eval_ds = parquet_to_sft_dataset(val_path, tokenizer, min(2000, cfg.get("max_samples") or 2000))

    resume_from = latest_complete_checkpoint(cfg["output_dir"])
    sft_args = dict(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            num_train_epochs=float(cfg["num_train_epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            logging_steps=int(cfg.get("logging_steps", 20)),
            save_steps=int(cfg.get("save_steps", 500)),
            save_total_limit=int(cfg.get("save_total_limit", 3)),
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=int(cfg.get("eval_steps", 500)),
            overwrite_output_dir=resume_from is None,
            save_only_model=True,
            bf16=True,
            fp16=False,
            packing=bool(cfg.get("packing", True)),
            max_length=int(cfg["max_seq_length"]),
            dataset_text_field="text",
            eos_token=tokenizer.eos_token,
            seed=int(cfg.get("seed", 42)),
            report_to="none",
    )
    if cfg.get("max_steps"):
        sft_args["max_steps"] = int(cfg["max_steps"])
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_args),
    )
    if resume_from is not None:
        print(f"resuming SFT from {resume_from}")
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"saved adapters to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
