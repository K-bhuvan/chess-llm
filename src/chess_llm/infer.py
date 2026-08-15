"""Generate a chess reply: LoRA model if adapters exist, otherwise a legal stub."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import chess

from chess_llm.data.format import build_user_prompt, is_legal_uci, normalize_fen
from chess_llm.train.config import resolve_adapter


def stub_generate(fen: str) -> str:
    board = chess.Board(normalize_fen(fen))
    moves = list(board.legal_moves)
    if not moves:
        return "0000 cp:0"
    move = random.choice(moves)
    return f"{move.uci()} cp:0"


def load_generator(adapter: Path | None = None) -> Callable[[str], str]:
    adapter_path = resolve_adapter(adapter)
    if adapter_path is None:
        return stub_generate
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        return stub_generate

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=256,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def generate(fen: str) -> str:
        messages = [{"role": "user", "content": build_user_prompt(fen)}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FEN in, UCI + eval out")
    parser.add_argument("--fen", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args(argv)
    generate = stub_generate if args.stub else load_generator(args.adapter)
    raw = generate(args.fen)
    print(raw)
    print("legal", is_legal_uci(args.fen, raw.split()[0] if raw.split() else ""))


if __name__ == "__main__":
    main()
