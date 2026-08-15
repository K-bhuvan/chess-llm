"""Offline eval metrics: legality, Stockfish PV top-1, median |cp| error."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from chess_llm.data.format import eval_to_cp, is_legal_uci


def legal_rate(preds: Sequence[dict[str, Any]]) -> float:
    if not preds:
        return 0.0
    n = 0
    for row in preds:
        uci = row.get("uci")
        fen = row.get("fen")
        if uci and fen and is_legal_uci(str(fen), str(uci)):
            n += 1
    return n / len(preds)


def top1_match(preds: Sequence[dict[str, Any]]) -> float:
    comparable = [p for p in preds if p.get("uci") and p.get("target_uci")]
    if not comparable:
        return 0.0
    hits = sum(
        1
        for p in comparable
        if str(p["uci"]).lower() == str(p["target_uci"]).lower()
    )
    return hits / len(comparable)


def median_cp_error(preds: Sequence[dict[str, Any]]) -> float | None:
    errors: list[float] = []
    for p in preds:
        pred = eval_to_cp(p.get("cp"), p.get("mate"))
        target = eval_to_cp(p.get("target_cp"), p.get("target_mate"))
        if pred is None or target is None:
            continue
        errors.append(abs(pred - target))
    if not errors:
        return None
    return float(statistics.median(errors))


def summarize(preds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(preds),
        "legal_rate": legal_rate(preds),
        "top1": top1_match(preds),
        "median_cp_error": median_cp_error(preds),
    }


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def evaluate_file(
    path: Path,
    generate,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = _load_parquet(path)
    if limit is not None:
        rows = rows[:limit]
    preds: list[dict[str, Any]] = []
    for row in rows:
        raw = generate(row["fen"])
        from chess_llm.data.format import parse_assistant_reply

        parsed = parse_assistant_reply(raw) if isinstance(raw, str) else raw
        uci = parsed.uci if parsed is not None and hasattr(parsed, "uci") else None
        cp = parsed.cp if parsed is not None and hasattr(parsed, "cp") else None
        mate = parsed.mate if parsed is not None and hasattr(parsed, "mate") else None
        if isinstance(raw, dict):
            uci = raw.get("uci")
            cp = raw.get("cp")
            mate = raw.get("mate")
        preds.append(
            {
                "fen": row["fen"],
                "uci": uci,
                "target_uci": row["uci"],
                "cp": cp,
                "target_cp": row.get("cp"),
                "mate": mate,
                "target_mate": row.get("mate"),
            }
        )
    return summarize(preds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chess LLM on a parquet split")
    parser.add_argument("--data", type=Path, default=Path("data/test.parquet"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval.json"))
    args = parser.parse_args(argv)

    if args.adapter:
        from chess_llm.infer import load_generator

        generate = load_generator(args.adapter)
    else:
        from chess_llm.infer import stub_generate

        generate = stub_generate

    summary = evaluate_file(args.data, generate, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
