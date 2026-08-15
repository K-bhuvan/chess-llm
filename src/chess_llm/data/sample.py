"""Stream Lichess Stockfish evals into disjoint parquet splits."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chess_llm.data.format import first_pv_move, is_legal_uci, normalize_fen

log = logging.getLogger(__name__)

DATASET_ID = "Lichess/chess-position-evaluations"
DEFAULT_TOTAL = 2_090_000
DEFAULT_TRAIN = 2_000_000
DEFAULT_VAL = 20_000
DEFAULT_TEST = 20_000
DEFAULT_RL = 50_000


@dataclass(frozen=True)
class Position:
    fen: str
    uci: str
    cp: int | None
    mate: int | None
    depth: int


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def accept_row(row: dict, min_depth: int = 20) -> Position | None:
    depth = _as_int(row.get("depth"))
    if depth is None or depth < min_depth:
        return None
    line = str(row.get("line") or "").strip()
    if not line:
        return None
    uci = first_pv_move(line)
    if not uci:
        return None
    fen = normalize_fen(str(row.get("fen") or ""))
    if not is_legal_uci(fen, uci):
        return None
    return Position(
        fen=fen,
        uci=uci.lower(),
        cp=_as_int(row.get("cp")),
        mate=_as_int(row.get("mate")),
        depth=depth,
    )


def fen_hash(fen: str) -> int:
    return int(hashlib.sha256(fen.encode("utf-8")).hexdigest(), 16)


def should_keep(fen: str, sample_mod: int = 1, sample_bucket: int = 0) -> bool:
    if sample_mod <= 1:
        return True
    return fen_hash(fen) % sample_mod == sample_bucket


def split_positions(
    positions: list[Position],
    train_n: int,
    val_n: int,
    test_n: int,
    rl_n: int,
    seed: int = 42,
) -> dict[str, list[Position]]:
    needed = train_n + val_n + test_n + rl_n
    if len(positions) < needed:
        raise ValueError(f"need {needed} positions, got {len(positions)}")
    rng = random.Random(seed)
    shuffled = positions[:needed]
    rng.shuffle(shuffled)
    i = 0
    train = shuffled[i : i + train_n]
    i += train_n
    val = shuffled[i : i + val_n]
    i += val_n
    test = shuffled[i : i + test_n]
    i += test_n
    rl = shuffled[i : i + rl_n]
    return {"train": train, "val": val, "test": test, "rl": rl}


def write_parquet(path: Path, positions: Iterable[Position], chunk_size: int = 50_000) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("fen", pa.string()),
            ("uci", pa.string()),
            ("cp", pa.int32()),
            ("mate", pa.int32()),
            ("depth", pa.int32()),
        ]
    )
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    written = 0

    def flush() -> None:
        nonlocal writer, batch, written
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(path, schema)
        writer.write_table(table)
        written += len(batch)
        batch = []

    try:
        for pos in positions:
            batch.append(asdict(pos))
            if len(batch) >= chunk_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    return written


def iter_accepted(
    rows: Iterable[dict],
    min_depth: int,
    limit: int,
    sample_mod: int = 11,
) -> Iterator[Position]:
    seen: set[int] = set()
    kept = 0
    for row in rows:
        pos = accept_row(row, min_depth=min_depth)
        if pos is None:
            continue
        if not should_keep(pos.fen, sample_mod=sample_mod):
            continue
        key = fen_hash(pos.fen)
        if key in seen:
            continue
        seen.add(key)
        yield pos
        kept += 1
        if kept >= limit:
            return


def stream_lichess_rows(dataset_id: str = DATASET_ID):
    from datasets import load_dataset

    return load_dataset(dataset_id, split="train", streaming=True)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sample Stockfish-labeled chess positions")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=int, default=DEFAULT_VAL)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST)
    parser.add_argument("--rl", type=int, default=DEFAULT_RL)
    parser.add_argument("--sample-mod", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=DATASET_ID)
    args = parser.parse_args(argv)

    total = args.train + args.val + args.test + args.rl
    log.info("streaming up to %s unique positions from %s", total, args.dataset)
    collected: list[Position] = []
    for pos in iter_accepted(
        stream_lichess_rows(args.dataset),
        min_depth=args.min_depth,
        limit=total,
        sample_mod=args.sample_mod,
    ):
        collected.append(pos)
        if len(collected) % 10_000 == 0:
            log.info("accepted %s / %s", len(collected), total)

    splits = split_positions(
        collected,
        train_n=args.train,
        val_n=args.val,
        test_n=args.test,
        rl_n=args.rl,
        seed=args.seed,
    )
    for name, rows in splits.items():
        path = args.out_dir / f"{name}.parquet"
        n = write_parquet(path, rows)
        log.info("wrote %s rows -> %s", n, path)
    log.info("done")
    # datasets/pyarrow can abort the interpreter on shutdown in some torch envs
    os._exit(0)


if __name__ == "__main__":
    main()
