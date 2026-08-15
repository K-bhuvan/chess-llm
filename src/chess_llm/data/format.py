"""FEN chat formatting, Stockfish White-POV conversion, and reply parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess

MATE_CP = 10_000

REPLY_RE = re.compile(
    r"([a-h][1-8][a-h][1-8][qrbnQRBN]?)\s*(?:(cp):(-?\d+)|(mate):(-?\d+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedMove:
    uci: str
    cp: int | None
    mate: int | None


def normalize_fen(fen: str) -> str:
    fen = fen.strip()
    parts = fen.split()
    if len(parts) == 4:
        return f"{fen} 0 1"
    return fen


def _black_to_move(fen: str) -> bool:
    parts = normalize_fen(fen).split()
    return len(parts) >= 2 and parts[1] == "b"


def side_to_move_cp(cp: int | None, fen: str) -> int | None:
    if cp is None:
        return None
    return -cp if _black_to_move(fen) else cp


def side_to_move_mate(mate: int | None, fen: str) -> int | None:
    if mate is None:
        return None
    return -mate if _black_to_move(fen) else mate


def eval_to_cp(cp: int | None, mate: int | None) -> int | None:
    if mate is not None:
        if mate > 0:
            return MATE_CP - mate
        return -MATE_CP - mate
    return cp


def build_user_prompt(fen: str) -> str:
    return (
        f"FEN: {normalize_fen(fen)}\n"
        "Reply with: <uci> cp:<int>   or   <uci> mate:<int>"
    )


def build_assistant_reply(
    uci: str,
    cp: int | None,
    mate: int | None,
    fen: str,
) -> str:
    stm_mate = side_to_move_mate(mate, fen)
    if stm_mate is not None:
        return f"{uci} mate:{stm_mate}"
    stm_cp = side_to_move_cp(cp, fen)
    value = 0 if stm_cp is None else stm_cp
    return f"{uci} cp:{value}"


def parse_assistant_reply(text: str) -> ParsedMove | None:
    match = REPLY_RE.search(text.strip())
    if not match:
        return None
    uci = match.group(1).lower()
    if match.group(2):
        return ParsedMove(uci=uci, cp=int(match.group(3)), mate=None)
    if match.group(4):
        return ParsedMove(uci=uci, cp=None, mate=int(match.group(5)))
    return ParsedMove(uci=uci, cp=None, mate=None)


def messages_for(
    fen: str,
    uci: str,
    cp: int | None,
    mate: int | None,
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": build_user_prompt(fen)},
        {"role": "assistant", "content": build_assistant_reply(uci, cp, mate, fen)},
    ]


def first_pv_move(line: str) -> str:
    return line.strip().split()[0] if line.strip() else ""


def is_legal_uci(fen: str, uci: str) -> bool:
    try:
        board = chess.Board(normalize_fen(fen))
        move = chess.Move.from_uci(uci)
        return move in board.legal_moves
    except (ValueError, chess.InvalidFenError, chess.IllegalMoveError):
        return False


def random_other_legal_uci(fen: str, chosen: str, rng) -> str | None:
    board = chess.Board(normalize_fen(fen))
    options = [m.uci() for m in board.legal_moves if m.uci() != chosen.lower()]
    if not options:
        return None
    return rng.choice(options)
