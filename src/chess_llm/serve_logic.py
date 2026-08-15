"""Pure move-decision logic used by FastAPI (easy to unit test)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from chess_llm.data.format import is_legal_uci, parse_assistant_reply


def decide_move(fen: str, generate: Callable[[str], str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = generate(fen)
    parsed = parse_assistant_reply(raw)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    if parsed is None:
        return {
            "uci": None,
            "cp": None,
            "mate": None,
            "raw": raw,
            "legal": False,
            "latency_ms": latency_ms,
        }
    legal = is_legal_uci(fen, parsed.uci)
    return {
        "uci": parsed.uci,
        "cp": parsed.cp,
        "mate": parsed.mate,
        "raw": raw.strip(),
        "legal": legal,
        "latency_ms": latency_ms,
    }


def health_payload(*, model_loaded: bool, backend: str) -> dict[str, Any]:
    return {"ok": True, "model_loaded": model_loaded, "backend": backend}
