from __future__ import annotations

import math

from chess_llm.data.format import eval_to_cp, is_legal_uci, parse_assistant_reply


def grpo_reward(
    completion: str,
    fen: str,
    target_uci: str,
    target_cp: int | None,
    target_mate: int | None,
) -> float:
    parsed = parse_assistant_reply(completion)
    if parsed is None:
        return 0.0
    if not is_legal_uci(fen, parsed.uci):
        return 0.0

    score = 0.5
    if parsed.uci == target_uci.lower():
        score += 1.5

    pred_cp = eval_to_cp(parsed.cp, parsed.mate)
    true_cp = eval_to_cp(target_cp, target_mate)
    if pred_cp is not None and true_cp is not None:
        score += 0.5 * math.exp(-abs(pred_cp - true_cp) / 100.0)
    return score
