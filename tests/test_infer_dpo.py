import random

from chess_llm.data.format import is_legal_uci, parse_assistant_reply, random_other_legal_uci
from chess_llm.infer import stub_generate

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_stub_generate_is_legal():
    raw = stub_generate(STARTPOS)
    parsed = parse_assistant_reply(raw)
    assert parsed is not None
    assert is_legal_uci(STARTPOS, parsed.uci)


def test_dpo_rejected_differs_and_is_legal():
    other = random_other_legal_uci(STARTPOS, "e2e4", random.Random(0))
    assert other is not None
    assert other != "e2e4"
    assert is_legal_uci(STARTPOS, other)
