from chess_llm.data.format import parse_assistant_reply
from chess_llm.serve_logic import decide_move, health_payload


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_stub_move_is_legal_and_matches_contract():
    result = decide_move(STARTPOS, generate=lambda _fen: "e2e4 cp:40")
    assert result["uci"] == "e2e4"
    assert result["cp"] == 40
    assert result["mate"] is None
    assert result["legal"] is True
    assert result["raw"] == "e2e4 cp:40"
    assert "latency_ms" in result


def test_illegal_model_output_is_flagged_not_replaced():
    result = decide_move(STARTPOS, generate=lambda _fen: "e2e5 cp:0")
    assert result["legal"] is False
    assert result["uci"] == "e2e5"


def test_health_payload_reports_backend():
    payload = health_payload(model_loaded=False, backend="stub")
    assert payload["ok"] is True
    assert payload["model_loaded"] is False
    assert payload["backend"] == "stub"


def test_parse_used_by_server_contract():
    parsed = parse_assistant_reply("g1f3 cp:-12")
    assert parsed is not None
    assert parsed.uci == "g1f3"
