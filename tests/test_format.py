from chess_llm.data.format import (
    build_assistant_reply,
    build_user_prompt,
    eval_to_cp,
    first_pv_move,
    is_legal_uci,
    messages_for,
    normalize_fen,
    parse_assistant_reply,
    side_to_move_cp,
    side_to_move_mate,
)

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
FOUR_FIELD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def test_pads_four_field_fen():
    assert normalize_fen(FOUR_FIELD) == STARTPOS


def test_white_pov_cp_unchanged_when_white_to_move():
    assert side_to_move_cp(35, STARTPOS) == 35


def test_negates_cp_when_black_to_move():
    assert side_to_move_cp(35, BLACK_TO_MOVE) == -35


def test_negates_mate_when_black_to_move():
    assert side_to_move_mate(3, BLACK_TO_MOVE) == -3
    assert side_to_move_mate(3, STARTPOS) == 3


def test_eval_to_cp_prefers_mate_as_large_score():
    assert eval_to_cp(cp=None, mate=2) == 9998
    assert eval_to_cp(cp=None, mate=-2) == -9998
    assert eval_to_cp(cp=40, mate=None) == 40


def test_assistant_reply_cp_uses_side_to_move():
    assert build_assistant_reply("e7e5", cp=12, mate=None, fen=BLACK_TO_MOVE) == "e7e5 cp:-12"


def test_assistant_reply_mate():
    assert build_assistant_reply("f7f8q", cp=None, mate=1, fen=STARTPOS) == "f7f8q mate:1"


def test_parse_cp_reply():
    parsed = parse_assistant_reply("e2e4 cp:35")
    assert parsed.uci == "e2e4"
    assert parsed.cp == 35
    assert parsed.mate is None


def test_parse_mate_reply_and_noise():
    parsed = parse_assistant_reply("  e7e8q mate:-2\n")
    assert parsed.uci == "e7e8q"
    assert parsed.mate == -2
    assert parsed.cp is None


def test_parse_garbage_returns_none():
    assert parse_assistant_reply("I play the Sicilian") is None


def test_user_prompt_includes_fen_and_schema():
    prompt = build_user_prompt(STARTPOS)
    assert STARTPOS in prompt
    assert "cp:" in prompt
    assert "mate:" in prompt


def test_messages_for_chat_roles():
    messages = messages_for(STARTPOS, "e2e4", cp=40, mate=None)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "e2e4 cp:40"


def test_first_pv_move_is_uci():
    assert first_pv_move("g2e4 f7f5 e4b7") == "g2e4"


def test_legal_uci_startpos_e2e4():
    assert is_legal_uci(STARTPOS, "e2e4") is True
    assert is_legal_uci(STARTPOS, "e2e5") is False
