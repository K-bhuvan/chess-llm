from chess_llm.eval.metrics import legal_rate, median_cp_error, summarize, top1_match

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_legal_rate():
    preds = [
        {"fen": STARTPOS, "uci": "e2e4"},
        {"fen": STARTPOS, "uci": "e2e5"},
        {"fen": STARTPOS, "uci": None},
    ]
    assert legal_rate(preds) == 1 / 3


def test_top1_match_ignores_missing():
    preds = [
        {"uci": "e2e4", "target_uci": "e2e4"},
        {"uci": "d2d4", "target_uci": "e2e4"},
        {"uci": None, "target_uci": "e2e4"},
    ]
    assert top1_match(preds) == 0.5


def test_median_cp_error():
    preds = [
        {"cp": 10, "target_cp": 0, "mate": None, "target_mate": None},
        {"cp": 30, "target_cp": 0, "mate": None, "target_mate": None},
        {"cp": 50, "target_cp": 0, "mate": None, "target_mate": None},
    ]
    assert median_cp_error(preds) == 30


def test_summarize_keys():
    summary = summarize(
        [
            {
                "fen": STARTPOS,
                "uci": "e2e4",
                "target_uci": "e2e4",
                "cp": 40,
                "target_cp": 30,
                "mate": None,
                "target_mate": None,
            }
        ]
    )
    assert summary["n"] == 1
    assert summary["legal_rate"] == 1.0
    assert summary["top1"] == 1.0
    assert summary["median_cp_error"] == 10
