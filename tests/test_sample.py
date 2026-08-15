from chess_llm.data.sample import accept_row, split_positions


def test_reject_shallow_depth():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "line": "e2e4 e7e5",
        "depth": 12,
        "cp": 30,
        "mate": None,
    }
    assert accept_row(row, min_depth=20) is None


def test_reject_empty_pv():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "line": "",
        "depth": 22,
        "cp": 30,
        "mate": None,
    }
    assert accept_row(row, min_depth=20) is None


def test_accept_deep_legal_pv_and_first_move():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "line": "e2e4 e7e5 g1f3",
        "depth": 22,
        "cp": 30,
        "mate": None,
    }
    pos = accept_row(row, min_depth=20)
    assert pos is not None
    assert pos.uci == "e2e4"
    assert pos.cp == 30
    assert "0 1" in pos.fen


def test_reject_illegal_pv_first_move():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "line": "e2e5 e7e5",
        "depth": 30,
        "cp": 0,
        "mate": None,
    }
    assert accept_row(row, min_depth=20) is None


def test_split_positions_are_disjoint_and_sized():
    rows = [
        {
            "fen": f"8/8/8/8/8/8/8/4K2{'k' if i % 2 == 0 else 'K'} {'w' if i % 2 == 0 else 'b'} - - 0 {i}",
            "line": "e1e2" if i % 2 == 0 else "e8e7",
            "depth": 25,
            "cp": i,
            "mate": None,
        }
        for i in range(100)
    ]
    # The dummy FENs above may be illegal; use accept_row-free constructor via split on Positions
    from chess_llm.data.sample import Position

    positions = [
        Position(fen=f"fen-{i}", uci="e2e4", cp=i, mate=None, depth=25)
        for i in range(100)
    ]
    splits = split_positions(
        positions,
        train_n=70,
        val_n=10,
        test_n=10,
        rl_n=10,
        seed=0,
    )
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert len(splits["rl"]) == 10
    fens = [p.fen for group in splits.values() for p in group]
    assert len(fens) == len(set(fens))
