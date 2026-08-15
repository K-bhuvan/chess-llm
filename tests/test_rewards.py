from chess_llm.data.rewards import grpo_reward

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_unparseable_completion_scores_zero():
    assert grpo_reward("nah", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None) == 0.0


def test_illegal_uci_scores_zero():
    assert grpo_reward("e2e5 cp:10", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None) == 0.0


def test_legal_non_matching_move_gets_legal_bonus_plus_eval():
    score = grpo_reward("d2d4 cp:35", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None)
    # legal 0.5 + eval proximity 0.5 * exp(0) = 1.0
    assert abs(score - 1.0) < 1e-6


def test_pv_match_adds_one_point_five():
    score = grpo_reward("e2e4 cp:35", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None)
    # 0.5 legal + 1.5 pv + 0.5 eval = 2.5
    assert abs(score - 2.5) < 1e-6


def test_eval_proximity_decays_with_cp_error():
    close = grpo_reward("d2d4 cp:40", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None)
    far = grpo_reward("d2d4 cp:335", fen=STARTPOS, target_uci="e2e4", target_cp=35, target_mate=None)
    assert close > far
    assert close < 1.0
    assert far > 0.5
