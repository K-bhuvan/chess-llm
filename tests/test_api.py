from fastapi.testclient import TestClient

from chess_llm.serve import app


def test_health_and_move_stub():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["backend"] in {"stub", "lora", "unloaded"}

    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = client.post("/move", json={"fen": start})
    assert move.status_code == 200
    payload = move.json()
    assert "uci" in payload
    assert "legal" in payload
    assert "raw" in payload
    if payload["legal"]:
        assert payload["uci"]
