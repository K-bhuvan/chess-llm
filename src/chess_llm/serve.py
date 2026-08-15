"""Local GPU (or stub) inference API for the Matrix chess UI."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chess_llm.infer import load_generator, stub_generate
from chess_llm.serve_logic import decide_move, health_payload
from chess_llm.train.config import resolve_adapter

app = FastAPI(title="chess-llm", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_generate: Callable[[str], str] | None = None
_backend = "unloaded"


class MoveRequest(BaseModel):
    fen: str
    stream: bool = False


def get_generate() -> Callable[[str], str]:
    global _generate, _backend
    if _generate is not None:
        return _generate
    adapter = resolve_adapter()
    if adapter is None:
        _generate = stub_generate
        _backend = "stub"
        return _generate
    try:
        _generate = load_generator(adapter)
        _backend = "lora" if _generate is not stub_generate else "stub"
    except Exception:
        _generate = stub_generate
        _backend = "stub"
    return _generate


@app.get("/health")
def health():
    adapter = resolve_adapter()
    backend = _backend if _generate is not None else ("lora" if adapter else "stub")
    return health_payload(model_loaded=adapter is not None, backend=backend)


@app.post("/move")
def move(req: MoveRequest):
    return decide_move(req.fen, get_generate())


def main() -> None:
    import uvicorn

    get_generate()
    uvicorn.run("chess_llm.serve:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
