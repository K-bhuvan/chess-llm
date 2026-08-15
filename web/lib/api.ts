import { Chess } from "chess.js";
import type { HealthResponse, MoveResponse } from "./types";

const API = process.env.NEXT_PUBLIC_INFERENCE_URL ?? "http://127.0.0.1:8000";

function mockMove(fen: string): MoveResponse {
  const game = new Chess(fen);
  const legal = game.moves({ verbose: true });
  if (!legal.length) {
    return { uci: null, cp: null, mate: null, raw: "0000 cp:0", legal: false, latency_ms: 0, stub: true };
  }
  const pick = legal[Math.floor(Math.random() * legal.length)];
  const uci = `${pick.from}${pick.to}${pick.promotion ?? ""}`;
  return {
    uci,
    cp: 0,
    mate: null,
    raw: `${uci} cp:0`,
    legal: true,
    latency_ms: 12,
    stub: true,
  };
}

export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${API}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as HealthResponse;
  } catch {
    return null;
  }
}

export async function requestMove(fen: string): Promise<MoveResponse> {
  try {
    const res = await fetch(`${API}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen }),
    });
    if (!res.ok) return mockMove(fen);
    return (await res.json()) as MoveResponse;
  } catch {
    return mockMove(fen);
  }
}
