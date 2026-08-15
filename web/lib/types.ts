export type MoveResponse = {
  uci: string | null;
  cp: number | null;
  mate: number | null;
  raw: string;
  legal: boolean;
  latency_ms: number;
  stub?: boolean;
};

export type HealthResponse = {
  ok: boolean;
  model_loaded: boolean;
  backend: string;
};

export type Thought = MoveResponse & {
  ply: number;
  fen: string;
  hallucination: boolean;
};
