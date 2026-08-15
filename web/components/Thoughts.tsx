import type { Thought } from "@/lib/types";

function evalLabel(t: Thought | null) {
  if (!t) return "—";
  if (t.mate != null) return `M${t.mate}`;
  if (t.cp == null) return "—";
  const pawns = t.cp / 100;
  return `${pawns >= 0 ? "+" : ""}${pawns.toFixed(2)}`;
}

function fillPct(t: Thought | null) {
  if (!t) return 50;
  if (t.mate != null) return t.mate > 0 ? 100 : 0;
  const cp = Math.max(-800, Math.min(800, t.cp ?? 0));
  return ((cp + 800) / 1600) * 100;
}

export default function Thoughts({
  link,
  backend,
  statusRaw,
  latest,
  history,
}: {
  link: "UP" | "DOWN";
  backend: string;
  statusRaw: string;
  latest: Thought | null;
  history: Thought[];
}) {
  return (
    <aside className="panel">
      <h2>ghost process // qwen-4b</h2>
      <div className={`link ${link === "UP" ? "up" : "down"}`}>
        NEURAL LINK: {link} · {backend.toUpperCase()}
      </div>
      <div className="row">
        <span className="k">MOVE</span>
        <span className={`v ${latest && !latest.legal ? "bad" : ""}`}>{latest?.uci ?? "—"}</span>
      </div>
      <div className="eval-col">
        <div className="eval-bar" title="side-to-move eval">
          <div className="eval-fill" style={{ height: `${fillPct(latest)}%` }} />
        </div>
        <div className="eval-meta">
          <div className="row">
            <span className="k">EVAL</span>
            <span className="v">{evalLabel(latest)}</span>
          </div>
          <div className="row">
            <span className="k">MS</span>
            <span className="v">{latest ? Math.round(latest.latency_ms) : "—"}</span>
          </div>
          <div className="row">
            <span className="k">LEGAL</span>
            <span className={`v ${latest && !latest.legal ? "bad" : ""}`}>
              {latest ? String(latest.legal).toUpperCase() : "—"}
            </span>
          </div>
        </div>
      </div>
      <div className="k">RAW STREAM</div>
      <div className="crt">
        {statusRaw || "> awaiting position"}
        <span className="caret">█</span>
      </div>
      {latest && <div className="fen">FEN {latest.fen}</div>}
      <h2 style={{ marginTop: 12 }}>thought scrollback</h2>
      <div className="crt">
        {history.length === 0 && <div className="logline">{"// empty"}</div>}
        {[...history].reverse().map((t) => (
          <div key={`${t.ply}-${t.raw}`} className={`logline ${t.hallucination ? "bad" : ""}`}>
            [{t.ply}] {t.uci ?? "????"} {t.mate != null ? `mate:${t.mate}` : `cp:${t.cp ?? "?"}`} {Math.round(t.latency_ms)}ms
            {t.hallucination ? "  HALLUCINATION" : ""}
            {t.stub ? "  STUB" : ""}
          </div>
        ))}
      </div>
    </aside>
  );
}
