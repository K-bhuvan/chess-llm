"use client";

import { Chess, type Square } from "chess.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Board from "@/components/Board";
import Rain from "@/components/Rain";
import Thoughts from "@/components/Thoughts";
import { fetchHealth, requestMove } from "@/lib/api";
import type { Thought } from "@/lib/types";

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function sanList(game: Chess) {
  const tmp = new Chess();
  const lines: string[] = [];
  for (const m of game.history({ verbose: true })) {
    const ply = tmp.moveNumber();
    const isWhite = tmp.turn() === "w";
    tmp.move(m);
    if (isWhite) lines.push(`${ply}. ${m.san}`);
    else lines[lines.length - 1] += ` ${m.san}`;
  }
  return lines;
}

export default function Home() {
  const [fen, setFen] = useState(START);
  const [orientation, setOrientation] = useState<"w" | "b">("w");
  const [selected, setSelected] = useState<Square | null>(null);
  const [busy, setBusy] = useState(false);
  const [raw, setRaw] = useState("");
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [link, setLink] = useState<"UP" | "DOWN">("DOWN");
  const [backend, setBackend] = useState("offline");
  const [status, setStatus] = useState("WAITING FOR HUMAN");
  const asked = useRef<string | null>(null);

  const game = useMemo(() => new Chess(fen), [fen]);
  const humanSide = orientation;
  const latest = thoughts[thoughts.length - 1] ?? null;

  const last = game.history({ verbose: true }).at(-1);
  const kingSq = useMemo(() => {
    const board = game.board();
    for (const row of board) {
      for (const cell of row) {
        if (cell && cell.type === "k" && cell.color === game.turn()) return cell.square;
      }
    }
    return null;
  }, [game]);

  const legalTargets = useMemo(() => {
    if (!selected) return [];
    return game.moves({ square: selected, verbose: true }).map((m) => m.to);
  }, [game, selected]);

  const refreshLink = useCallback(async () => {
    const h = await fetchHealth();
    if (!h) {
      setLink("DOWN");
      setBackend("client-stub");
      return;
    }
    setLink("UP");
    setBackend(h.backend);
  }, []);

  useEffect(() => {
    void refreshLink();
    const id = setInterval(() => void refreshLink(), 8000);
    return () => clearInterval(id);
  }, [refreshLink]);

  const askModel = useCallback(
    async (position: string, ply: number) => {
      setBusy(true);
      setStatus("SYNTHESIZING");
      setRaw("> querying neural net…");
      const res = await requestMove(position);
      setRaw(res.raw || "> empty completion");
      const thought: Thought = {
        ...res,
        ply,
        fen: position,
        hallucination: !res.legal,
      };
      setThoughts((prev) => [...prev, thought]);
      if (!res.legal || !res.uci) {
        setStatus("HALLUCINATION");
        setBusy(false);
        return;
      }
      const next = new Chess(position);
      const from = res.uci.slice(0, 2) as Square;
      const to = res.uci.slice(2, 4) as Square;
      const promotion = res.uci.length > 4 ? res.uci[4] : undefined;
      const moved = next.move({ from, to, promotion });
      if (!moved) {
        setStatus("HALLUCINATION");
        setBusy(false);
        return;
      }
      setFen(next.fen());
      if (next.isCheckmate()) setStatus("MATE");
      else if (next.isDraw()) setStatus("DRAW");
      else if (next.isCheck()) setStatus("CHECK");
      else setStatus("WAITING FOR HUMAN");
      setBusy(false);
    },
    [],
  );

  const onSquare = (sq: Square) => {
    if (busy) return;
    if (game.isGameOver()) return;
    if (game.turn() !== humanSide) return;
    if (selected) {
      const promo = game
        .moves({ square: selected, verbose: true })
        .find((m) => m.to === sq && m.promotion);
      const moved = game.move({
        from: selected,
        to: sq,
        promotion: promo ? "q" : undefined,
      });
      setSelected(null);
      if (moved) {
        const nextFen = game.fen();
        setFen(nextFen);
        if (game.isGameOver()) {
          setStatus(game.isCheckmate() ? "MATE" : "DRAW");
          return;
        }
        return;
      }
    }
    const occ = game.get(sq);
    if (occ && occ.color === humanSide) setSelected(sq);
  };

  useEffect(() => {
    if (busy || game.isGameOver()) return;
    if (game.turn() === humanSide) return;
    if (asked.current === fen) return;
    asked.current = fen;
    void askModel(fen, game.history().length + 1);
  }, [askModel, busy, fen, game, humanSide]);

  const reset = () => {
    asked.current = null;
    setFen(START);
    setSelected(null);
    setThoughts([]);
    setRaw("");
    setBusy(false);
    setStatus("WAITING FOR HUMAN");
  };

  const retry = () => {
    if (busy || game.turn() === humanSide) return;
    asked.current = null;
    void askModel(fen, game.history().length + 1);
  };

  const tickerClass =
    status === "HALLUCINATION" || status === "MATE" || status === "CHECK" ? "ticker alert" : status === "WAITING FOR HUMAN" ? "ticker wait" : "ticker";

  return (
    <>
      <Rain />
      <main className="shell">
        <aside className="panel">
          <h2>pgn stream</h2>
          <div className="moves">
            {sanList(game).length === 0 && <div className="logline">{"// 1. ?"}</div>}
            {sanList(game).map((line) => (
              <div key={line}>
                <span className="num">{line.split(" ")[0]}</span>
                {line.slice(line.indexOf(" ") + 1)}
              </div>
            ))}
          </div>
          <div className="btns" style={{ justifyContent: "flex-start" }}>
            <button type="button" onClick={reset}>
              NEW GAME
            </button>
            <button type="button" onClick={() => setOrientation((o) => (o === "w" ? "b" : "w"))}>
              FLIP
            </button>
          </div>
        </aside>

        <section className="board-wrap">
          <div className={tickerClass}>{status}</div>
          <Board
            fen={fen}
            orientation={orientation}
            selected={selected}
            legal={legalTargets}
            lastFrom={(last?.from as Square) ?? null}
            lastTo={(last?.to as Square) ?? null}
            inCheck={game.inCheck()}
            kingSq={kingSq}
            onSquare={onSquare}
          />
          <div className="btns">
            <button type="button" onClick={retry} disabled={busy || game.turn() === humanSide}>
              RETRY
            </button>
          </div>
        </section>

        <Thoughts
          link={link}
          backend={backend}
          statusRaw={raw}
          latest={latest}
          history={thoughts}
        />
      </main>
    </>
  );
}
