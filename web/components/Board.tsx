"use client";

import { useMemo } from "react";
import type { Square } from "chess.js";
import Piece from "./Piece";

type BoardProps = {
  fen: string;
  orientation: "w" | "b";
  selected: Square | null;
  legal: Square[];
  lastFrom: Square | null;
  lastTo: Square | null;
  inCheck: boolean;
  kingSq: Square | null;
  onSquare: (sq: Square) => void;
};

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

function parsePieces(fen: string): Map<Square, { color: "w" | "b"; type: string }> {
  const [placement] = fen.split(" ");
  const rows = placement.split("/");
  const pieces = new Map<Square, { color: "w" | "b"; type: string }>();
  for (let r = 0; r < 8; r += 1) {
    let col = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") {
        col += Number(ch);
        continue;
      }
      const file = String.fromCharCode(97 + col);
      const rank = 8 - r;
      pieces.set(`${file}${rank}` as Square, {
        color: ch === ch.toUpperCase() ? "w" : "b",
        type: ch.toLowerCase(),
      });
      col += 1;
    }
  }
  return pieces;
}

export default function Board({
  fen,
  orientation,
  selected,
  legal,
  lastFrom,
  lastTo,
  inCheck,
  kingSq,
  onSquare,
}: BoardProps) {
  const ranks = orientation === "w" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = orientation === "w" ? FILES : [...FILES].reverse();
  const pieces = useMemo(() => parsePieces(fen), [fen]);

  return (
    <div className="board">
      {ranks.flatMap((rank) =>
        files.map((file) => {
          const sq = `${file}${rank}` as Square;
          const dark = (file.charCodeAt(0) + rank) % 2 === 1;
          const occ = pieces.get(sq);
          const isLast = sq === lastFrom || sq === lastTo;
          const isCheck = inCheck && kingSq === sq;
          return (
            <button
              key={sq}
              type="button"
              className={`sq ${dark ? "dark" : "light"} ${selected === sq ? "sel" : ""} ${isLast ? "last" : ""} ${isCheck ? "check" : ""}`}
              onClick={() => onSquare(sq)}
            >
              {file === files[0] && <span className="coord rank">{rank}</span>}
              {rank === ranks[ranks.length - 1] && <span className="coord file">{file}</span>}
              {legal.includes(sq) && <span className="plus" />}
              {occ && <Piece piece={occ.type} color={occ.color} />}
            </button>
          );
        }),
      )}
    </div>
  );
}
