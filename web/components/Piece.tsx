type PieceProps = { piece: string; color: "w" | "b" };

const STROKE_W = "#00ff41";
const FILL_W = "#b6ffc4";
const STROKE_B = "#00ff41";
const FILL_B = "#031403";

export default function Piece({ piece, color }: PieceProps) {
  const stroke = color === "w" ? STROKE_W : STROKE_B;
  const fill = color === "w" ? FILL_W : FILL_B;
  const common = { fill, stroke, strokeWidth: 1.4, strokeLinejoin: "round" as const };
  return (
    <svg className="piece" viewBox="0 0 45 45" aria-hidden>
      {piece === "p" && (
        <path {...common} d="M22.5 9c-2.2 0-4 1.8-4 4 0 1.1.5 2.1 1.3 2.8C17.1 17.3 15 20.4 15 24c0 2.2 1.3 4.1 3.2 5.1C16 30.4 14 33.2 14 36.5h17c0-3.3-2-6.1-4.2-7.4 1.9-1 3.2-2.9 3.2-5.1 0-3.6-2.1-6.7-4.8-8.2.8-.7 1.3-1.7 1.3-2.8 0-2.2-1.8-4-4-4z" />
      )}
      {piece === "r" && (
        <path {...common} d="M9 39h27v-3H9v3zm3-4.5v-9h3v-5h4v5h7v-5h4v5h3v9H12zm1.5-22V9h4v3.5h3V9h6v3.5h3V9h4v3.5H33v7H12v-7h1.5z" />
      )}
      {piece === "n" && (
        <path {...common} d="M 22 10 C 32 10 36 18 34 24 C 32 29 28 31 27 36 L 14 36 C 14 30 16 28 13 24 C 10 20 12 14 18 12 L 16 16 L 19 15 C 20 12 22 10 22 10 Z" />
      )}
      {piece === "b" && (
        <>
          <path {...common} d="M22.5 8c-1.7 0-3 1.5-3 3.2 0 1 .4 1.9 1.1 2.4C16 16.6 12.5 22 12.5 27.5 12.5 33 16.5 37 22.5 37s10-4 10-9.5c0-5.5-3.5-10.9-8.1-13.9.7-.5 1.1-1.4 1.1-2.4 0-1.7-1.3-3.2-3-3.2z" />
          <path fill="none" stroke={stroke} strokeWidth="1.2" d="M15 27.5h15M22.5 15v16" />
        </>
      )}
      {piece === "q" && (
        <path {...common} d="M8 38h29v-2.5H8V38zm3-5l2.2-16L18 24l4.5-18 4.5 18 4.8-7.5L36 33H11z" />
      )}
      {piece === "k" && (
        <>
          <path {...common} d="M11 38h23v-3H11v3zm2.5-6c0-6 4-11 9-14 5 3 9 8 9 14H13.5z" />
          <path fill="none" stroke={stroke} strokeWidth="1.6" d="M22.5 6v10M18 11h9" />
        </>
      )}
    </svg>
  );
}
