import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEURAL CHESS // QWEN-4B",
  description: "Matrix terminal chess vs a Stockfish-distilled Qwen-4B",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="scanlines" />
        {children}
      </body>
    </html>
  );
}
