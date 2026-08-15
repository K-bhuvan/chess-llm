"""Stream a few Lichess eval rows and print one accepted position. For WSL."""
from chess_llm.data.sample import accept_row, stream_lichess_rows

def main() -> None:
    n = 0
    for row in stream_lichess_rows():
        n += 1
        pos = accept_row(dict(row), min_depth=20)
        if pos is not None:
            print("ok", pos)
            print("scanned", n)
            return
        if n >= 5000:
            print("no accept in first", n)
            return

if __name__ == "__main__":
    main()
