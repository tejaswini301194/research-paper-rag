"""Ask questions against the indexed papers.

  python scripts/ask.py "What attention mechanism does the paper propose?"
  python scripts/ask.py            # interactive loop
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import load_index   # noqa: E402
from src.qa import answer          # noqa: E402


def _show(result):
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} [{s['section']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="question to ask")
    args = ap.parse_args()

    store = load_index()

    if args.question:
        _show(answer(store, args.question))
        return

    print("Ask a question (empty line or Ctrl-D to quit):")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        _show(answer(store, q))


if __name__ == "__main__":
    main()
