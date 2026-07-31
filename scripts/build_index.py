"""Build the FAISS index from papers in data/papers/.

Examples:
  # download a couple of papers, then build
  python scripts/build_index.py --arxiv 1706.03762 2005.11401
  # build from whatever PDFs are already in data/papers/
  python scripts/build_index.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import build_index          # noqa: E402
from src.ingest import download_arxiv       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", nargs="*", default=[], help="arXiv ids to download first")
    args = ap.parse_args()

    if args.arxiv:
        print("Downloading papers...")
        download_arxiv(args.arxiv)
    build_index()


if __name__ == "__main__":
    main()
