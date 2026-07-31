"""Generate an eval set and/or score retrieval with Recall@K and MRR.

  # generate 30 synthetic QA pairs from the index, save them, then score
  python scripts/run_eval.py --generate 30 --k 4
  # score an existing eval set at several K values
  python scripts/run_eval.py --k 1 3 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import load_index                                   # noqa: E402
from src.evaluate import (                                         # noqa: E402
    build_eval_set, save_eval_set, load_eval_set, evaluate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", type=int, metavar="N",
                    help="generate N synthetic QA pairs and save them")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                    help="K values to score (default: 1 3 5)")
    args = ap.parse_args()

    store = load_index()

    if args.generate:
        print(f"Generating {args.generate} QA pairs...")
        eval_set = build_eval_set(store, n=args.generate)
        save_eval_set(eval_set)
        print(f"Saved {len(eval_set)} pairs.")
    else:
        eval_set = load_eval_set()

    print(f"\nScoring {len(eval_set)} queries\n" + "=" * 40)
    for k in args.k:
        print(json.dumps(evaluate(store, eval_set, k), indent=2))


if __name__ == "__main__":
    main()
