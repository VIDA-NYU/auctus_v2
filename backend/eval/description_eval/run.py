"""CLI: python -m eval.description_eval.run <stage> [...]

  gold [--records ID ...] [--overwrite]   build gold/<record>.json (curator fields + extracted dictionaries)
  collect                                 pull AutoDDG/profiler output for every gold record
  compare [--include-unverified]          run the four comparisons, write results/
  human-sample [--n 30]                   write a CSV of judge decisions for a person to grade
  human-score                             agreement between the judge and the person's grades

Runs inside the arq-worker container (portkey, minio, sentence-transformers); see
run_in_worker.sh. Needs PORTKEY_API_KEY and the NYU VPN for LLM stages.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="description_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("gold")
    p.add_argument("--records", nargs="+")
    p.add_argument("--overwrite", action="store_true", help="rebuild even hand-verified files")
    sub.add_parser("collect")
    p = sub.add_parser("compare")
    p.add_argument("--include-unverified", action="store_true",
                   help="use unverified dictionaries too (for a dry run; never for reported numbers)")
    p = sub.add_parser("human-sample")
    p.add_argument("--n", type=int, default=30)
    sub.add_parser("human-score")
    args = parser.parse_args()

    if args.stage == "gold":
        from eval.description_eval import gold
        gold.main(args.records, args.overwrite)
    elif args.stage == "collect":
        from eval.description_eval import collect
        collect.main()
    elif args.stage == "compare":
        from eval.description_eval import compare
        compare.main(include_unverified=args.include_unverified)
    elif args.stage == "human-sample":
        from eval.description_eval import human_check
        human_check.sample(args.n)
    elif args.stage == "human-score":
        from eval.description_eval import human_check
        human_check.score()


if __name__ == "__main__":
    main()
