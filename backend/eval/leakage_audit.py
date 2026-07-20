"""Leakage audit: does any query have 'home advantage' on one description arm?

A query generated from arm X's prose would trivially match X and let X win by
construction (report F1). We generate from neutral facts precisely to avoid this,
and this audit is the check: for each query, measure token containment against
EVERY arm's text on the query's source dataset, and flag a query whose best arm
is far above the rest. Flagged queries are candidates to quarantine before scoring.

Containment = |query_tokens ∩ arm_tokens| / |query_tokens| (how much of the query
the arm text already contains). Reported per arm; a query is flagged when
(max − second-max) containment exceeds --gap on a single arm.

    python -m eval.leakage_audit --queries eval/benchmark/queries.json \
        --out eval/benchmark/leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from storage.opensearch_client import (
    AUCTUS_INDEX_NAME,
    DESCRIPTION_SOURCE_FIELDS,
    get_client,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of for in on to and or with is are be by from at as data "
                  "dataset find get list show me i need can where how what".split())


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def containment(query: str, arm_text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(arm_text)) / len(q)


def audit(os_client, queries: list[dict], gap: float) -> dict:
    arms = list(DESCRIPTION_SOURCE_FIELDS)  # original, ufd, sfd, llm_direct, profile_only, t_od_s
    fields = list(DESCRIPTION_SOURCE_FIELDS.values())
    doc_cache: dict[str, dict] = {}

    per_query = []
    arm_totals = {a: 0.0 for a in arms}
    flagged = []
    for q in queries:
        did = q["source_dataset_id"]
        if did not in doc_cache:
            doc_cache[did] = os_client.get(
                index=AUCTUS_INDEX_NAME, id=did, _source=fields
            ).get("_source") or {}
        src = doc_cache[did]
        scores = {a: round(containment(q["text"], src.get(DESCRIPTION_SOURCE_FIELDS[a], "")), 3)
                  for a in arms}
        for a, s in scores.items():
            arm_totals[a] += s
        ordered = sorted(scores.values(), reverse=True)
        home_gap = round(ordered[0] - ordered[1], 3) if len(ordered) > 1 else ordered[0]
        best_arm = max(scores, key=scores.get)
        entry = {"query_id": q["query_id"], "scores": scores,
                 "best_arm": best_arm, "home_gap": home_gap}
        per_query.append(entry)
        if home_gap > gap:
            flagged.append(entry)

    n = len(queries) or 1
    return {
        "gap_threshold": gap,
        "n_queries": len(queries),
        "mean_containment_by_arm": {a: round(arm_totals[a] / n, 3) for a in arms},
        "flagged_count": len(flagged),
        "flagged": flagged,
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/benchmark/queries.json")
    parser.add_argument("--out", default="eval/benchmark/leakage_audit.json")
    parser.add_argument("--gap", type=float, default=0.5,
                        help="flag a query if best-arm containment exceeds the next by > gap")
    args = parser.parse_args(argv)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    report = audit(get_client(), queries, args.gap)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Leakage audit: {report['flagged_count']}/{report['n_queries']} queries flagged "
          f"(home_gap > {args.gap})")
    print(f"mean containment by arm: {report['mean_containment_by_arm']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
