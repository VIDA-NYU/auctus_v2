"""Check every number and date in UV AutoDDG descriptions against the profile (no LLM).

Uses check_numbers from Claudio's AutoDDG branch `description-report`
(autoddg/evaluation/numbers.py): each number is "match" (as written, or a bound / range that
fits), "close" (within 2%) or "missing" (nothing in the profile, sample or title supports it).

Our profiles are atlas-profiler (auctus-fixes) output, so they're mapped to AutoDDG's
profile_metadata shape first: min/max -> minimum/maximum, and the file's temporal coverage
-> one date column (our profiles don't say which column the dates come from). k-means
coverage clusters are left out, as in AutoDDG.

Two sets:
- arm_a: the 69 files of the semtypes run, arm A (today's AutoDDG input), next to the
  Claude judge's claims on the same text, so the two can be compared claim by claim;
- production: every UV dataset's stored autoddg_description (what search shows).

  python -m eval.description_eval.number_check [--bbox] [--out numbers]

Needs autoddg from the description-report branch, or AUTODDG_NUMBERS=<path to numbers.py>.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from eval.description_eval.collect import load_profile
from eval.description_eval.common import DATA_DIR, UV_DOMAIN, read_json, write_json

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def check_numbers_fn():
    try:
        from autoddg.evaluation.numbers import check_numbers
        return check_numbers
    except ImportError:
        path = os.environ.get("AUTODDG_NUMBERS")
        if not path:
            raise SystemExit("needs autoddg >= description-report, or AUTODDG_NUMBERS=<numbers.py>")
        spec = importlib.util.spec_from_file_location("autoddg_numbers", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
        return module.check_numbers


def _timestamp(day: str) -> float | None:
    try:
        return datetime.fromisoformat(day[:10]).replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def to_autoddg_metadata(profile: dict[str, Any], bbox: bool = False) -> dict[str, Any]:
    """Our atlas profile in the shape autoddg.profiling.profile_metadata returns.

    With bbox, the file's bounding box (computed from every geometry by our connector, so
    exact) becomes a longitude and a latitude column: shapefiles and GeoJSON have no
    coordinate columns for the check to find."""
    pm = profile.get("profiler_metadata") or {}
    columns = []
    for col in pm.get("columns") or []:
        if not isinstance(col, dict):
            continue
        entry = {"name": col.get("name")}
        for ours, theirs in (("min", "minimum"), ("max", "maximum"), ("mean", "mean"),
                             ("num_distinct_values", "num_distinct_values"),
                             ("missing_values_ratio", "missing_values_ratio")):
            if col.get(ours) is not None:
                entry[theirs] = col[ours]
        columns.append(entry)
    tc = profile.get("temporal_coverage") or {}
    earliest, latest = _timestamp(tc.get("start")), _timestamp(tc.get("end"))
    if earliest is not None and latest is not None:
        columns.append({"name": "dates (file temporal coverage)", "earliest": earliest, "latest": latest})
    box = pm.get("spatial_bbox") or profile.get("spatial_bbox")
    if bbox and isinstance(box, list) and len(box) == 4:
        columns.append({"name": "longitude (bounding box)", "minimum": box[0], "maximum": box[2]})
        columns.append({"name": "latitude (bounding box)", "minimum": box[1], "maximum": box[3]})
    return {"nb_rows": pm.get("nb_rows"), "nb_columns": pm.get("nb_columns"), "columns": columns}


def check(fn, description: str, profile: dict[str, Any], bbox: bool = False) -> list[dict[str, Any]]:
    found = fn(description, to_autoddg_metadata(profile, bbox), sample=profile.get("sample"),
               references={"dataset name": profile.get("title")})
    return [{"text": c.text, "value": c.value, "status": c.status, "source": c.source} for c in found]


def claim_vs_judge(claims: list[dict], checks: list[dict]) -> list[dict[str, Any]]:
    """Each judge claim with numbers, and what the number check says about those numbers."""
    by_value: dict[float, set[str]] = {}
    for c in checks:
        by_value.setdefault(c["value"], set()).add(c["status"])
    out = []
    for claim in claims:
        numbers = [float(n.replace(",", "")) for n in _NUMBER.findall(claim.get("claim", ""))]
        statuses = [by_value.get(n) for n in numbers if n in by_value]
        if not statuses:
            continue
        verdict = "missing" if any("missing" in s for s in statuses) else "match"
        out.append({"claim": claim.get("claim"), "judge_supported": bool(claim.get("supported")), "numbers": verdict})
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(c["status"] for r in rows for c in r["checks"])
    with_numbers = [r for r in rows if r["checks"]]
    return {
        "descriptions": len(rows),
        "with_numbers": len(with_numbers),
        "numbers": sum(statuses.values()),
        "status": dict(statuses),
        "missing_rate": round(statuses["missing"] / max(sum(statuses.values()), 1), 3),
        "descriptions_with_a_missing_number": sum(any(c["status"] == "missing" for c in r["checks"]) for r in rows),
    }


def production_ids() -> list[str]:
    from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

    hits = get_client().search(index=AUCTUS_INDEX_NAME, body={
        "size": 2000, "_source": False,
        "query": {"bool": {"filter": [{"term": {"domain": UV_DOMAIN}}, {"exists": {"field": "autoddg_description"}}]}}})
    return [h["_id"] for h in hits["hits"]["hits"]]


def main() -> None:
    import argparse

    from storage.minio_client import get_storage_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", action="store_true", help="count the file's bounding box as coordinate facts")
    parser.add_argument("--out", default="numbers", help="output folder under the eval data dir")
    args = parser.parse_args()
    out = DATA_DIR / args.out

    fn = check_numbers_fn()
    storage = get_storage_client()
    profiles: dict[str, dict] = {}

    def profile_of(dataset_id: str) -> dict | None:
        if dataset_id not in profiles:
            try:
                profiles[dataset_id] = load_profile(storage, dataset_id)
            except Exception:
                profiles[dataset_id] = None
        return profiles[dataset_id]

    # 1. arm A of the semtypes run, next to the judge
    semtypes = read_json(DATA_DIR / "semtypes" / "results.json")["rows"]
    arm_a = []
    for r in semtypes:
        if r["arm"] != "A" or r.get("error") or not (p := profile_of(r["dataset_id"])):
            continue
        checks = check(fn, r["description"], p, args.bbox)
        arm_a.append({"record": r["record"], "dataset_id": r["dataset_id"], "file": r["file"],
                      "wide": (p["profiler_metadata"].get("nb_columns") or 0) >= 40,
                      "description": r["description"], "checks": checks,
                      "judge": claim_vs_judge(r.get("claims") or [], checks)})
    pairs = Counter((j["judge_supported"], j["numbers"]) for r in arm_a for j in r["judge"])

    # 2. every stored UV description
    production = []
    for dataset_id in production_ids():
        p = profile_of(dataset_id)
        if not p or not p.get("autoddg_description"):
            continue
        production.append({"dataset_id": dataset_id, "title": p.get("title"),
                           "wide": (p["profiler_metadata"].get("nb_columns") or 0) >= 40,
                           "description": p["autoddg_description"], "checks": check(fn, p["autoddg_description"], p, args.bbox)})

    summary = {
        "arm_a": summarize(arm_a),
        "arm_a_narrow": summarize([r for r in arm_a if not r["wide"]]),
        "arm_a_wide": summarize([r for r in arm_a if r["wide"]]),
        "judge_vs_numbers": {f"judge {'supported' if s else 'unsupported'} / numbers {n}": c
                             for (s, n), c in sorted(pairs.items())},
        "production": summarize(production),
        "production_narrow": summarize([r for r in production if not r["wide"]]),
        "production_wide": summarize([r for r in production if r["wide"]]),
    }
    write_json(out / "results.json", {"summary": summary, "arm_a": arm_a, "production": production})
    with open(out / "missing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "dataset_id", "wide", "number", "source", "context"])
        for name, rows in (("arm_a", arm_a), ("production", production)):
            for r in rows:
                for c in r["checks"]:
                    if c["status"] == "missing":
                        i = r["description"].find(c["text"])
                        w.writerow([name, r["dataset_id"], r["wide"], c["text"], c["source"],
                                    r["description"][max(0, i - 80):i + 60].replace("\n", " ")])
    with open(out / "judge_disagreements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "judge_supported", "numbers", "claim"])
        for r in arm_a:
            for j in r["judge"]:
                if j["judge_supported"] != (j["numbers"] == "match"):
                    w.writerow([r["dataset_id"], j["judge_supported"], j["numbers"], j["claim"]])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
