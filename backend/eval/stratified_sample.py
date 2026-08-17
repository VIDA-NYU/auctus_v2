"""Draw a stratified 100-dataset sample from a catalog_frame.py snapshot.

Pure function of the snapshot file and a recorded seed — no live catalog
access (tasks.md 3.6). Re-running against the same snapshot and seed
reproduces the same slice exactly.

Two-axis stratification (design.md D3):

1. **By `domain_category`, proportionally**, via largest-remainder
   apportionment. A category with at least one member in the catalog gets at
   least one seat even if strict proportionality would round it to zero —
   the "small category" question design.md leaves open, decided here: floor
   every non-empty category at 1, funded by trimming one seat off the
   category(ies) with the largest remainder headroom. Recorded per-round in
   the slice's `parameters` block so a different call is a diffable decision,
   not a silent behaviour change.
2. **Within each category, by description-length decile**, computed once
   over the whole catalog so the sample's overall length distribution tracks
   the catalog's rather than each category's own (possibly skewed) shape.
   A category with fewer members in a decile than its sub-target simply
   contributes what it has; the shortfall is redistributed to the category's
   other deciles before being redistributed across categories.

    python -m eval.stratified_sample --snapshot eval/frame/catalog_snapshot_....json \
        --figures eval/frame/catalog_figures_....json --out-dir eval/frame
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.build_corpus_slice import SHORT_DESCRIPTION_CHARS

LOGGER = logging.getLogger("stratified_sample")

SAMPLE_SIZE = 100
# Recorded, not re-rolled per run. Changing the round's draw means changing
# this constant deliberately and re-running, which is itself the record.
DRAW_SEED = 20260816


def _decile_edges(lens: list[int]) -> list[float]:
    if len(lens) < 2:
        return []
    return statistics.quantiles(sorted(lens), n=10)


def _decile_bucket(length: int, edges: list[float]) -> int:
    """Return 0-9 for which global decile bucket `length` falls into."""
    bucket = 0
    for edge in edges:
        if length > edge:
            bucket += 1
        else:
            break
    return min(bucket, 9)


def _largest_remainder(shares: dict[Any, int], n: int) -> dict[Any, int]:
    """Largest-remainder (Hamilton) apportionment of `n` seats across `shares`
    (key -> weight), summing to exactly `n`. No minimum-seat guarantee — a
    zero-weight key gets zero seats."""
    total = sum(shares.values())
    if total <= 0 or n <= 0:
        return {k: 0 for k in shares}

    raw = {k: (v / total) * n for k, v in shares.items()}
    targets = {k: int(r) for k, r in raw.items()}
    remainders = sorted(shares.keys(), key=lambda k: raw[k] - targets[k], reverse=True)

    allocated = sum(targets.values())
    i = 0
    while allocated < n and i < len(remainders):
        targets[remainders[i]] += 1
        allocated += 1
        i += 1

    return targets


def _apportion_categories(shares: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder apportionment across categories, then floor every
    non-empty category at 1 seat (design decision, see module docstring),
    funded from the largest current holder(s)."""
    targets = _largest_remainder(shares, n)

    zero_cats = [c for c, v in targets.items() if v == 0 and shares[c] > 0]
    for cat in zero_cats:
        donor = max(targets, key=lambda c: targets[c])
        if targets[donor] <= 1:
            break  # nothing left to trim from; leave this category at 0
        targets[donor] -= 1
        targets[cat] = 1

    return targets


def _draw_category(
    rng: random.Random,
    records: list[dict],
    target: int,
    global_edges: list[float],
) -> list[dict]:
    """Draw `target` records from one category's pool, distributed across the
    global length deciles proportionally to what that category actually has
    in each decile.

    Decile seats are apportioned by largest-remainder (not round-then-consume
    in size order) so a decile with few members — often the short-description
    one, since short descriptions cluster in the bottom deciles — cannot be
    starved by rounding overflow in the larger deciles processed first. Any
    decile's shortfall against its capped availability is redistributed to
    whichever other decile in the same category still has spare capacity.
    """
    by_decile: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_decile[_decile_bucket(r["description_len"], global_edges)].append(r)

    pool_size = len(records)
    if pool_size == 0 or target <= 0:
        return []
    target = min(target, pool_size)

    decile_shares = {d: len(items) for d, items in by_decile.items()}
    decile_targets = _largest_remainder(decile_shares, target)

    shortfall = 0
    spare_capacity: dict[int, int] = {}
    for d, want in decile_targets.items():
        capped = min(want, decile_shares[d])
        shortfall += want - capped
        decile_targets[d] = capped
        spare_capacity[d] = decile_shares[d] - capped

    while shortfall > 0:
        candidates = [d for d, cap in spare_capacity.items() if cap > 0]
        if not candidates:
            break
        d = max(candidates, key=lambda x: spare_capacity[x])
        decile_targets[d] += 1
        spare_capacity[d] -= 1
        shortfall -= 1

    drawn: list[dict] = []
    for d, want in decile_targets.items():
        if want > 0:
            drawn.extend(rng.sample(by_decile[d], want))

    return drawn


def draw_sample(
    catalog_records: list[dict],
    n: int = SAMPLE_SIZE,
    seed: int = DRAW_SEED,
) -> dict[str, Any]:
    rng = random.Random(seed)
    global_edges = _decile_edges([r["description_len"] for r in catalog_records])

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in catalog_records:
        by_category[r["domain_category"] or "(uncategorized)"].append(r)

    category_shares = {cat: len(items) for cat, items in by_category.items()}
    category_targets = _apportion_categories(category_shares, n)

    drawn: list[dict] = []
    for cat, target in category_targets.items():
        drawn.extend(_draw_category(rng, by_category[cat], target, global_edges))

    # Largest-remainder can under/overshoot by a seat or two once categories
    # are capped by availability; true up against the requested n by trimming
    # or topping up from the full remaining pool, deterministically.
    if len(drawn) > n:
        drawn = rng.sample(drawn, n)
    elif len(drawn) < n:
        drawn_ids = {r["id"] for r in drawn}
        leftover_pool = [r for r in catalog_records if r["id"] not in drawn_ids]
        top_up = min(n - len(drawn), len(leftover_pool))
        if top_up:
            drawn.extend(rng.sample(leftover_pool, top_up))

    realized_lens = [r["description_len"] for r in drawn]
    realized_short = sum(1 for l in realized_lens if l < SHORT_DESCRIPTION_CHARS)
    realized_categories: dict[str, int] = defaultdict(int)
    realized_agency: dict[str, int] = defaultdict(int)
    for r in drawn:
        realized_categories[r["domain_category"] or "(uncategorized)"] += 1
        realized_agency[r["agency"] or "(no agency)"] += 1

    top_agency_count = max(realized_agency.values()) if realized_agency else 0

    return {
        "stage": "drawn_not_ingested",
        "seed": seed,
        "requested_n": n,
        "drawn_n": len(drawn),
        "targets": {
            "category_targets": category_targets,
        },
        "realized": {
            "category_counts": dict(realized_categories),
            "short_description_count": realized_short,
            "short_description_share": realized_short / len(drawn) if drawn else 0.0,
            "top_agency_share": (top_agency_count / len(drawn)) if drawn else 0.0,
            "agency_counts": dict(sorted(realized_agency.items(), key=lambda kv: kv[1], reverse=True)),
        },
        "datasets": sorted(drawn, key=lambda r: r["id"]),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out-dir", default="eval/frame")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DRAW_SEED)
    args = parser.parse_args(argv)

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    result = draw_sample(snapshot["records"], n=args.n, seed=args.seed)
    result["domain"] = snapshot["domain"]
    result["source_snapshot"] = args.snapshot
    result["drawn_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    round_date = snapshot["pulled_at"][:10]
    out_path = Path(args.out_dir) / f"drawn_slice_{snapshot['domain']}_{round_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info(
        "Drew %d/%d datasets. short_description_share=%.3f top_agency_share=%.3f",
        result["drawn_n"],
        result["requested_n"],
        result["realized"]["short_description_share"],
        result["realized"]["top_agency_share"],
    )
    LOGGER.info("Category counts: %s", result["realized"]["category_counts"])
    LOGGER.info("Slice -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
