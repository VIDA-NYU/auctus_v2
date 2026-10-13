"""How big should the profiling sample be, and where should it come from?

For each file with more rows than the smallest sample size: profile the whole file
once (the reference), then profile the first N rows and a random N rows (seeded,
original row order kept) for each N. Every profile goes through the connector's
own code in a separate process. Each sample is compared with the reference:
- column-type agreement (coarse classes, per column present in both);
- temporal span recall: overlap of the sample's time range with the full range,
  as a share of the full range;
- bbox recall: overlap of the sample's map box with the full box, as a share of
  the full box's area;
- profiling time.

  python -m eval.profiler_benchmark.run sampling-sweep --harvest ../../data/ultraviolet
"""

from __future__ import annotations

import csv
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_SIZES = (500, 2000, 5000)
STRATEGIES = ("head", "random")
FULL_MAX_ROWS = 200_000


def read_full(path: str) -> pd.DataFrame:
    """The whole table (up to FULL_MAX_ROWS) as strings, as the connector would parse it."""
    from crawlers.invenio.transformer import detect_separator

    p = Path(path)
    ext = p.name.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        return pd.read_excel(p, nrows=FULL_MAX_ROWS, dtype=str)
    return pd.read_csv(p, sep=detect_separator(p, ext), nrows=FULL_MAX_ROWS, dtype=str,
                       keep_default_na=False, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")


def write_sample(frame: pd.DataFrame, out_dir: Path, name: str) -> str:
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return str(path)


def _span(temporal: dict | None) -> tuple[date, date] | None:
    if not temporal or not temporal.get("start") or not temporal.get("end"):
        return None
    return date.fromisoformat(temporal["start"]), date.fromisoformat(temporal["end"])


def temporal_recall(sample: dict | None, full: dict | None) -> float | None:
    f, s = _span(full), _span(sample)
    if not f:
        return None
    if not s:
        return 0.0
    total = (f[1] - f[0]).days
    if total <= 0:
        return 1.0 if s[0] <= f[0] <= s[1] else 0.0
    overlap = (min(s[1], f[1]) - max(s[0], f[0])).days
    return round(max(0, overlap) / total, 3)


def bbox_recall(sample: list | None, full: list | None) -> float | None:
    if not full:
        return None
    if not sample:
        return 0.0
    w, s_, e, n = max(sample[0], full[0]), max(sample[1], full[1]), min(sample[2], full[2]), min(sample[3], full[3])
    area = (full[2] - full[0]) * (full[3] - full[1])
    if area <= 0:
        return 1.0
    return round(max(0.0, e - w) * max(0.0, n - s_) / area, 3)


def type_agreement(sample: dict, full: dict) -> tuple[int, int]:
    """(columns with the same coarse type, columns compared)."""
    full_types = {c["name"]: c["coarse"] for c in full.get("columns", [])}
    shared = [(c["coarse"], full_types[c["name"]]) for c in sample.get("columns", []) if c["name"] in full_types]
    return sum(a == b for a, b in shared), len(shared)


def sweep(corpus: list[dict], jobs: int, timeout: int, sizes=DEFAULT_SIZES, seed: int = 0) -> list[dict]:
    from eval.profiler_benchmark.run import run_many

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        plan = []  # (file, strategy, size, spec)
        for i, f in enumerate(corpus):
            try:
                frame = read_full(f["path"])
            except Exception as exc:
                rows.append({"path": f["path"], "status": f"unreadable: {exc!r}"[:120]})
                continue
            if len(frame) <= min(sizes):
                continue
            full_path = write_sample(frame, tmp, f"{i}_full")
            plan.append((f, "full", len(frame), {"path": full_path, "mode": "full", "full_max_rows": FULL_MAX_ROWS + 1,
                                                 "full_max_bytes": 10**10}))
            for size in sizes:
                if len(frame) <= size:
                    continue
                for strategy in STRATEGIES:
                    sample = frame.head(size) if strategy == "head" else frame.sample(n=size, random_state=seed).sort_index()
                    path = write_sample(sample, tmp, f"{i}_{strategy}_{size}")
                    plan.append((f, strategy, size, {"path": path, "mode": "full", "full_max_rows": size + 1,
                                                     "full_max_bytes": 10**10}))
        results = run_many([spec for *_, spec in plan], jobs, timeout, "sampling-sweep")

    fulls = {f["path"]: r for (f, strategy, _, _), r in zip(plan, results) if strategy == "full"}
    for (f, strategy, size, _), result in zip(plan, results):
        if strategy == "full":
            continue
        full = fulls.get(f["path"], {})
        row = {"path": f["path"], "strategy": strategy, "size": size, "full_rows": full.get("nb_rows"),
               "status": result.get("status"), "seconds": result.get("seconds")}
        if result.get("status") == "ok" and full.get("status") == "ok":
            same, compared = type_agreement(result, full)
            row.update({"types_same": same, "types_compared": compared,
                        "temporal_recall": temporal_recall(result.get("pipeline_temporal"), full.get("pipeline_temporal")),
                        "bbox_recall": bbox_recall(result.get("pipeline_bbox"), full.get("pipeline_bbox"))})
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """One row per (strategy, size): averages over the files where that size applies."""
    out = []
    for size in sorted({r["size"] for r in rows if "size" in r}):
        for strategy in STRATEGIES:
            sel = [r for r in rows if r.get("size") == size and r.get("strategy") == strategy and "types_compared" in r]
            if not sel:
                continue
            temporal = [r["temporal_recall"] for r in sel if r["temporal_recall"] is not None]
            bbox = [r["bbox_recall"] for r in sel if r["bbox_recall"] is not None]
            secs = sorted(r["seconds"] for r in sel if r.get("seconds") is not None)
            out.append({
                "size": size, "strategy": strategy, "files": len(sel),
                "type_agreement": round(sum(r["types_same"] for r in sel) / max(1, sum(r["types_compared"] for r in sel)), 3),
                "files_with_type_changes": sum(r["types_same"] < r["types_compared"] for r in sel),
                "temporal_files": len(temporal),
                "temporal_recall_mean": round(sum(temporal) / len(temporal), 3) if temporal else None,
                "temporal_recall_min": min(temporal) if temporal else None,
                "bbox_files": len(bbox),
                "bbox_recall_mean": round(sum(bbox) / len(bbox), 3) if bbox else None,
                "median_seconds": secs[len(secs) // 2] if secs else None,
            })
    return out


def report(summary: list[dict], rows: list[dict]) -> str:
    lines = ["# Sampling sweep: first N rows vs random N rows, compared with the whole file", "",
             "| rows | strategy | files | column types same as whole file | files with any type change | "
             "time-span recall (mean / worst) | map-box recall (mean) | median s |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        t = f"{s['temporal_recall_mean']} / {s['temporal_recall_min']} ({s['temporal_files']} files)" if s["temporal_files"] else "-"
        b = f"{s['bbox_recall_mean']} ({s['bbox_files']} files)" if s["bbox_files"] else "-"
        lines.append(f"| {s['size']} | {s['strategy']} | {s['files']} | {s['type_agreement']:.1%} | "
                     f"{s['files_with_type_changes']} | {t} | {b} | {s['median_seconds']} |")
    failures = [r for r in rows if r.get("status") not in (None, "ok")]
    if failures:
        lines += ["", "Not compared:"] + [f"- `{Path(r['path']).name}` {r.get('strategy', '')} {r.get('size', '')}: {r['status']}"
                                          for r in failures]
    return "\n".join(lines) + "\n"
