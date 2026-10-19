"""Metrics, the Markdown report, and baseline comparison for the profiler benchmark."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval.profiler_benchmark.run import BACKEND

# A metric may drop by at most this much before `compare` calls it a regression.
TOLERANCE = {"robustness.usable_rate": 0.005, "types.synthetic_accuracy": 0.0, "types.readme_accuracy": 0.02}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def robustness_metrics(rows: list[dict]) -> dict:
    def usable(r: dict) -> bool:  # ran cleanly and produced more than one column
        return r.get("status") == "ok" and (r.get("nb_columns") or 0) > 1

    by_key: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        c = by_key[f"{r['instance']}/{r['ext']}"]
        c["files"] += 1
        c["usable"] += usable(r)
        c[r.get("status", "missing")] += 1
        c["one_column"] += r.get("status") == "ok" and (r.get("nb_columns") or 0) <= 1
        c["profiler_temporal"] += bool(r.get("profiler_temporal"))
        c["pipeline_temporal"] += bool((r.get("pipeline_temporal") or {}).get("start"))
        c["profiler_spatial"] += bool(r.get("profiler_spatial"))
        c["pipeline_bbox"] += bool(r.get("pipeline_bbox"))
    ok = [r for r in rows if r.get("status") == "ok"]
    secs = sorted(r["seconds"] for r in ok)
    return {"files": len(rows), "usable": sum(map(usable, rows)), "usable_rate": _rate(sum(map(usable, rows)), len(rows)),
            "median_seconds": secs[len(secs) // 2] if secs else None,
            "max_peak_rss_mb": max((r["peak_rss_mb"] for r in ok), default=None),
            "by_instance_format": {k: dict(v) for k, v in sorted(by_key.items())},
            "failures": [{"path": r["path"], "status": r.get("status"), "detail": (r.get("errors") or r.get("stderr") or [""])[0][:160]}
                         for r in rows if not usable(r)]}


def types_metrics(synthetic: list[dict], readme: list[dict]) -> dict:
    scored = [r for r in readme if r["correct"] is not None]
    confusion = Counter(f"{r['expected']}->{r['got']}" for r in scored if not r["correct"])
    return {"synthetic_columns": len(synthetic), "synthetic_correct": sum(r["correct"] for r in synthetic),
            "synthetic_accuracy": _rate(sum(r["correct"] for r in synthetic), len(synthetic)),
            "synthetic_failures": sorted({f"{r['column']} (got {r['got']}, want {r['acceptable']})"
                                          for r in synthetic if not r["correct"] and r["case"] == "types.csv"}),
            "synthetic_by_case": {case: _rate(sum(r["correct"] for r in synthetic if r["case"] == case),
                                              sum(1 for r in synthetic if r["case"] == case))
                                  for case in sorted({r["case"] for r in synthetic})},
            "readme_columns": len(readme), "readme_scored": len(scored),
            "readme_verified_gold": all(r["verified"] for r in readme) if readme else None,
            "readme_accuracy": _rate(sum(r["correct"] for r in scored), len(scored)),
            "readme_strict_accuracy": _rate(sum(bool(r.get("strict_correct")) for r in scored), len(scored)),
            "readme_confusions": dict(confusion.most_common(10))}


def sampling_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    compared = sum(r["columns_compared"] for r in ok)
    recalls = [r["temporal_span_recall"] for r in ok if r.get("temporal_span_recall") is not None]
    return {"files": len(rows), "compared": len(ok), "column_type_change_rate": _rate(sum(r["type_changes"] for r in ok), compared),
            "files_with_type_changes": sum(1 for r in ok if r["type_changes"]),
            "temporal_files": len(recalls),
            "median_temporal_span_recall": sorted(recalls)[len(recalls) // 2] if recalls else None,
            "temporal_span_recall_below_0_5": sum(1 for x in recalls if x < 0.5)}


def summarize_dir(out: Path, run_info: dict) -> dict:
    commit = subprocess.run(["git", "-C", str(BACKEND), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    import atlas_profiler  # noqa: F401  (version recorded for provenance)
    from importlib.metadata import version

    summary = {"run": {**run_info, "commit": commit, "atlas_profiler": version("atlas-profiler"),
                       "pandas": version("pandas"), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    if (out / "robustness.jsonl").exists():
        summary["robustness"] = robustness_metrics(read_rows(out / "robustness.jsonl"))
    if (out / "types_synthetic.jsonl").exists():
        summary["types"] = types_metrics(read_rows(out / "types_synthetic.jsonl"), read_rows(out / "types_readme.jsonl"))
    if (out / "sampling.jsonl").exists():
        summary["sampling"] = sampling_metrics(read_rows(out / "sampling.jsonl"))
    if (out / "performance.jsonl").exists():
        summary["performance"] = read_rows(out / "performance.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text(report(summary))
    print(f"wrote {out / 'summary.json'} and {out / 'report.md'}")
    return summary


def report(s: dict) -> str:
    lines = [f"# Profiler benchmark report", "", f"Run: `{json.dumps(s['run'])}`", ""]
    if "robustness" in s:
        r = s["robustness"]
        lines += ["## 1. Robustness", "", f"Usable profiles: **{r['usable']}/{r['files']}** ({r['usable_rate']}). "
                  f"Median {r['median_seconds']} s per file; max peak memory {r['max_peak_rss_mb']} MB.", "",
                  "| instance/format | files | usable | errors | ≤1 column | profiler time cov. | pipeline time cov. | profiler spatial | pipeline bbox |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, c in r["by_instance_format"].items():
            errors = c["files"] - c.get("ok", 0)
            lines.append(f"| {key} | {c['files']} | {c['usable']} | {errors} | {c['one_column']} | {c['profiler_temporal']} | "
                         f"{c['pipeline_temporal']} | {c['profiler_spatial']} | {c['pipeline_bbox']} |")
        lines += [""] + [f"- `{Path(f['path']).name}`: {f['status']} {f['detail']}" for f in r["failures"][:25]] + [""]
    if "types" in s:
        t = s["types"]
        lines += ["## 2. Type correctness", "",
                  f"Synthetic (known answers): **{t['synthetic_correct']}/{t['synthetic_columns']}** ({t['synthetic_accuracy']}).", "",
                  "| synthetic file | accuracy |", "|---|---|"] + [f"| {k} | {v} |" for k, v in t["synthetic_by_case"].items()] + \
                 ["", "Wrong on the baseline file:"] + [f"- {x}" for x in t["synthetic_failures"]] + \
                 ["", f"README dictionaries: {t['readme_scored']} of {t['readme_columns']} columns scorable; accuracy "
                  f"**{t['readme_accuracy']}** (strict, README type only: {t['readme_strict_accuracy']}; gold verified: {t['readme_verified_gold']}). Most common errors: "
                  + ", ".join(f"{k} ({v})" for k, v in t["readme_confusions"].items()), ""]
    if "sampling" in s and not s["sampling"].get("compared"):
        lines += ["## 3. Sampling", "",
                  "Nothing to compare: robustness profiles whole files, so no file was truncated. "
                  "Use `run.py sampling-sweep` to measure how much a row sample would lose.", ""]
    elif "sampling" in s:
        m = s["sampling"]
        lines += ["## 3. Sampling (first 500 rows / 2 MB vs whole file)", "",
                  f"{m['compared']} files compared. Column types change on {m['files_with_type_changes']} files "
                  f"(rate {m['column_type_change_rate']} of columns). Temporal span recall median "
                  f"{m['median_temporal_span_recall']} over {m['temporal_files']} files; {m['temporal_span_recall_below_0_5']} "
                  f"files' samples cover less than half of the real time span.", ""]
    if "performance" in s:
        lines += ["## 4. Performance (whole table, one run at a time)", "", "| rows | columns | status | seconds | peak MB |", "|---|---|---|---|---|"]
        lines += [f"| {p['rows']} | {p['cols']} | {p['status']} | {p['seconds']} | {p['peak_rss_mb']} |" for p in s["performance"]]
    return "\n".join(lines) + "\n"


def compare(baseline_dir: Path, current_dir: Path) -> int:
    """Exit status 1 if any tracked metric regressed beyond tolerance, or a synthetic column broke."""
    base = json.loads((baseline_dir / "summary.json").read_text())
    cur = json.loads((current_dir / "summary.json").read_text())
    # Robustness is compared on the files both runs profiled, so runs over different
    # corpora (e.g. UltraViolet only vs UltraViolet + Caltech) stay comparable.
    base_files = {r["path"]: r for r in read_rows(baseline_dir / "robustness.jsonl")}
    cur_files = {r["path"]: r for r in read_rows(current_dir / "robustness.jsonl")}
    shared = sorted(base_files.keys() & cur_files.keys())
    if shared:
        base["robustness"] = robustness_metrics([base_files[p] for p in shared])
        cur["robustness"] = robustness_metrics([cur_files[p] for p in shared])
        print(f"robustness compared on {len(shared)} files present in both runs")
    regressions = []
    for dotted, tolerance in TOLERANCE.items():
        section, key = dotted.split(".")
        b, c = base.get(section, {}).get(key), cur.get(section, {}).get(key)
        if b is not None and c is not None:
            flag = "REGRESSION" if c < b - tolerance else "ok"
            print(f"{flag:<10} {dotted}: {b} -> {c}")
            if flag != "ok":
                regressions.append(dotted)
    passed = lambda d: {(r["case"], r["column"]) for r in read_rows(d / "types_synthetic.jsonl") if r["correct"]}
    broken = sorted(passed(baseline_dir) - passed(current_dir))
    for case, column in broken:
        print(f"REGRESSION synthetic {case}:{column} passed in baseline, fails now")
    usable = lambda files: {p for p in shared if files[p].get("status") == "ok" and (files[p].get("nb_columns") or 0) > 1}
    lost = sorted(usable(base_files) - usable(cur_files))
    gained = sorted(usable(cur_files) - usable(base_files))
    for path in gained[:20]:
        print(f"improved   file now usable: {path}")
    for key in ("pipeline_temporal", "pipeline_bbox"):
        def has(r):
            return bool((r.get(key) or {}).get("start")) if key == "pipeline_temporal" else bool(r.get(key))
        print(f"info       files with {key}: {sum(has(base_files[p]) for p in shared)} -> {sum(has(cur_files[p]) for p in shared)}")
    for path in lost[:20]:
        print(f"REGRESSION file no longer usable: {path}")
    return 1 if regressions or broken or lost else 0
