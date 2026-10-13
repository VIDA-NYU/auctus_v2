"""Profiler / ingestion reliability benchmark.

  python -m eval.profiler_benchmark.run all --harvest ../../data/ultraviolet ../../data/caltech --out results/<name>
  python -m eval.profiler_benchmark.run compare results/baseline results/<name>

Checks (see README.md):
  robustness   every corpus file through the connector, one process each
  types        synthetic tables with known types + README data dictionaries
  sampling     pipeline sample vs whole file, for files larger than the sample
  performance  time and peak memory over a grid of table shapes
Each writes a CSV under --out; `all` also writes summary.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]
# Hand-verified dictionaries (review_gold.py verify) take precedence over raw extractions.
VERIFIED_GOLD_DIR = HERE / "gold" / "ultraviolet"
GOLD_DIR = HERE.parent / "description_eval" / "data" / "gold"
PERF_SHAPES = [(1_000, 10), (1_000, 100), (1_000, 1_000), (10_000, 10), (10_000, 100), (100_000, 10), (100_000, 100)]


def run_child(spec: dict, timeout: int) -> dict:
    """Profile one input in its own process; failures become a status, never an exception."""
    env = {**os.environ, "PYTHONPATH": str(BACKEND)}
    with tempfile.TemporaryDirectory() as cwd:  # atlas-profiler writes output/ files into the cwd
        try:
            proc = subprocess.run([sys.executable, "-m", "eval.profiler_benchmark.profile_one", json.dumps(spec)],
                                  capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
    if proc.returncode != 0:
        status = "killed" if proc.returncode < 0 else "crash"  # negative = signal, e.g. SIGKILL from OOM
        return {"status": status, "returncode": proc.returncode, "stderr": proc.stderr.strip().splitlines()[-1:] }
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_many(specs: list[dict], jobs: int, timeout: int, label: str) -> list[dict]:
    results: list[dict] = [{}] * len(specs)

    def work(i: int) -> None:
        results[i] = run_child(specs[i], timeout)
        done = sum(1 for r in results if r)
        if done % 10 == 0 or done == len(specs):
            print(f"  {label}: {done}/{len(specs)}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(work, range(len(specs))))
    return results


def robustness(corpus: list[dict], jobs: int, timeout: int) -> list[dict]:
    results = run_many([{"path": f["path"], "mode": "pipeline"} for f in corpus], jobs, timeout, "robustness")
    return [{**f, **r} for f, r in zip(corpus, results)]


def types(jobs: int, timeout: int, tmp: Path) -> tuple[list[dict], list[dict]]:
    from eval.profiler_benchmark.common import coarse_from_gold
    from eval.profiler_benchmark.synthetic import write_suite

    cases = write_suite(tmp / "synthetic")
    synthetic = []
    for case, result in zip(cases, run_many([{"path": c["path"]} for c in cases], jobs, timeout, "types/synthetic")):
        found = {c["name"]: c for c in result.get("columns", [])}
        for name, gold in case["gold"].items():
            got = found.get(name, {}).get("coarse")
            synthetic.append({"case": case["case"], "note": case["note"], "column": name, "expected": gold["expected"],
                              "acceptable": "|".join(gold["acceptable"]), "got": got,
                              "correct": got in gold["acceptable"], "status": result.get("status")})

    readme = []
    gold_files = {p.stem: p for p in (sorted(GOLD_DIR.glob("*.json")) if GOLD_DIR.exists() else [])}
    gold_files.update({p.stem: p for p in VERIFIED_GOLD_DIR.glob("*.json")} if VERIFIED_GOLD_DIR.exists() else {})
    for gold_path in [gold_files[k] for k in sorted(gold_files)]:
        gold = json.loads(gold_path.read_text())
        for file_key, columns in (gold.get("dictionary") or {}).items():
            targets = [f["file_key"] for f in gold["files"]] if file_key == "*" else [file_key]
            for target in targets:
                local = BACKEND.parents[1] / "data" / "ultraviolet" / "files" / gold["record_id"] / target
                if not local.exists():
                    continue
                result = run_child({"path": str(local)}, timeout)
                found = {c["name"].strip().lower(): c for c in result.get("columns", [])}
                for col in columns:
                    expected, acceptable = coarse_from_gold(col.get("type", ""), col.get("units", ""),
                                                            col["column"], col.get("definition", ""))
                    got = found.get(str(col["column"]).strip().lower(), {}).get("coarse")
                    scorable = expected is not None and got is not None
                    readme.append({"record": gold["record_id"], "file": target, "column": col["column"],
                                   "gold_type": col.get("type", ""), "expected": expected,
                                   "acceptable": "|".join(sorted(acceptable)), "got": got,
                                   "verified": gold.get("verified", False),
                                   "strict_correct": got == expected if scorable else None,
                                   "correct": got in acceptable if scorable else None})
    return synthetic, readme


def sampling(robust: list[dict], jobs: int, timeout: int, limit: int) -> list[dict]:
    candidates = [r for r in robust if r.get("status") == "ok" and r.get("sample_truncated")][:limit]
    fulls = run_many([{"path": r["path"], "mode": "full"} for r in candidates], jobs, timeout, "sampling")
    rows = []
    for sample, full in zip(candidates, fulls):
        if full.get("status") != "ok":
            rows.append({"path": sample["path"], "status": full.get("status")})
            continue
        s_types = {c["name"]: c["coarse"] for c in sample["columns"]}
        f_types = {c["name"]: c["coarse"] for c in full["columns"]}
        shared = [n for n in s_types if n in f_types]
        s_t, f_t = (sample["profiler_temporal"] or [None])[0], (full["profiler_temporal"] or [None])[0]
        span_recall = None
        if s_t and f_t and f_t["max"] > f_t["min"]:
            span_recall = round((min(s_t["max"], f_t["max"]) - max(s_t["min"], f_t["min"])) / (f_t["max"] - f_t["min"]), 3)
        rows.append({"instance": sample["instance"], "path": sample["path"], "status": "ok",
                     "sample_rows": sample["nb_rows"], "full_rows": full["nb_rows"],
                     "columns_compared": len(shared),
                     "type_changes": sum(s_types[n] != f_types[n] for n in shared),
                     "temporal_span_recall": span_recall})
    return rows


def performance(jobs: int, timeout: int, tmp: Path) -> list[dict]:
    from eval.profiler_benchmark.synthetic import performance_frame

    specs = []
    for rows, cols in PERF_SHAPES:
        path = tmp / f"perf_{rows}x{cols}.csv"
        performance_frame(rows, cols).to_csv(path, index=False)
        specs.append({"path": str(path), "mode": "full", "full_max_rows": rows + 1, "full_max_bytes": 10**10})
    # One at a time: concurrent runs would distort each other's timings.
    results = run_many(specs, 1, timeout, "performance")
    return [{"rows": r, "cols": c, "status": res.get("status"), "seconds": res.get("seconds"),
             "peak_rss_mb": res.get("peak_rss_mb"), "profiled_columns": res.get("profiled_columns")}
            for (r, c), res in zip(PERF_SHAPES, results)]


def main() -> None:
    from eval.profiler_benchmark import score
    from eval.profiler_benchmark.common import load_corpus

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["all", "robustness", "types", "sampling", "performance", "compare"])
    parser.add_argument("paths", nargs="*", help="compare: BASELINE_DIR CURRENT_DIR")
    parser.add_argument("--harvest", nargs="+", type=Path, default=[], help="invenio_fetching harvest dirs")
    parser.add_argument("--per-instance", type=int, default=0, help="random files per instance (0 = all)")
    parser.add_argument("--sampling-limit", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "latest")
    args = parser.parse_args()

    if args.stage == "compare":
        sys.exit(score.compare(Path(args.paths[0]), Path(args.paths[1])))

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.harvest)
    if args.per_instance:
        rng = random.Random(args.seed)
        by_instance: dict[str, list] = {}
        for f in corpus:
            by_instance.setdefault(f["instance"], []).append(f)
        corpus = [f for files in by_instance.values() for f in rng.sample(files, min(args.per_instance, len(files)))]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        robust = []
        if args.stage in ("all", "robustness", "sampling"):
            robust = robustness(corpus, args.jobs, args.timeout)
            # Per-column detail is only needed in memory (sampling check); keep the saved file small.
            score.write_rows(args.out / "robustness.jsonl", [{k: v for k, v in r.items() if k != "columns"} for r in robust])
        if args.stage in ("all", "types"):
            synthetic, readme = types(args.jobs, args.timeout, tmp)
            score.write_rows(args.out / "types_synthetic.jsonl", synthetic)
            score.write_rows(args.out / "types_readme.jsonl", readme)
        if args.stage in ("all", "sampling"):
            score.write_rows(args.out / "sampling.jsonl", sampling(robust, args.jobs, args.timeout, args.sampling_limit))
        if args.stage in ("all", "performance"):
            score.write_rows(args.out / "performance.jsonl", performance(args.jobs, args.timeout, tmp))
    score.summarize_dir(args.out, {"harvest": [str(h) for h in args.harvest], "per_instance": args.per_instance,
                                   "seed": args.seed, "stage": args.stage})


if __name__ == "__main__":
    main()
