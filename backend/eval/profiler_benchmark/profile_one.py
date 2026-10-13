"""Profile one input in a fresh process and print a JSON summary on stdout.

run.py starts one of these per input so that a crash, an out-of-memory kill or a
hang affects only that input, and so peak memory (ru_maxrss) is per input.

Modes (all through the connector's own code):
  pipeline  the file as ingestion samples it (first 500 rows / 2 MB)
  full      the whole file (up to full_max_rows / full_max_bytes)

  python -m eval.profiler_benchmark.profile_one '{"path": "f.csv", "mode": "pipeline"}'
"""

from __future__ import annotations

import json
import logging
import resource
import sys
import time
import warnings
from pathlib import Path


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)  # bytes on macOS, KB on Linux


def _epoch_bounds(coverage: list | None) -> list[dict]:
    out = []
    for item in coverage or []:
        ranges = [r["range"] for r in item.get("ranges") or [] if "range" in r]
        if ranges:
            out.append({"columns": item.get("column_names"), "min": min(r["gte"] for r in ranges),
                        "max": max(r["lte"] for r in ranges), "resolution": item.get("temporal_resolution")})
    return out


def summarize(record: dict) -> dict:
    from eval.profiler_benchmark.common import coarse_from_profiler

    meta = record.get("profiler_metadata") or {}
    return {
        "errors": record.get("errors") or [],
        "nb_rows": meta.get("nb_rows"),
        "nb_columns": meta.get("nb_columns"),
        "profiled_columns": len(meta.get("columns") or []),
        "columns_truncated": bool(meta.get("columns_truncated")),
        "types": meta.get("types"),
        "columns": [{"name": str(c.get("name")), "coarse": coarse_from_profiler(c),
                     "structural": (c.get("structural_type") or "").rsplit("/", 1)[-1],
                     "semantic": [s.rsplit("/", 1)[-1] for s in c.get("semantic_types") or []],
                     "distinct": c.get("num_distinct_values")}
                    for c in meta.get("columns") or []],
        # What the profiler itself found vs what the pipeline put in the record.
        "profiler_temporal": _epoch_bounds(meta.get("temporal_coverage")),
        "profiler_spatial": [{"type": s.get("type"), "columns": s.get("column_names")} for s in meta.get("spatial_coverage") or []],
        "pipeline_temporal": record.get("temporal_coverage"),
        "pipeline_bbox": meta.get("spatial_bbox"),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    spec = json.loads(sys.argv[1])
    from crawlers.invenio.transformer import assemble_record, profile_sample, sample_local_file

    path = Path(spec["path"])
    ext = path.name.rsplit(".", 1)[-1].lower()
    started = time.perf_counter()
    if spec.get("mode") == "full":
        stats = sample_local_file(path, ext, max_rows=spec.get("full_max_rows", 200_000),
                                  max_bytes=spec.get("full_max_bytes", 200 * 1024 * 1024))
    else:
        stats = sample_local_file(path, ext)
    headers, rows, output, errors = profile_sample(stats, ext)
    record = assemble_record({"dataset_id": f"bench:{path.name}", "file_key": path.name},
                             stats, headers, rows, output, errors)
    result = summarize(record)
    result.update({"status": "ok" if not result["errors"] else "profiler_error",
                   "seconds": round(time.perf_counter() - started, 2), "peak_rss_mb": peak_rss_mb(),
                   "sample_bytes": stats.bytes_written,
                   "sample_truncated": stats.truncated_by_rows or stats.truncated_by_bytes})
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
