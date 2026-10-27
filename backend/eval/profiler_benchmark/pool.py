"""Persistent profiling workers for large benchmark runs.

run_child starts a fresh Python per file, which isolates crashes and memory but pays
the profiler's import cost (torch, transformers, the geo-classifier model: ~15 s on a
loaded machine) every time. Here each worker imports once and profiles many files;
workers are recycled every MAX_TASKS_PER_WORKER files to bound memory growth, and a
worker that dies (OOM, segfault) only loses its current file.

Used by run.py with --persistent-workers. Results have the same shape as run_child's.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

MAX_TASKS_PER_WORKER = 40
BACKEND = Path(__file__).resolve().parents[2]


def _init() -> None:
    sys.path.insert(0, str(BACKEND))
    os.chdir(tempfile.mkdtemp())  # atlas-profiler writes output/ files into the cwd
    import warnings, logging

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import crawlers.invenio.transformer  # noqa: F401  (the expensive import, once per worker)


def _profile(spec: dict) -> dict:
    import io
    import contextlib
    import json

    from eval.profiler_benchmark import profile_one

    out = io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["profile_one", json.dumps(spec)]
        with contextlib.redirect_stdout(out):
            profile_one.main()
    except Exception as exc:
        return {"status": "crash", "stderr": [repr(exc)]}
    finally:
        sys.argv = argv
    return json.loads(out.getvalue().strip().splitlines()[-1])


def run_many_pooled(specs: list[dict], jobs: int, timeout: int, label: str) -> list[dict]:
    ctx = mp.get_context("spawn")
    results: list[dict] = [{}] * len(specs)
    with ctx.Pool(processes=jobs, initializer=_init, maxtasksperchild=MAX_TASKS_PER_WORKER) as pool:
        pending = [(i, pool.apply_async(_profile, (spec,))) for i, spec in enumerate(specs)]
        for done, (i, handle) in enumerate(pending, 1):
            try:
                results[i] = handle.get(timeout=timeout)
            except (mp.TimeoutError, FutureTimeout):
                results[i] = {"status": "timeout"}
            except Exception as exc:  # worker died
                results[i] = {"status": "killed", "stderr": [repr(exc)]}
            if done % 10 == 0 or done == len(specs):
                print(f"  {label}: {done}/{len(specs)}", file=sys.stderr, flush=True)
    return results
