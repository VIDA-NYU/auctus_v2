"""LLM judge -> provisional BINARY qrels, under the anti-leakage code invariant.

For every pooled (query, dataset) pair the judge assigns a binary relevance
label 0/1 (aligned with the AutoDDG paper's evaluation protocol). The judge sees
ONLY the neutral bundle (title + profile + data sample) — the description arm
under test is PHYSICALLY ABSENT from the prompt (report F1b). If the judge saw
the arm, NDCG would systematically favour the description arms and the whole
evaluation would be void. The judge is held to the same F1 enforcement as the
query generator, by construction rather than by repetition: it fetches with the
shared ``NEUTRAL_SOURCE_FIELDS`` allowlist and builds its prompt through the same
``build_neutral_bundle``, which refuses any document carrying an arm field.

Label: 1 = relevant (the dataset answers the query, fully or partially);
       0 = not relevant.

Datasets outside the pool are non-relevant by convention (not judged, grade 0).
These qrels are PROVISIONAL (LLM-only, un-calibrated on NYC) — a pipeline
shakedown, not benchmark ground truth.

    python -m eval.judge_qrels --pool eval/benchmark/pool.json \
        --out eval/benchmark/qrels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client
from eval.backfill_descriptions import load_full_profile
from eval.generate_queries import NEUTRAL_SOURCE_FIELDS, build_neutral_bundle
from eval.provenance import code_version
from eval.llm_client import (
    LLM_MODEL, MODEL_LAB, complete_verbose, get_llm_client, supports_temperature,
    temperature_pinned,
)
from storage.minio_client import get_storage_client

LOGGER = logging.getLogger("judge_qrels")

JUDGE_PROMPT = """You are a relevance assessor for a dataset-search benchmark. \
Given a user QUERY and a numbered list of candidate datasets (described ONLY by \
neutral facts: title, an algorithmic column/coverage profile, and a data sample), \
label whether each dataset is relevant to the query.

Labels:
- 1 = relevant: the dataset answers the query, fully or partially (topically \
related, or its columns/coverage contain what the query asks for).
- 0 = not relevant.

Judge on the data itself (columns, coverage, sample) — a dataset whose columns \
contain what the query asks for is relevant even if the wording differs. But a \
shared column name or field type is not enough on its own: check that the \
candidate's row-level entity actually matches the query's subject. (Example: a \
driver-license roster and a vehicle-license roster can share columns like \
"License Number, Expiration Date" while licensing two different things — that \
shared shape does not make one relevant to a query about the other.)

When a query names a time period, a candidate is relevant on temporal grounds \
only if its subject matches what the query asks for AND its temporal coverage \
includes that period — temporal coverage alone, without a matching subject, is \
not sufficient. "Temporal coverage" means an explicit computed date range \
reported in the profile (a temporal_coverage field) — not the mere presence \
of a date-type column (e.g. an ARREST_DATE field), and not words in the title \
such as "historic", "archive", "current", or a specific year. Column names \
and title wording are NOT coverage evidence — do not use them to guess what \
period a candidate covers. If a candidate's profile reports no \
temporal_coverage range at all, that is missing information, not a failed \
match: do not reject the candidate for an unconfirmed period. Instead, judge \
it on subject match alone, exactly as you would for a query that named no \
time period at all.

When a query names a place, a candidate is relevant on spatial grounds only \
if its subject matches what the query asks for AND its spatial coverage \
includes that place — spatial coverage alone, without a matching subject, is \
not sufficient. A dataset whose records are situated in the queried place but \
whose subject is something else is NOT relevant. Latitude/longitude/address \
columns cannot substitute for a subject match.

Return STRICT JSON mapping each dataset number to its label, e.g. \
{{"1": 1, "2": 0, "3": 1}}. No prose outside the JSON.

QUERY: {query}

DATASETS:
{datasets}
"""


def _render_bundle(idx: int, bundle: dict) -> str:
    return (f"[{idx}] Title: {bundle['title']}\n"
            f"    Profile: {bundle['profile']}\n"
            f"    Sample: {bundle['sample'][:600]}")


def parse_label_json(raw: str) -> dict:
    """Recover the label object from a model response.

    Panel models differ in output convention: Claude wraps JSON in ``` fences,
    others may preface it with prose. Raising on unparseable output is
    deliberate — the caller records the failure per query rather than letting a
    whole batch silently collapse to "nothing relevant".
    """
    text = raw.strip()
    fenced = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span, for responses wrapped in prose.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    return json.loads(text[start:end + 1])


def judge_query(client, query_text: str, items: list[tuple[str, dict, str | None]],
                model: str, seed: int | None = None,
                no_cache: bool = False) -> tuple[dict, dict]:
    """Grade every (query, dataset) in ``items``. Returns ({dataset_id: grade}, usage)."""
    bundles = []
    for _id, doc, sample in items:
        # F1b invariant: build_neutral_bundle refuses a doc carrying any arm field.
        bundle = build_neutral_bundle(doc, sample)
        bundles.append(bundle)
    rendered = "\n\n".join(_render_bundle(i + 1, b) for i, b in enumerate(bundles))
    prompt = JUDGE_PROMPT.format(query=query_text, datasets=rendered)

    raw, usage = complete_verbose(client, prompt, temperature=0.0, model=model,
                                  seed=seed, no_cache=no_cache)
    grades = parse_label_json(raw)

    out: dict[str, int] = {}
    for i, (_id, _doc, _s) in enumerate(items, 1):
        g = grades.get(str(i), grades.get(i, 0))
        try:
            g = int(g)
        except (TypeError, ValueError):
            g = 0
        out[_id] = max(0, min(1, g))
    return out, usage


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="eval/benchmark/pool.json")
    parser.add_argument("--out", default="eval/benchmark/qrels.json")
    parser.add_argument("--model", default=LLM_MODEL,
                        help="judge model string on the Portkey gateway")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed, when the provider honours it")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="judge only the first N queries (pilot runs)")
    parser.add_argument("--query-ids", default=None,
                        help="comma-separated query_ids to judge (pilot runs)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the gateway response cache; required for "
                             "self-consistency measurement, which would otherwise "
                             "measure the cache rather than the model")
    args = parser.parse_args(argv)

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    os_client = get_client()
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM client (PORTKEY_API_KEY set? on NYU VPN?).")
    try:
        storage_client = get_storage_client()
    except Exception:
        storage_client = None

    # Cache doc + sample per dataset id (a dataset appears in many pools).
    doc_cache: dict[str, dict] = {}
    sample_cache: dict[str, str | None] = {}

    def load(dataset_id: str):
        if dataset_id not in doc_cache:
            doc_cache[dataset_id] = os_client.get(
                index=AUCTUS_INDEX_NAME, id=dataset_id,
                _source=list(NEUTRAL_SOURCE_FIELDS),
            ).get("_source") or {}
            rec = load_full_profile(storage_client, dataset_id) if storage_client else None
            sample_cache[dataset_id] = rec.get("sample") if isinstance(rec, dict) else None
        return doc_cache[dataset_id], sample_cache[dataset_id]

    queries = pool["queries"]
    if args.query_ids:
        wanted = {qid.strip() for qid in args.query_ids.split(",") if qid.strip()}
        queries = [q for q in queries if q["query_id"] in wanted]
    if args.limit_queries:
        queries = queries[:args.limit_queries]

    out_queries = []
    total_pairs = 0
    failures: list[dict] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    started = time.monotonic()
    for i, q in enumerate(queries, 1):
        items = []
        for dataset_id in q["pool"]:
            doc, sample = load(dataset_id)
            items.append((dataset_id, doc, sample))
        failed = None
        try:
            grades, usage = judge_query(client, q["text"], items,
                                        model=args.model, seed=args.seed,
                                        no_cache=args.no_cache)
            for key in usage_totals:
                usage_totals[key] += usage.get(key) or 0
        except Exception as exc:
            # A failed batch is NOT the same as "nothing is relevant" — record it
            # so a parse/API failure can never masquerade as a judgment of 0.
            LOGGER.warning("Judge failed for %s: %s", q["query_id"], exc)
            failed = f"{type(exc).__name__}: {exc}"
            failures.append({"query_id": q["query_id"], "error": failed,
                             "pool_size": len(items)})
            grades = {}
        # Keep only positives in the qrels file (retrieval_eval treats missing as 0).
        relevant = {k: v for k, v in grades.items() if v > 0}
        total_pairs += len(items)
        entry = {
            "query_id": q["query_id"],
            "text": q["text"],
            "relevant": relevant,
        }
        if failed:
            entry["judge_failed"] = failed
        out_queries.append(entry)
        print(f"  [{i}/{len(queries)}] {q['query_id']}: "
              f"{'FAILED' if failed else f'{len(relevant)}/{len(items)} relevant'}")

    pinned = temperature_pinned(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "code_version": code_version(),
        "_provenance": "PROVISIONAL LLM-judge BINARY qrels (un-calibrated on NYC); "
                       "anti-leakage guard: judge saw title+profile+sample only, "
                       "never any description arm.",
        "judge": {
            "model": args.model,
            "lab": MODEL_LAB.get(args.model, "unknown"),
            "temperature": 0.0 if supports_temperature(args.model) else "provider default",
            "seed": args.seed,
            # INTENT ONLY: whether temperature 0 was actually set. This is NOT
            # a stability claim — gpt-5-mini cannot pin temperature yet returned
            # identical labels across fresh runs. Stability evidence is the
            # measured self-consistency below, which carries its own scope.
            "temperature_pinned": pinned,
            "cache_bypassed": args.no_cache,
            "self_consistency": {
                "_note": "Cited from a separate measurement, NOT measured on "
                         "this run. Repeat runs over the pilot subset with the "
                         "gateway cache bypassed.",
                "kappa": 1.0,
                "scope": "10 queries x 2 runs (254 pooled pairs)",
                "source": "eval/benchmark/selfcon_*.json",
            },
        },
        "run": {
            "pool": args.pool,
            "queries_judged": len(out_queries),
            "pairs_judged": total_pairs,
            "failed_queries": len(failures),
            "failures": failures,
            "usage": usage_totals,
            "wall_clock_s": round(time.monotonic() - started, 1),
        },
        "queries": out_queries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJudged {total_pairs} pooled pairs across {len(out_queries)} queries -> {out}")
    print(f"  judge={args.model} temperature_pinned={pinned} "
          f"failures={len(failures)} usage={usage_totals}")
    if failures:
        print(f"  WARNING: {len(failures)} queries failed and hold NO judgments "
              f"(not 'no relevant datasets') — exclude or re-run before scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
