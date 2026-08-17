"""LLM judge -> provisional GRADED (0/1/2) qrels, under the anti-leakage code
invariant.

For every pooled (query, dataset) pair the judge assigns a graded relevance
label: 0 = subject does not match; 1 = subject matches but a stated constraint
fails; 2 = subject matches and every stated constraint holds. See ``JUDGE_PROMPT``
for the full rule (one constraint-mismatch rule, four constraint types). A
binary judge cannot express the 1: "subject matches, one constraint fails" is
neither "relevant" nor "not relevant". The judge sees ONLY the neutral bundle
(title + profile + data sample) — the description arm under test is PHYSICALLY
ABSENT from the prompt (report F1b). If the judge saw the arm, NDCG would
systematically favour the description arms and the whole evaluation would be
void. The judge is held to the same F1 enforcement as the query generator, by
construction rather than by repetition: the live-index path fetches with the
shared ``NEUTRAL_SOURCE_FIELDS`` allowlist and builds its prompt through the
same ``build_neutral_bundle``, which refuses any document carrying an arm
field; the ``--bundles`` offline path (which never touches a raw document at
all) asserts the equivalent key-set restriction at load time instead.

The judge's response carries a second signal beside the grade: ``unverified``,
the candidates whose grade relied on a stated constraint the profile carried
no field to check (e.g. no spatial coverage at all, post-2026-08-08). A
missing field does NOT lower the grade — see ``JUDGE_PROMPT`` — so this flag is
what keeps "constraint genuinely satisfied" and "constraint could not be
checked" distinguishable in the qrels artifact.

Datasets outside the pool are non-relevant by convention (not judged, grade 0).
A pooled dataset absent from a *judged* query's ``relevant`` map was graded 0,
not left unjudged — a query that failed entirely carries ``judge_failed``
instead, never a silently empty grade map. These qrels are PROVISIONAL
(LLM-only, un-calibrated on NYC) — a pipeline shakedown, not benchmark ground
truth.

    python -m eval.judge_qrels --pool eval/benchmark/pool.json \
        --out eval/benchmark/qrels.json

    # Offline, from a pinned capture (no OpenSearch/MinIO call):
    python -m eval.judge_qrels --pool eval/benchmark/pool_n32.json \
        --bundles ../../../useful/2026-07-22/data/bundles_n32.json \
        --out eval/benchmark/qrels_n32.json
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

# The 17 approved worked examples (judge-fewshot-examples.md, approved
# 2026-08-16). Copied verbatim per that file's own instruction: "The judge
# never sees the facet column... only the query / candidate / grade / why
# rows, and no facet name appears in any of those fields" — so the six
# facet-group headers from the source file are NOT reproduced here, only the
# flattened 17 rows in the source's order.
_JUDGE_WORKED_EXAMPLES = """\
| grade | query | candidate | why |
| --- | --- | --- | --- |
| 0 | restaurant inspections | a dataset of restaurant permits issued city-wide | The query asks for individual restaurant inspection records, but the dataset provides individual restaurant permit records; therefore, no constraints are evaluated. |
| 2 | noise complaints | a dataset of all municipal service requests, with a complaint-type column | The query asks for individual noise complaint records, and the dataset provides individual service request records, which are the same kind of record; the query names a category of complaint but states no temporal, spatial, record grain or value span constraint, and a dataset being broader than the query's category is not a constraint it can fail. |
| 0 | bus delays 2021 | a dataset of bus route schedules for 2021 | The query asks for individual bus delay records, but the dataset provides individual bus route schedule records; one row in the dataset describes a planned departure, not an observed deviation from it, so no constraints are evaluated. |
| 1 | taxi trips after 2020 | a dataset of individual taxi trips with coverage ending in 2019 | The query asks for individual taxi trip records, and the dataset provides individual taxi trip records; however, the query's temporal constraint for records after 2020 fails because the dataset's coverage ends in 2019. |
| 2 | taxi trips in 2018 | a dataset of individual taxi trips from 2015 to 2020 | The query asks for individual taxi trip records, and the dataset provides individual taxi trip records; the query's temporal constraint for 2018 is met as the dataset covers 2015 to 2020. |
| 0 | tree planting locations Brooklyn | a dataset of park maintenance schedules for Brooklyn | The query asks for individual tree planting location records, but the dataset provides individual park maintenance schedule records; therefore, no constraints are evaluated. |
| 1 | air quality in Staten Island | a dataset of individual air quality measurements covering only Manhattan | The query asks for individual air quality measurements, and the dataset provides individual air quality measurements; however, the query's spatial constraint for Staten Island fails because the dataset only covers Manhattan. |
| 2 | air quality in Queens | a dataset of individual air quality measurements city-wide | The query asks for individual air quality measurements, and the dataset provides individual air quality measurements; the query's spatial constraint for Queens is met as the dataset covers city-wide. |
| 0 | collision-level records | a dataset of traffic camera images showing collisions | The query asks for individual collision records, but the dataset provides individual traffic camera image records; therefore, no constraints are evaluated. |
| 1 | each row is a single collision | a dataset of yearly collision totals by borough | The query asks for individual collision records, and the dataset provides collision records; however, the query's record grain constraint for single collisions fails because the dataset provides yearly collision totals. |
| 2 | individual complaint records | a dataset where each row details a single service request made to the city | The query asks for individual complaint records, and the dataset provides records where each row details a single service request, which are the same kind of record despite different wording; the stated constraint on record grain is met. |
| 0 | school performance ratings 1-5 | a dataset of school attendance records with a 'performance' column indicating attendance rate | The query asks for individual school performance rating records, but the dataset provides individual school attendance records; therefore, no constraints are evaluated. |
| 1 | housing quality scores on a 0-100 scale | a dataset of individual housing quality scores where the scale runs 1-5 | The query asks for individual housing quality score records, and the dataset provides individual housing quality score records; however, the query's value span constraint for a 0-100 scale fails because the dataset's scores are on a 1-5 scale. |
| 2 | housing quality scores on a 0-100 scale | a dataset of individual housing quality scores on a 0-100 scale | The query asks for individual housing quality score records, and the dataset provides individual housing quality score records; the query's value span constraint for a 0-100 scale is met. |
| 0 | building code violations in Queens 2020 | a dataset of new construction permits in Queens for 2020 | The query asks for individual building code violation records, but the dataset provides individual new construction permit records; therefore, no constraints are evaluated. |
| 1 | parking violations in Brooklyn 2022 | a dataset of individual parking violations city-wide with coverage ending in 2021 | The query asks for individual parking violation records, and the dataset provides individual parking violation records; however, the query's temporal constraint for 2022 fails because the dataset's coverage ends in 2021. |
| 2 | parking violations in Queens 2021 | a dataset of individual parking violation records city-wide from 2019 to 2023 | The query asks for individual parking violation records, and the dataset provides individual parking violation records; the query's spatial constraint for Queens and temporal constraint for 2021 are both met. |
"""

# The constraint rule, copied verbatim from next-round-design-decisions.md
# §4c ("The rule, written as it should reach the prompt"). Approved payload —
# not re-derived here.
#
# Built by plain concatenation (head + examples + tail), NOT by chaining two
# .format() calls on one template: the JSON example below (`{"1": 2, ...}`)
# and the runtime `{query}`/`{datasets}` placeholders both use braces, and a
# second .format() pass would try to re-parse the JSON braces as fields once
# the first pass had already collapsed their escaping. One template, one
# runtime .format() call (in judge_query), avoids the ambiguity entirely.
_JUDGE_PROMPT_HEAD = """You are a relevance assessor for a dataset-search benchmark. \
Given a user QUERY and a numbered list of candidate datasets (described ONLY by \
neutral facts: title, an algorithmic column/coverage profile, and a data sample), \
grade each dataset's relevance to the query on a 0/1/2 scale.

Decide first whether the subject matches. The subject is the kind of record \
the query asks for — what one row would be — not the domain it sits in: a \
query for drivers is not answered by a roster of vehicles, however much both \
are about medallions.

- Subject does not match -> 0. Shared words, matching columns and overlapping \
coverage do not change this.
- Subject matches -> check every constraint the query states about those \
records. All hold -> 2. Any one fails -> 1.

| constraint | a failure looks like |
| --- | --- |
| time | query says "after 2020", the dataset's coverage ends 2016 |
| place | query says "in Staten Island", the dataset covers Brooklyn only |
| record grain | query says "each row is a single collision", the dataset holds yearly totals. It still records collisions, so the subject matches and only the grain fails — a 1, not a 0 |
| value span | query says "scores on a 0-100 scale", the column runs 1-5 |

A stated constraint the candidate's profile carries no field to check does \
NOT lower the grade: grade 1 requires a constraint to fail, and failing \
requires evidence you can point to in the profile. Absence of evidence is not \
evidence of failure. This does not soften the subject test above — the \
subject must still match before any constraint is checked.

Worked examples:

"""

_JUDGE_PROMPT_TAIL = """\
Return STRICT JSON with exactly two top-level keys. "grades" maps each \
dataset number to its 0/1/2 grade, e.g. {{"1": 2, "2": 0, "3": 1}}. \
"unverified" lists the dataset numbers (as strings) graded 1 or 2 where the \
grade relied on a stated constraint the profile carried no field to check — \
an empty list, e.g. [], when nothing applies. Always include both keys, even \
when "unverified" is empty. No prose outside the JSON.

QUERY: {query}

DATASETS:
{datasets}
"""

JUDGE_PROMPT = _JUDGE_PROMPT_HEAD + _JUDGE_WORKED_EXAMPLES + _JUDGE_PROMPT_TAIL


def _render_bundle(idx: int, bundle: dict) -> str:
    return (f"[{idx}] Title: {bundle['title']}\n"
            f"    Profile: {bundle['profile']}\n"
            f"    Sample: {bundle['sample'][:600]}")


def _extract_json_object(raw: str) -> dict:
    """Recover a JSON object from a model response, tolerant of fencing/prose.

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
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span, for responses wrapped in prose.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in response: {raw[:200]!r}")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"response JSON is not an object: {raw[:200]!r}")
    return data


def parse_label_json(raw: str) -> tuple[dict, list]:
    """Recover ``(grades, unverified)`` from a model response.

    The response must be the two-key ``{"grades": {...}, "unverified": [...]}``
    shape JUDGE_PROMPT asks for. A response missing either key is an ERROR, not
    an implicit empty/absent value — a model that dropped ``unverified`` has
    not answered the question the prompt asked, and silently defaulting it to
    ``[]`` would make that indistinguishable from "checked everything, nothing
    was unverifiable".
    """
    data = _extract_json_object(raw)
    if "grades" not in data:
        raise ValueError(f"response missing 'grades' key: {raw[:200]!r}")
    if "unverified" not in data:
        raise ValueError(f"response missing 'unverified' key: {raw[:200]!r}")
    return data["grades"], data["unverified"]


def judge_query(client, query_text: str, items: list[tuple[str, dict]],
                model: str, seed: int | None = None,
                no_cache: bool = False) -> tuple[dict, set, dict]:
    """Grade every (query, dataset) in ``items``.

    ``items`` holds already-built neutral bundles (``{title, profile, sample}``)
    — not raw documents. Building the bundle (and enforcing F1) is the loader's
    job now, not this function's: the live-index loader builds one via
    ``build_neutral_bundle``, the ``--bundles`` loader validates one read
    straight from the capture file. This function only renders and grades.

    Returns ``({dataset_id: grade}, {dataset_id, ...} unverified, usage)``.
    """
    rendered = "\n\n".join(_render_bundle(i + 1, bundle) for i, (_id, bundle) in enumerate(items))
    prompt = JUDGE_PROMPT.format(query=query_text, datasets=rendered)

    raw, usage = complete_verbose(client, prompt, temperature=0.0, model=model,
                                  seed=seed, no_cache=no_cache)
    grades_raw, unverified_raw = parse_label_json(raw)
    if not isinstance(grades_raw, dict):
        raise ValueError(f"'grades' is not an object: {grades_raw!r}")
    if not isinstance(unverified_raw, list):
        raise ValueError(f"'unverified' is not a list: {unverified_raw!r}")

    out: dict[str, int] = {}
    for i, (_id, _bundle) in enumerate(items, 1):
        g = grades_raw.get(str(i), grades_raw.get(i, 0))
        try:
            g = int(g)
        except (TypeError, ValueError):
            g = 0
        out[_id] = max(0, min(2, g))

    # Map the model's 1-based candidate numbers back to dataset ids. Recorded
    # as-is, including a candidate the model flagged unverified but graded 0 —
    # the prompt says that combination shouldn't happen, and silently
    # dropping it would hide a rubric violation instead of surfacing one.
    unverified: set[str] = set()
    for n in unverified_raw:
        try:
            idx = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(items):
            unverified.add(items[idx - 1][0])
    return out, unverified, usage


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
    parser.add_argument(
        "--bundles", default=None,
        help="path to a pre-captured {id, title, profile, sample} bundle file "
             "(the 'bundles' list, e.g. bundles_n32.json). When set, issues no "
             "OpenSearch or MinIO call at all -- reads bundles from this file "
             "instead. Use for a pinned-capture re-pilot when the live index "
             "has drifted past what the capture recorded.")
    args = parser.parse_args(argv)

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    client = get_llm_client()
    if client is None:
        raise SystemExit("No LLM client (PORTKEY_API_KEY set? on NYU VPN?).")

    # Two loader paths, one contract: load(dataset_id) -> {title, profile,
    # sample}. The live path builds that bundle via build_neutral_bundle,
    # which enforces F1b (refuses a doc carrying an arm field) on the raw
    # OpenSearch document. The --bundles path never sees a raw document at
    # all, so it can't run that same assertion -- bundles_by_id() below
    # asserts the file-shaped equivalent instead: every record's key set is a
    # subset of {id, title, profile, sample}, so an arm field smuggled into
    # the capture file would be refused here rather than silently reaching
    # the judge.
    bundle_cache: dict[str, dict] = {}
    if args.bundles:
        bundles_data = json.loads(Path(args.bundles).read_text(encoding="utf-8"))
        allowed_keys = {"id", "title", "profile", "sample"}
        bundles_by_id: dict[str, dict] = {}
        for rec in bundles_data.get("bundles", []):
            extra = set(rec.keys()) - allowed_keys
            if extra:
                raise AssertionError(
                    f"bundle record {rec.get('id', '?')!r} in {args.bundles} "
                    f"carries unexpected key(s) {sorted(extra)} -- this is the "
                    f"file-source equivalent of build_neutral_bundle's F1 "
                    f"assertion (which this path bypasses), refusing rather "
                    f"than risk an arm field reaching the judge."
                )
            bundles_by_id[rec["id"]] = {
                "title": rec.get("title") or "",
                "profile": rec.get("profile") or "",
                "sample": rec.get("sample") or "",
            }

        def load(dataset_id: str) -> dict:
            if dataset_id not in bundle_cache:
                if dataset_id not in bundles_by_id:
                    raise KeyError(
                        f"dataset {dataset_id!r} not present in {args.bundles} "
                        f"-- the pool references an id the pinned capture "
                        f"doesn't carry; never substitute or silently skip"
                    )
                bundle_cache[dataset_id] = bundles_by_id[dataset_id]
            return bundle_cache[dataset_id]
    else:
        os_client = get_client()
        try:
            storage_client = get_storage_client()
        except Exception:
            storage_client = None

        def load(dataset_id: str) -> dict:
            if dataset_id not in bundle_cache:
                doc = os_client.get(
                    index=AUCTUS_INDEX_NAME, id=dataset_id,
                    _source=list(NEUTRAL_SOURCE_FIELDS),
                ).get("_source") or {}
                rec = load_full_profile(storage_client, dataset_id) if storage_client else None
                sample = rec.get("sample") if isinstance(rec, dict) else None
                # F1b invariant: refuses a doc carrying any arm field.
                bundle_cache[dataset_id] = build_neutral_bundle(doc, sample)
            return bundle_cache[dataset_id]

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
        items = [(dataset_id, load(dataset_id)) for dataset_id in q["pool"]]
        failed = None
        unverified_ids: set[str] = set()
        try:
            grades, unverified_ids, usage = judge_query(
                client, q["text"], items,
                model=args.model, seed=args.seed, no_cache=args.no_cache,
            )
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
        # Keep only positives in the qrels file. Absence means judged-0, per
        # the file's own provenance statement below -- NOT "left unjudged"
        # (that's what judge_failed on the whole entry is for).
        relevant = {k: v for k, v in grades.items() if v > 0}
        total_pairs += len(items)
        entry = {
            "query_id": q["query_id"],
            "text": q["text"],
            "relevant": relevant,
            "unverified": sorted(unverified_ids),
        }
        if failed:
            entry["judge_failed"] = failed
        out_queries.append(entry)
        print(f"  [{i}/{len(queries)}] {q['query_id']}: "
              f"{'FAILED' if failed else f'{len(relevant)}/{len(items)} relevant'}"
              f"{f', {len(unverified_ids)} unverified' if unverified_ids else ''}")

    pinned = temperature_pinned(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "code_version": code_version(),
        "_provenance": "PROVISIONAL LLM-judge GRADED (0/1/2) qrels (un-calibrated "
                       "on NYC); anti-leakage guard: judge saw title+profile+sample "
                       "only, never any description arm. A pooled dataset absent "
                       "from a judged query's 'relevant' map was judged 0, not "
                       "left unjudged -- a query that failed entirely carries "
                       "'judge_failed' instead, never a silently empty grade map."
                       + (f" Bundles read from a pinned offline capture ({args.bundles}), "
                          "not the live index." if args.bundles else ""),
        "grade_scale": "0/1/2",
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
