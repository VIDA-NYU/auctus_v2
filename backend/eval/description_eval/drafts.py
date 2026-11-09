"""Curator-draft mode: a record-level description draft for the landing page.

Curators describe records, not files, so this drafts one description per record from
  docs  the record's README / codebook text + a profile of each of its tables
  data  the profiles only (what Auctus can do without documentation)
and grades both against the curator's own description with semtypes.FACTS_JUDGE
(curator facts split into content / provenance; claims checked against the docs and
the data). The README is an input here on purpose: this is the drafting tool for
records that have one, not the ingestion path.

  python -m eval.description_eval.drafts
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval.arms import documentation_text, paired
from eval.description_eval.collect import load_profile
from eval.description_eval.common import DATA_DIR, GOLD_DIR, INSTANCES, ask_json, read_json, write_json
from eval.description_eval.semtypes import data_evidence, judge_facts, mean

ARMS = ("docs", "data")
OUT_DIR = DATA_DIR / "drafts"
DRAFT_MODEL = "@vertexai/gemini-2.5-flash"
MAX_TABLES = 6

PROMPT = """Draft the description for a dataset record in NYU's research data repository, for a curator to edit.

Record title: {title}

{documentation}Tables in the record (profiled over every row):
{tables}

Write 120 to 200 words: what the data is and covers (variables, place, time span, size), who
produced it and why, and how it was collected, as far as the inputs say. State only what the
inputs support; leave out anything they don't say rather than guessing. Plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""


def table_summary(file_key: str, profile: dict[str, Any]) -> str:
    coverage = profile.get("temporal_coverage") or {}
    spatial = (profile.get("profiler_metadata") or {}).get("spatial_bbox")
    extra = []
    if coverage.get("start"):
        extra.append(f"time {coverage['start']} to {coverage.get('end')}")
    if spatial:
        extra.append(f"bbox {[round(v, 2) for v in spatial]}")
    return f"### {file_key}{' (' + '; '.join(extra) + ')' if extra else ''}\n{data_evidence(profile, rows=3)[:2500]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default="ultraviolet", choices=sorted(INSTANCES))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    tasks = []
    for gold_path in sorted(GOLD_DIR.glob("**/*.json")):  # caltech gold is in gold/caltech/
        gold = read_json(gold_path)
        if gold.get("domain", INSTANCES["ultraviolet"]) != INSTANCES[args.instance]:
            continue
        curator = gold.get("curator") or {}
        if not curator.get("description"):
            continue
        profiles = []
        for f in sorted(gold["files"], key=lambda f: f["file_key"])[:MAX_TABLES]:
            try:
                profiles.append((f["file_key"], load_profile(storage, f["dataset_id"])))
            except Exception:
                continue
        if not profiles:
            continue
        docs = documentation_text(gold["record_id"], args.instance)
        tables = "\n\n".join(table_summary(k, p) for k, p in profiles)
        evidence = "\n\n".join(x for x in (docs and f"Documentation:\n{docs}", tables) if x)
        for arm in ARMS:
            if arm == "docs" and not docs:
                continue
            documentation = f"Documentation (README / codebook):\n{docs}\n\n" if arm == "docs" else ""
            tasks.append((gold, curator, arm, PROMPT.format(title=curator.get("title", ""), documentation=documentation,
                                                           tables=tables[:12000]), evidence))
    print(f"{len(tasks)} drafts over {len({t[0]['record_id'] for t in tasks})} records")

    def run(task) -> dict[str, Any]:
        gold, curator, arm, prompt, evidence = task
        try:
            draft = ask_json(prompt, model=DRAFT_MODEL).get("description", "")
            graded = judge_facts(curator["description"], evidence, "the whole record", draft)
        except Exception as exc:
            return {"record": gold["record_id"], "dataset_id": gold["record_id"], "arm": arm, "error": repr(exc),
                    "fact_recall": None, "content_recall": None, "provenance_recall": None, "unsupported_rate": None}
        return {"record": gold["record_id"], "dataset_id": gold["record_id"], "arm": arm,
                "words": len(draft.split()), "draft": draft, **graded}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))

    summary: dict[str, Any] = {"model": DRAFT_MODEL, "arms": {}}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and not r.get("error")]
        summary["arms"][arm] = {
            "records": len(mine), "errors": sum(1 for r in rows if r["arm"] == arm and r.get("error")),
            **{m: mean([r[m] for r in mine]) for m in ("fact_recall", "content_recall", "provenance_recall",
                                                        "unsupported_rate")},
            "median_words": statistics.median(r["words"] for r in mine) if mine else None,
        }
    # docs vs data on the records that have documentation ("base" is what arms.paired compares against)
    rebased = [{**r, "arm": "base"} if r["arm"] == "data" else r for r in rows]
    summary["docs_vs_data"] = {m: paired(rebased, "docs", m) for m in ("fact_recall", "content_recall",
                                                                         "provenance_recall", "unsupported_rate")}
    write_json(OUT_DIR / ("results.json" if args.instance == "ultraviolet" else f"results-{args.instance}.json"), {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
