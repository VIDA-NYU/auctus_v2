"""Known-item search benchmark: can someone who knows what a record is about find it?

For every indexed record of an instance, an LLM writes queries a researcher might
type for it, from the curator's title and description (paraphrased, not copied).
The answer is the record itself, so nothing needs hand-labelling. Each query runs
as plain BM25 over a set of fields; a record counts as found when any of its
documents ranks in the top k.

Arms (fields searched):
  metadata          title + curator description + subjects          (what the repository has)
  columns           column names                                    (data only, no curator text)
  columns+types     column names + semantic type labels
  autoddg           AutoDDG description + search description
  data              column names + types + AutoDDG                  (everything Auctus derives)
  all               metadata + data

Caveat: queries are written from the curator's text, so the metadata arm has an
echo advantage. The data-only arms don't see that text, so comparing them with each
other is fair; comparing them with metadata shows how findable a record would be
without a curator (self-deposits, Zenodo).

  python -m eval.known_item [--instance ultraviolet] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any

from eval.description_eval.common import DATA_DIR, INSTANCES, ask_json, write_json

QUERY_MODEL = "@vertexai/gemini-2.5-flash"
QUERY_PROMPT = """A researcher is looking for this dataset in a search engine. Write 3 search queries they might
type, each a different kind:
1. a short keyword query (2-4 words) about the topic;
2. a query naming what the data measures or contains (variables, units, place, period), 4-8 words;
3. a natural-language request, 6-12 words.
Paraphrase: don't copy the title, and use at most two words from it in any query.

Title: {title}
Description: {description}

Return JSON only: {{"queries": ["...", "...", "..."]}}
"""

ARMS: dict[str, dict[str, Any]] = {
    "metadata": {"fields": ["title", "description", "source_record.subjects"]},
    "columns": {"nested": ["profiler_metadata.columns.name"]},
    "columns+types": {"nested": ["profiler_metadata.columns.name", "profiler_metadata.columns.semantic_type_label"]},
    "autoddg": {"fields": ["autoddg_description", "autoddg_search_description"]},
    "data": {"fields": ["autoddg_description", "autoddg_search_description"],
             "nested": ["profiler_metadata.columns.name", "profiler_metadata.columns.semantic_type_label"]},
    "all": {"fields": ["title", "description", "source_record.subjects", "autoddg_description",
                       "autoddg_search_description"],
            "nested": ["profiler_metadata.columns.name", "profiler_metadata.columns.semantic_type_label"]},
    # The real search endpoint (keyword + vector), with each ranking it offers.
    "api:legacy": {"api": "legacy"},
    "api:weighted": {"api": "weighted"},
    "api:hybrid": {"api": "hybrid"},
}
API_URL = os.getenv("AUCTUS_API", "http://backend:8000/api/v1/search")


def arm_query(text: str, arm: dict[str, Any], domain: str) -> dict[str, Any]:
    should: list[dict[str, Any]] = []
    if arm.get("fields"):
        should.append({"multi_match": {"query": text, "fields": arm["fields"]}})
    if arm.get("nested"):
        should.append({"nested": {"path": "profiler_metadata.columns", "score_mode": "sum",
                                  "query": {"multi_match": {"query": text, "fields": arm["nested"]}}}})
    return {"bool": {"filter": [{"term": {"domain": domain}}], "should": should, "minimum_should_match": 1}}


def records(os_client, index: str, domain: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    body = {"size": 2000, "_source": ["source_record.record_id", "source_record.title", "description"],
            "query": {"term": {"domain": domain}}}
    for hit in os_client.search(index=index, body=body)["hits"]["hits"]:
        src = hit["_source"]
        rid = (src.get("source_record") or {}).get("record_id")
        if rid and rid not in out:
            out[rid] = {"title": (src.get("source_record") or {}).get("title", ""), "description": src.get("description", "")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default="ultraviolet", choices=sorted(INSTANCES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--arms", nargs="+", help="only these arms")
    args = parser.parse_args()

    from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

    os_client, domain = get_client(), INSTANCES[args.instance]
    recs = records(os_client, AUCTUS_INDEX_NAME, domain)
    queries = []
    for rid, rec in sorted(recs.items()):
        if not rec["description"]:
            continue
        reply = ask_json(QUERY_PROMPT.format(title=rec["title"], description=rec["description"][:3000]), model=QUERY_MODEL)
        for n, q in enumerate((reply or {}).get("queries", [])[:3]):
            queries.append({"record": rid, "kind": ("keyword", "content", "natural")[n], "text": q})
    print(f"{len(recs)} records, {len(queries)} queries")

    import httpx

    rows = []
    arms = {k: v for k, v in ARMS.items() if not args.arms or k in args.arms}
    for q in queries:
        for name, arm in arms.items():
            if arm.get("api"):
                docs = httpx.post(API_URL, json={"keywords": q["text"], "source": [domain], "limit": 50,
                                                 "ranking": arm["api"]}, timeout=60).json().get("results", [])
            else:
                docs = [h["_source"] for h in os_client.search(index=AUCTUS_INDEX_NAME, body={
                    "size": 100, "_source": ["source_record.record_id"], "query": arm_query(q["text"], arm, domain)})["hits"]["hits"]]
            ranked = list(dict.fromkeys((d.get("source_record") or {}).get("record_id") for d in docs))
            rank = ranked.index(q["record"]) + 1 if q["record"] in ranked else None
            rows.append({**q, "arm": name, "rank": rank})

    summary: dict[str, Any] = {"instance": args.instance, "records": len(recs), "queries": len(queries), "k": args.k, "arms": {}}
    for name in arms:
        mine = [r for r in rows if r["arm"] == name]
        by_kind = defaultdict(list)
        for r in mine:
            by_kind[r["kind"]].append(r)
        hit = lambda rs: round(sum(1 for r in rs if r["rank"] and r["rank"] <= args.k) / len(rs), 3) if rs else None  # noqa: E731
        summary["arms"][name] = {
            f"hit@{args.k}": hit(mine), "hit@1": round(sum(1 for r in mine if r["rank"] == 1) / len(mine), 3),
            "mrr": round(statistics.fmean(1 / r["rank"] if r["rank"] else 0 for r in mine), 3),
            **{f"hit@{args.k}_{k}": hit(v) for k, v in by_kind.items()}}
    tag = "-api" if all(a.startswith("api:") for a in arms) else ""
    write_json(DATA_DIR / "results" / f"known-item-{args.instance}{tag}.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
