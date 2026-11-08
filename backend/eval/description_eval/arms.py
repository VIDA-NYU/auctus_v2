"""Does more context make AutoDDG describe datasets the way curators do?

Same files, same sample, same semantic profile and topic (the stored ones from
ingestion); only the profile string handed to AutoDDG's describe_dataset changes:

  base   what ingestion sends today (storage.arq_worker.build_profile_text)
  types  + what Auctus inferred per column: semantic types, codebook value labels,
           identifier schemes (no prose from the documentation)
  docs   + the record's README / codebook text
  paper  + title and abstract of the paper the record links to (OpenAlex, by DOI)
  all    types + docs + paper

Every description is graded with compare.DESCRIPTION_JUDGE against the curator's
record description, with the same evidence for every arm (documentation, paper
abstract, column list), so a claim backed by the README counts as supported in
every arm, not only in the arm that saw it. Caveat: curators often write the
record description from the README or the paper, so recall gains in docs/paper
partly measure copying; that is what a curator draft would do too.

  python -m eval.description_eval.arms [--per-record 4] [--arms base types ...]

Generations and judge calls are cached under data/arms/, so reruns are free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from eval.description_eval.collect import load_profile
from eval.description_eval.common import DATA_DIR, GOLD_DIR, ask_json, read_json, write_json
from eval.description_eval.compare import DESCRIPTION_JUDGE

ARMS = ("base", "types", "docs", "paper", "all")
ARMS_DIR = DATA_DIR / "arms"
JUDGE_MODEL = "@vertexai/anthropic.claude-sonnet-5-5"  # AutoDDG generates with Gemini
DOC_CHARS = 6000
PAPER_CHARS = 3000
UV_DOI_PREFIX = "10.58153/"  # UltraViolet's own DOIs: other datasets, not papers


def column_semantics(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-column meaning Auctus inferred, beyond the storage type."""
    out = []
    for col in (profile.get("profiler_metadata") or {}).get("columns") or []:
        if not isinstance(col, dict):
            continue
        entry: dict[str, Any] = {"name": col.get("name")}
        types = [t.rsplit("/", 1)[-1] for t in col.get("semantic_types") or []]
        if types:
            entry["semantic_types"] = types
        if col.get("value_labels_text"):
            entry["value_labels"] = str(col["value_labels_text"])[:300]
        if col.get("identifier_scheme"):
            entry["identifier_scheme"] = col["identifier_scheme"]
        if len(entry) > 1:
            out.append(entry)
    return out[:80]


def documentation_text(record_id: str) -> str:
    path = DATA_DIR / "sources" / "ultraviolet" / f"{record_id}.json"
    if not path.exists():
        return ""
    parts, used = [], 0
    for doc in sorted(read_json(path)["docs"], key=lambda d: -d["column_hits"]):
        if used >= DOC_CHARS:
            break
        chunk = f"[{doc['name']}]\n{doc['text'][: DOC_CHARS - used]}"
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)


def _abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    words = sorted((i, w) for w, positions in inverted.items() for i in positions)
    return " ".join(w for _, w in words)


def linked_paper(record: dict[str, Any]) -> dict[str, str] | None:
    """Title and abstract of the first linked publication with a DOI (cached)."""
    cache = ARMS_DIR / "papers" / f"{record['id']}.json"
    if cache.exists():
        return read_json(cache) or None
    paper = None
    for rel in (record.get("metadata") or {}).get("related_identifiers") or []:
        doi = rel.get("identifier", "")
        if (rel.get("scheme") == "doi" and not doi.startswith(UV_DOI_PREFIX)
                and (rel.get("resource_type") or {}).get("id", "").startswith("publication")):
            try:
                work = httpx.get(f"https://api.openalex.org/works/doi:{doi}", timeout=30).json()
            except Exception:
                continue
            abstract = _abstract(work.get("abstract_inverted_index"))
            if work.get("title"):
                paper = {"doi": doi, "title": work["title"], "abstract": abstract[:PAPER_CHARS]}
                break
    write_json(cache, paper or {})
    return paper


def arm_profile(base_text: str, arm: str, semantics: list, docs: str, paper: dict | None) -> str | None:
    """The profile string for an arm, or None if the arm adds nothing for this file."""
    profile = json.loads(base_text) if base_text else {}
    added = False
    if arm in ("types", "all") and semantics:
        profile["column_semantics"] = semantics
        added = True
    if arm in ("docs", "all") and docs:
        profile["documentation"] = docs
        added = True
    if arm in ("paper", "all") and paper:
        profile["related_paper"] = paper
        added = True
    if arm != "base" and not added:
        return None
    return json.dumps(profile, ensure_ascii=False)


def generate(autoddg, profile: dict[str, Any], profile_text: str) -> str:
    key = hashlib.sha256(f"{profile['id']}\n{profile_text}".encode()).hexdigest()
    cache = ARMS_DIR / "gen" / f"{key}.json"
    if cache.exists():
        return read_json(cache)["description"]
    semantic = profile.get("autoddg_semantic_profile")
    topic = profile.get("autoddg_topic")
    _prompt, description = autoddg.describe_dataset(
        dataset_sample=profile["sample"], dataset_profile=profile_text, use_profile=True,
        semantic_profile=semantic, use_semantic_profile=bool(semantic), data_topic=topic, use_topic=bool(topic))
    write_json(cache, {"id": profile["id"], "description": description})
    return description


def pick_files(gold: dict[str, Any], per_record: int) -> list[dict[str, Any]]:
    files = sorted(gold["files"], key=lambda f: f["file_key"])
    if len(files) <= per_record:
        return files
    return random.Random(gold["record_id"]).sample(files, per_record)


def judge(curator: str, evidence: str, file_key: str, description: str) -> dict[str, Any]:
    verdict = ask_json(DESCRIPTION_JUDGE.format(curator=curator[:6000], context=evidence[:14000],
                                                file_key=file_key, generated=description[:6000]), model=JUDGE_MODEL)
    facts, claims = verdict.get("curator_facts") or [], verdict.get("generated_claims") or []
    return {"fact_recall": sum(bool(x.get("covered")) for x in facts) / len(facts) if facts else None,
            "unsupported_rate": sum(not x.get("supported") for x in claims) / len(claims) if claims else None,
            "facts": facts, "claims": claims}


def paired(rows: list[dict], arm: str, metric: str) -> dict[str, Any]:
    """Mean difference arm - base over files that have both, with a bootstrap 95% CI."""
    by_file: dict[str, dict[str, float]] = {}
    for r in rows:
        if r[metric] is not None:
            by_file.setdefault(r["dataset_id"], {})[r["arm"]] = r[metric]
    diffs = [v[arm] - v["base"] for v in by_file.values() if arm in v and "base" in v]
    if not diffs:
        return {"n": 0}
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"n": len(diffs), "mean_diff": round(statistics.fmean(diffs), 3),
            "ci95": [round(means[50], 3), round(means[1949], 3)],
            "higher": sum(d > 0 for d in diffs), "lower": sum(d < 0 for d in diffs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-record", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from storage.arq_worker import build_profile_text, get_autoddg
    from storage.minio_client import get_storage_client

    autoddg = get_autoddg()
    if autoddg is None:
        raise SystemExit("AutoDDG unavailable (AUTODDG_ENABLED, PORTKEY_API_KEY, VPN)")
    storage = get_storage_client()
    records = {r["id"]: r for r in read_json(DATA_DIR / "records-ultraviolet.json")}

    tasks = []  # (gold, file, profile, arm, profile_text, evidence)
    for gold_path in sorted(GOLD_DIR.glob("*.json")):
        gold = read_json(gold_path)
        curator = (gold.get("curator") or {}).get("description")
        if not curator:
            continue
        docs = documentation_text(gold["record_id"])
        paper = linked_paper(records[gold["record_id"]]) if gold["record_id"] in records else None
        for f in pick_files(gold, args.per_record):
            try:
                profile = load_profile(storage, f["dataset_id"])
            except Exception:
                continue
            if not profile.get("sample"):
                continue
            base_text = build_profile_text(profile)
            columns = ", ".join(c.get("name", "") for c in (profile.get("profiler_metadata") or {}).get("columns") or [])
            evidence = "\n\n".join(x for x in (
                docs and f"Documentation:\n{docs}",
                paper and f"Linked paper: {paper['title']}\n{paper['abstract']}",
                f"Columns: {columns[:3000]}") if x)
            for arm in args.arms:
                text = arm_profile(base_text, arm, column_semantics(profile), docs, paper)
                if text is not None:
                    tasks.append((gold, f, profile, arm, text, evidence, curator))
    print(f"{len(tasks)} generations over {len({t[1]['dataset_id'] for t in tasks})} files")

    def run(task) -> dict[str, Any]:
        gold, f, profile, arm, text, evidence, curator = task
        try:
            description = generate(autoddg, profile, text)
            graded = judge(curator, evidence, f["file_key"], description)
        except Exception as exc:
            return {"record": gold["record_id"], "dataset_id": f["dataset_id"], "arm": arm, "error": repr(exc),
                    "fact_recall": None, "unsupported_rate": None}
        return {"record": gold["record_id"], "dataset_id": f["dataset_id"], "file": f["file_key"], "arm": arm,
                "words": len(description.split()), "description": description, **graded}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))

    summary: dict[str, Any] = {"judge": JUDGE_MODEL, "per_record": args.per_record, "arms": {}}
    for arm in args.arms:
        mine = [r for r in rows if r["arm"] == arm and r.get("fact_recall") is not None]
        unsupported = [r["unsupported_rate"] for r in mine if r["unsupported_rate"] is not None]
        summary["arms"][arm] = {
            "files": len(mine), "errors": sum(1 for r in rows if r["arm"] == arm and r.get("error")),
            "mean_fact_recall": round(statistics.fmean(r["fact_recall"] for r in mine), 3) if mine else None,
            "mean_unsupported_rate": round(statistics.fmean(unsupported), 3) if unsupported else None,
            "median_words": statistics.median(r["words"] for r in mine) if mine else None,
            **({} if arm == "base" else {"vs_base_recall": paired(rows, arm, "fact_recall"),
                                         "vs_base_unsupported": paired(rows, arm, "unsupported_rate")}),
        }
    write_json(ARMS_DIR / "results.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
