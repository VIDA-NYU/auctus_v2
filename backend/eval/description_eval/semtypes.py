"""Do semantic column types in AutoDDG's input improve its descriptions?

Inputs are what ingestion has (the data sample, the profiler output, the title).
The README / codebook and the curator's description are validation only, except
in the oracle arm, which is labelled as such.

  A0  coarse profiler types, no AutoDDG semantic profile   (do column semantics matter?)
  A   today's input: profile + AutoDDG's own semantic profile (baseline)
  B   A + semantic types from a small LLM typer that sees only column names and values
  C   A + the README's column meanings (oracle: the ceiling, not a deployable arm)

Scored three ways:
  types     the typer's labels and AutoDDG's semantic profile vs the README definitions
            (COLUMN_JUDGE, column name withheld), on the documented columns of the files
  facts     the curator's key facts, each tagged content (what is measured, units, place,
            time) or provenance (who, why, how collected); recall per kind, so an effect
            on content isn't washed out by provenance facts no column type can supply
  claims    the description's claims backed by the curator text, README or column list

  python -m eval.description_eval.semtypes [--per-record 4]

Uses the same files as eval.description_eval.arms (same seed). Cached under data/semtypes/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from eval.description_eval.arms import documentation_text, paired, pick_files
from eval.description_eval.collect import load_profile, parse_semantic_profile
from eval.description_eval.common import DATA_DIR, GOLD_DIR, ask_json, normalize_name, read_json, write_json
from eval.description_eval.compare import COLUMN_JUDGE

ARMS = ("A0", "A", "B", "C")
OUT_DIR = DATA_DIR / "semtypes"
TYPER_MODEL = "@vertexai/gemini-2.5-flash"  # what ingestion would run
JUDGE_MODEL = "@vertexai/anthropic.claude-sonnet-5-5"
MAX_COLUMNS = 80
SAMPLE_VALUES = 8

TYPER_PROMPT = """Give each column of this table a semantic type: what the values ARE, in 2 to 6 words,
as specific as the evidence allows (e.g. "US ZIP code", "agreement scale 1-5", "ISO country code",
"participant ID", "survey completion date", "household size count", "free-text comment").
Use only the column name, the table's title and the sample values. If the values don't
say, give the most specific type you can defend, not a guess about the study.

Table: {title}
Columns (name: sample of distinct values):
{columns}

Return JSON only: {{"<column name>": "<semantic type>", ...}}
"""

FACTS_JUDGE = """Compare an automatically generated dataset description with the curator's description of the record it belongs to.

Curator's record description:
<<<
{curator}
>>>

Documentation excerpt and column list (for checking claims):
<<<
{context}
>>>

Generated description of one file in this record ({file_key}):
<<<
{generated}
>>>

1. List the key facts in the curator's description (at most 8). Tag each "content" (what the
   data contains: variables or measures, units, places covered, time span, size) or "provenance"
   (who made it, why, how or where it was collected, funding, related publications). Say whether
   the generated description states it.
2. List the generated description's factual claims (at most 10). For each, say whether the curator's
   description, the documentation or the column list supports it.

Return JSON only:
{{"curator_facts": [{{"fact": "...", "kind": "content|provenance", "covered": true|false}}],
  "generated_claims": [{{"claim": "...", "supported": true|false}}]}}
"""


def column_values(sample_csv: str) -> dict[str, list[str]]:
    frame = pd.read_csv(io.StringIO(sample_csv), dtype=str, keep_default_na=False)
    return {str(c): [v for v in dict.fromkeys(frame[c].tolist()) if v.strip()][:SAMPLE_VALUES]
            for c in list(frame.columns)[:MAX_COLUMNS]}


def semantic_types(profile: dict[str, Any]) -> dict[str, str]:
    """The typer: one call per table, names and values only."""
    values = column_values(profile["sample"])
    listing = "\n".join(f"- {name}: {', '.join(v[:40] for v in vals)}" for name, vals in values.items())
    reply = ask_json(TYPER_PROMPT.format(title=profile.get("title", ""), columns=listing), model=TYPER_MODEL)
    return {str(k): str(v) for k, v in (reply or {}).items() if isinstance(v, (str, int, float))}


def oracle_meanings(gold: dict[str, Any], file_key: str) -> dict[str, str]:
    out = {}
    for key, columns in (gold.get("dictionary") or {}).items():
        if key in (file_key, "*"):
            for col in columns:
                meaning = " ".join(filter(None, [col.get("definition", ""), col.get("type", "")]))
                if meaning:
                    out[col["column"]] = " ".join(meaning.split()[:20])
    return out


def arm_profile(base_text: str, arm: str, typed: dict[str, str], oracle: dict[str, str]) -> str | None:
    profile = json.loads(base_text) if base_text else {}
    if arm == "B":
        if not typed:
            return None
        profile["column_semantic_types"] = typed
    elif arm == "C":
        if not oracle:
            return None
        profile["column_meanings_from_documentation"] = oracle
    return json.dumps(profile, ensure_ascii=False)


def generate(autoddg, profile: dict[str, Any], profile_text: str, use_semantic: bool) -> str:
    # Same key as eval.description_eval.arms for arm A, so its base generations are reused.
    tag = "" if use_semantic else "\nno-semantic-profile"
    key = hashlib.sha256(f"{profile['id']}\n{profile_text}{tag}".encode()).hexdigest()
    cache = DATA_DIR / "arms" / "gen" / f"{key}.json"
    if cache.exists():
        return read_json(cache)["description"]
    semantic = profile.get("autoddg_semantic_profile") if use_semantic else None
    topic = profile.get("autoddg_topic")
    _prompt, description = autoddg.describe_dataset(
        dataset_sample=profile["sample"], dataset_profile=profile_text, use_profile=True,
        semantic_profile=semantic, use_semantic_profile=bool(semantic), data_topic=topic, use_topic=bool(topic))
    write_json(cache, {"id": profile["id"], "description": description})
    return description


def judge_facts(curator: str, evidence: str, file_key: str, description: str) -> dict[str, Any]:
    verdict = ask_json(FACTS_JUDGE.format(curator=curator[:6000], context=evidence[:14000], file_key=file_key,
                                          generated=description[:6000]), model=JUDGE_MODEL)
    facts, claims = verdict.get("curator_facts") or [], verdict.get("generated_claims") or []

    def recall(kind: str | None) -> float | None:
        mine = [f for f in facts if kind is None or f.get("kind") == kind]
        return sum(bool(f.get("covered")) for f in mine) / len(mine) if mine else None

    return {"fact_recall": recall(None), "content_recall": recall("content"),
            "provenance_recall": recall("provenance"),
            "unsupported_rate": sum(not c.get("supported") for c in claims) / len(claims) if claims else None,
            "facts": facts, "claims": claims}


def judge_types(pairs: list[dict]) -> list[dict]:
    graded = []
    for start in range(0, len(pairs), 50):
        batch = pairs[start:start + 50]
        text = "\n\n".join(f"id {i}:\n  documentation: {p['definition']}"
                           + (f" (values: {p['labels'][:300]})" if p["labels"] else "")
                           + f"\n  generated: {p['generated']}" for i, p in enumerate(batch))
        reply = ask_json(COLUMN_JUDGE.format(pairs=text), model=JUDGE_MODEL)
        grades = {g.get("id"): g for g in reply if isinstance(g, dict)} if isinstance(reply, list) else {}
        graded += [{**p, "grade": (grades.get(i) or {}).get("grade")} for i, p in enumerate(batch)]
    return graded


def grade_counts(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    counts = {g: sum(r["grade"] == g for r in rows) for g in ("correct", "partial", "wrong")}
    return {"n": n, **counts, "correct_rate": round(counts["correct"] / n, 3) if n else None}


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-record", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from storage.arq_worker import build_profile_text, get_autoddg
    from storage.minio_client import get_storage_client

    autoddg = get_autoddg()
    if autoddg is None:
        raise SystemExit("AutoDDG unavailable (AUTODDG_ENABLED, PORTKEY_API_KEY, VPN)")
    storage = get_storage_client()

    files = []  # (gold, file, profile, curator, evidence)
    for gold_path in sorted(GOLD_DIR.glob("*.json")):
        gold = read_json(gold_path)
        curator = (gold.get("curator") or {}).get("description")
        if not curator:
            continue
        docs = documentation_text(gold["record_id"])
        for f in pick_files(gold, args.per_record):
            try:
                profile = load_profile(storage, f["dataset_id"])
            except Exception:
                continue
            if not profile.get("sample"):
                continue
            columns = ", ".join(c.get("name", "") for c in (profile.get("profiler_metadata") or {}).get("columns") or [])
            evidence = "\n\n".join(x for x in (docs and f"Documentation:\n{docs}", f"Columns: {columns[:3000]}") if x)
            files.append((gold, f, profile, curator, evidence))

    with ThreadPoolExecutor(args.workers) as pool:
        typed_all = list(pool.map(lambda t: _safe(semantic_types, t[2]), files))
    print(f"{len(files)} files typed")

    # 1. Column semantics vs the README: the typer's labels and AutoDDG's semantic profile.
    type_pairs, autoddg_pairs = [], []
    for (gold, f, profile, _c, _e), typed in zip(files, typed_all):
        semantic = {normalize_name(k): v for k, v in parse_semantic_profile(profile.get("autoddg_semantic_profile")).items()}
        typed_by = {normalize_name(k): v for k, v in (typed or {}).items()}
        for key, columns in (gold.get("dictionary") or {}).items():
            if key not in (f["file_key"], "*"):
                continue
            for col in columns:
                name = normalize_name(col["column"])
                if not col.get("definition") or name not in typed_by or name not in semantic:
                    continue  # compare on columns both sides describe
                base = {"record": gold["record_id"], "file": f["file_key"], "column": col["column"],
                        "definition": col["definition"], "labels": col.get("units", "")}
                type_pairs.append({**base, "generated": typed_by[name]})
                autoddg_pairs.append({**base, "generated": semantic[name]})
    column_result = {"typer": grade_counts(judge_types(type_pairs)),
                     "autoddg_semantic_profile": grade_counts(judge_types(autoddg_pairs))}

    # 2-3. Descriptions per arm.
    tasks = []
    for (gold, f, profile, curator, evidence), typed in zip(files, typed_all):
        base_text = build_profile_text(profile)
        for arm in ARMS:
            text = arm_profile(base_text, arm, typed or {}, oracle_meanings(gold, f["file_key"]))
            if text is not None:
                tasks.append((gold, f, profile, curator, evidence, arm, text))
    print(f"{len(tasks)} descriptions")

    def run(task) -> dict[str, Any]:
        gold, f, profile, curator, evidence, arm, text = task
        try:
            description = generate(autoddg, profile, text, use_semantic=arm != "A0")
            graded = judge_facts(curator, evidence, f["file_key"], description)
        except Exception as exc:
            return {"record": gold["record_id"], "dataset_id": f["dataset_id"], "arm": arm, "error": repr(exc),
                    "fact_recall": None, "content_recall": None, "provenance_recall": None, "unsupported_rate": None}
        return {"record": gold["record_id"], "dataset_id": f["dataset_id"], "file": f["file_key"], "arm": arm,
                "words": len(description.split()), "description": description, **graded}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))

    summary: dict[str, Any] = {"typer": TYPER_MODEL, "judge": JUDGE_MODEL, "columns": column_result, "arms": {}}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and not r.get("error")]
        summary["arms"][arm] = {
            "files": len(mine), "errors": sum(1 for r in rows if r["arm"] == arm and r.get("error")),
            **{m: mean([r[m] for r in mine]) for m in ("fact_recall", "content_recall", "provenance_recall",
                                                        "unsupported_rate")},
            "median_words": statistics.median(r["words"] for r in mine) if mine else None,
        }
        if arm != "A":
            summary["arms"][arm]["vs_A"] = {m: paired(_rebase(rows, "A"), arm, m)
                                           for m in ("fact_recall", "content_recall", "unsupported_rate")}
    write_json(OUT_DIR / "results.json", {"summary": summary, "rows": rows,
                                          "typed": {f[1]["dataset_id"]: t for f, t in zip(files, typed_all)}})
    print(json.dumps(summary, indent=1))


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def _rebase(rows: list[dict], reference: str) -> list[dict]:
    """arms.paired compares against "base"; rename the reference arm."""
    return [{**r, "arm": "base"} if r["arm"] == reference else r for r in rows]


if __name__ == "__main__":
    main()
