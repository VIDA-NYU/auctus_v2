"""Does fixing AutoDDG's inputs and prompt stop it making things up? (dataset level, per file)

The 2026-11-26 dataset-level eval traced AutoDDG's errors to two places:
  inputs  storage.arq_worker.build_profile_text never sends the file's date range (temporal_coverage),
          so years come from the few sample rows ("2001-2020 as shown in the sample"); it keeps `std`
          but the profile key is `stddev`; tables with 40+ columns get no min/max at all; distinct
          counts are atlas-profiler estimates that can exceed the row count (418 for 399 rows)
  prompt  AutoDDG's own prompts.yaml asks "describe what this dataset can be used for", shows the
          sample first and never says to take ranges from the profile, never says to state only what
          the inputs support, and gives no title (so it invents one: "titled 'COVID Zipcode Rates'");
          the semantic profile's "Function/Usage context" turns into "year acts as an aggregation key"

Three arms on the same files (the dataset-level sample: up to 4 per record), same model (AutoDDG's,
Gemini 2.5 Flash, temperature 0), same 100-word target, AutoDDG's prompt assembly rebuilt here so
only the named pieces differ:
  stock         AutoDDG's prompts.yaml + build_profile_text, as ingestion runs today (regenerated)
  inputs_fixed  stock prompt + fixed profile text
  all_fixed     fixed prompt + fixed profile text + semantic profile without usage context
  fixed_with_uses  all_fixed, but uses kept ("what it can be used for, naming only uses its columns support")
Scored with dataset_level.py's checks (grounding, Koesten vs the frozen evidence reference, tell-apart).

  MODULE=eval.description_eval.autoddg_fix ./run_in_worker.sh [--records ID ...]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval.claims import claimify_direct, rate
from eval.description_eval.collect import load_profile
from eval.description_eval.common import COLLECTED_DIR, DATA_DIR, GOLD_DIR, ask_text, read_json, write_json
from eval.description_eval.dataset_level import (MAX_FILES, file_evidence, koesten_reference, koesten_states,
                                                 tell_apart)
from eval.description_eval.evidence import record_evidence

OUT = DATA_DIR / "autoddg_fix"
MODEL = "@vertexai/gemini-2.5-flash"  # AUTODDG_MODEL at ingestion
WORDS = 100                           # description_words at ingestion
ARMS = ("stock", "inputs_fixed", "all_fixed", "fixed_with_uses")

FIXED_PROMPT = {
    "introduction": (
        "Answer the question using the following information.\n\n"
        "First, consider the dataset sample. It shows only a few rows: take ranges, dates and counts from "
        "the profile below, never from these rows.\n\n{dataset_sample}\n"),
    "profile_instruction": (
        "Additionally, the dataset profile, computed over every row, is as follows:\n\n{dataset_profile}\n\n"
        "Based on this profile, please add sentence(s) to enrich the dataset description, using its ranges, "
        "dates and counts as given.\n"),
    "semantic_instruction": (
        "Furthermore, the semantic profile of the dataset columns is as follows:\n{semantic_profile}\n\n"
        "Based on this information, please add sentence(s) discussing the semantic profile in the description.\n"),
    "topic_instruction": (
        "Moreover, the dataset topic is: {data_topic}. Based on this topic, please add sentence(s) describing "
        "what the data covers.\n"),
    "closing_instruction": (
        "Question: Based on the information above and the requirements, provide a dataset description in "
        "sentences. State only what the sample, profile, semantic profile and topic support: do not give the "
        "dataset a title or name, do not guess who made it, why, or what it could be used for, and do not "
        "describe ranges from the sample rows. Use only natural, readable sentences without special "
        "formatting.\n\nAnswer:\n"),
}


# Ablation (Cedric, 2026-11-27): uses are search terms, so keep them, but only ones the columns
# support; everything else as all_fixed.
WITH_USES = {
    **FIXED_PROMPT,
    "topic_instruction": (
        "Moreover, the dataset topic is: {data_topic}. Based on this topic, please add sentence(s) describing "
        "what this dataset can be used for, naming only uses its columns support.\n"),
    "closing_instruction": FIXED_PROMPT["closing_instruction"].replace(
        "do not guess who made it, why, or what it could be used for", "do not guess who made it or why"),
}


def stock_prompt() -> dict[str, str]:
    from autoddg.utils import load_prompts

    p = load_prompts()["dataset_description"]
    return {k: p[k] for k in ("introduction", "profile_instruction", "semantic_instruction", "topic_instruction",
                              "closing_instruction", "system_message")}


# Ingestion's build_profile_text as it was before the input fix (auctus_v2 825daaa^), frozen here so
# the "stock" arm stays stock after the fix landed in storage.arq_worker.
WIDE_TABLE_COLUMN_THRESHOLD = 40
MAX_COLUMNS_IN_PROFILE = 80
_COLUMN_CORE_FIELDS = ("name", "structural_type", "semantic_types")
_COLUMN_STAT_FIELDS = ("num_distinct_values", "mean", "std", "min", "max")


def stock_profile_text(record: dict[str, Any]) -> str:
    """Render a trimmed, content-focused profile as a JSON string for AutoDDG.

    AutoDDG injects this string into its prompt (use_profile=True). We keep only
    fields that describe *what the dataset contains* — size, column schema, and
    geographic coverage — and drop operational/rendering fields (telemetry,
    profiling times, raw geohash grids, redundant keyword splits). This is the
    profile-aware usage emphasized in the AutoDDG paper. Returns "" if there is
    nothing useful to report. The selection is deterministic (no LLM).

    The per-column detail is *adaptive* to table width: narrow tables keep full
    numeric stats, wide tables keep only the core meaning-bearing fields and cap
    the number of columns, so the prompt stays focused on wide datasets. The
    keep/drop rules and thresholds are documented in
    "docs/Profiler_metadata — Field Reference.md".
    """
    pm = record.get("profiler_metadata")
    if not isinstance(pm, dict):
        return ""

    profile: dict[str, Any] = {}
    for key in ("nb_rows", "nb_columns", "types"):
        if pm.get(key) is not None:
            profile[key] = pm[key]

    columns = [col for col in (pm.get("columns") or []) if isinstance(col, dict)]
    # Decide per-column detail by table width: wide tables -> core fields only.
    n_columns = pm.get("nb_columns") or len(columns)
    is_wide = isinstance(n_columns, int) and n_columns >= WIDE_TABLE_COLUMN_THRESHOLD
    kept_fields = _COLUMN_CORE_FIELDS if is_wide else _COLUMN_CORE_FIELDS + _COLUMN_STAT_FIELDS

    trimmed_columns: list[dict[str, Any]] = []
    for col in columns[:MAX_COLUMNS_IN_PROFILE]:
        entry = {
            key: col[key] for key in kept_fields if col.get(key) not in (None, [], "")
        }
        if entry:
            trimmed_columns.append(entry)

    if trimmed_columns:
        profile["columns"] = trimmed_columns
        if len(columns) > MAX_COLUMNS_IN_PROFILE:
            # Tell the LLM the schema was truncated so it doesn't over-claim coverage.
            profile["columns_truncated"] = {
                "shown": len(trimmed_columns),
                "total": len(columns),
            }
    elif pm.get("attribute_keywords"):
        # Fallback when profiling produced no columns (edge cases): names only.
        profile["column_names"] = pm["attribute_keywords"]

    # Geographic coverage: clean label + bbox only (drop the raw geohash grids).
    spatial = record.get("spatial_coverage")
    if isinstance(spatial, dict) and (spatial.get("label") or spatial.get("bbox")):
        profile["spatial_coverage"] = {
            k: spatial[k] for k in ("label", "bbox") if spatial.get(k)
        }

    return json.dumps(profile, ensure_ascii=False) if profile else ""


def fixed_profile_text(profile: dict[str, Any]) -> str:
    """build_profile_text with the input bugs fixed: dates, stddev, min/max for wide tables, distinct
    counts capped at the row count."""
    pm = profile.get("profiler_metadata") or {}
    out: dict[str, Any] = {k: pm[k] for k in ("nb_rows", "nb_columns", "types") if pm.get(k) is not None}
    columns = [c for c in (pm.get("columns") or []) if isinstance(c, dict)]
    n = pm.get("nb_columns") or len(columns)
    wide = isinstance(n, int) and n >= WIDE_TABLE_COLUMN_THRESHOLD
    fields = ("name", "structural_type", "semantic_types", "num_distinct_values", "min", "max") + (() if wide else ("mean", "stddev"))
    rows = pm.get("nb_rows")
    cols = []
    for c in columns[:MAX_COLUMNS_IN_PROFILE]:
        e = {k: c[k] for k in fields if c.get(k) not in (None, [], "")}
        if isinstance(rows, int) and isinstance(e.get("num_distinct_values"), (int, float)):
            e["num_distinct_values"] = min(e["num_distinct_values"], rows)
        if e:
            cols.append(e)
    if cols:
        out["columns"] = cols
        if len(columns) > MAX_COLUMNS_IN_PROFILE:
            out["columns_truncated"] = {"shown": len(cols), "total": len(columns)}
    dates = profile.get("temporal_coverage") or {}
    if dates.get("start"):
        out["temporal_coverage"] = {"start": dates["start"], "end": dates.get("end")}
    spatial = profile.get("spatial_coverage")
    if isinstance(spatial, dict) and (spatial.get("label") or spatial.get("bbox")):
        out["spatial_coverage"] = {k: spatial[k] for k in ("label", "bbox") if spatial.get(k)}
    return json.dumps(out, ensure_ascii=False, default=str) if out else ""


def describe(arm: str, profile: dict[str, Any]) -> str:
    stock = stock_prompt()
    seg = {**stock, **FIXED_PROMPT} if arm == "all_fixed" else {**stock, **WITH_USES} if arm == "fixed_with_uses" else stock
    profile_text = stock_profile_text(profile) if arm == "stock" else fixed_profile_text(profile)
    semantic = profile.get("autoddg_semantic_profile")
    if arm in ("all_fixed", "fixed_with_uses") and isinstance(semantic, str):
        semantic = re.sub(r"\s*Function/Usage context:[^.\n]*\.?", "", semantic)
    topic = profile.get("autoddg_topic")
    parts = [seg["introduction"].format(dataset_sample=profile.get("sample") or "")]
    if profile_text:
        parts.append(seg["profile_instruction"].format(dataset_profile=profile_text))
    if semantic:
        parts.append(seg["semantic_instruction"].format(semantic_profile=semantic))
    if topic:
        parts.append(seg["topic_instruction"].format(data_topic=topic))
    parts += [seg["closing_instruction"], f"Target length: approximately {WORDS} words."]
    return ask_text("\n".join(parts), model=MODEL, max_tokens=8000, system=stock["system_message"].strip()).strip()


def run_record(storage, rid: str) -> dict[str, Any]:
    gold = read_json(GOLD_DIR / f"{rid}.json")
    ev = record_evidence(storage, rid)
    collected = read_json(COLLECTED_DIR / f"{rid}.json")["files"]
    ids = {f["file_key"]: f["dataset_id"] for f in gold["files"]}
    row: dict[str, Any] = {"record": rid, "files": [], "tell_apart": {}}
    described: dict[str, list] = {a: [] for a in ARMS}
    for key, info in [(k, i) for k, i in sorted(collected.items()) if i.get("ufd") and k in ids][:MAX_FILES]:
        try:
            profile = load_profile(storage, ids[key])
        except Exception:
            continue
        evidence = file_evidence(ev, key, profile)
        ref = koesten_reference(key, evidence)
        entry: dict[str, Any] = {"file": key, "arms": {}}
        for arm in ARMS:
            text = describe(arm, profile)
            claims = claimify_direct(text, ev["title"], {**ev, "evidence": evidence}, Counter())
            labels = [c["label"] for c in claims]
            states = [s for s in koesten_states(text, ref) if s is not None]
            entry["arms"][arm] = {
                "text": text, "words": len(text.split()), "n_claims": len(claims),
                "flagged": round((labels.count("unsupported") + labels.count("contradicted")) / len(labels), 3) if labels else None,
                "contradicted": rate(labels, "contradicted"),
                "koesten_ref_recall": rate(states, "stated", "partial"),
                "flagged_claims": [f"[{c['label']}] {c['claim']}" for c in claims if c["label"] != "supported"]}
            described[arm].append((key, text, profile))
        row["files"].append(entry)
    for arm in ARMS:
        if len(described[arm]) > 1:
            row["tell_apart"][arm] = tell_apart(described[arm][:8])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    rids = args.records or [read_json(p)["record_id"] for p in sorted(GOLD_DIR.glob("*.json"))
                            if read_json(p).get("domain") == "ultraviolet.library.nyu.edu"
                            and len(((read_json(p).get("curator") or {}).get("description") or "")) > 80]

    def run(rid):
        try:
            return run_record(storage, rid)
        except Exception as exc:
            return {"record": rid, "error": repr(exc), "files": [], "tell_apart": {}}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, rids))
    files = [f for r in rows for f in r["files"]]
    summary: dict[str, Any] = {"records": len(rows), "files": len(files),
                               "errors": [(r["record"], r["error"]) for r in rows if r.get("error")], "arms": {}}
    for arm in ARMS:
        a = [f["arms"][arm] for f in files if arm in f["arms"]]
        s = {m: round(statistics.mean(x[m] for x in a if x[m] is not None), 3) if a else None
             for m in ("flagged", "contradicted", "koesten_ref_recall", "words", "n_claims")}
        ta = [r["tell_apart"][arm] for r in rows if arm in r["tell_apart"]]
        s["tell_apart"] = f"{sum(t['right'] for t in ta)}/{sum(t['files'] for t in ta)}"
        if arm != "stock":  # paired per file vs stock
            for m in ("flagged", "contradicted", "koesten_ref_recall"):
                d = [f["arms"][arm][m] - f["arms"]["stock"][m] for f in files
                     if f["arms"].get(arm, {}).get(m) is not None and f["arms"].get("stock", {}).get(m) is not None]
                s[f"{m}_vs_stock"] = round(statistics.mean(d), 3) if d else None
        summary["arms"][arm] = s
    write_json(OUT / ("results.json" if not args.records else "results-subset.json"), {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
