"""Compare AutoDDG's output with the depositors' documentation and curator metadata.

Reads gold/<record>.json (curator fields + extracted dictionaries) and
collected/<record>.json (AutoDDG + profiler output per file) and writes
results/description-eval-<date>.json and .md.

Comparisons
1. Column meanings: AutoDDG's per-column semantic profile vs the documented
   definition (and value labels). Coverage needs no LLM; a judge grades each pair
   correct / partial / wrong. Negative control: the same judge grades shuffled pairs
   (an AutoDDG text against another column's definition), which should come out wrong.
2. Descriptions: each file's AutoDDG description (UFD) vs the curator's record
   description: which of the curator's key facts it covers (recall), and which of its
   claims nothing supports (unsupported rate).
3. Topic vs subjects: does AutoDDG's topic match any curator subject?
4. Time range (no LLM): the profiler's temporal coverage vs the curator's dates.

LLM calls go through common.ask_json (cached on disk, so reruns are free). Run with
--no-llm for the deterministic parts only (no VPN needed).
"""

from __future__ import annotations

import random
import statistics
from datetime import date
from typing import Any

from eval.description_eval.common import (COLLECTED_DIR, GOLD_DIR, JUDGE_MODEL, RESULTS_DIR, ask_json,
                                          normalize_name, read_json, write_json)

COLUMN_BATCH = 50
SEED = 20261026

COLUMN_JUDGE = """You are checking automatically generated column descriptions against the dataset's own documentation.

For each column below you get the documentation's definition (and value labels, if any) and an
automatically generated description. Grade whether the generated description says what the
column is:
- "correct": same meaning (wording may differ; it need not repeat every value label)
- "partial": right general area but missing or blurring something important (e.g. calls a
  coded category a number, misses what the codes mean, wrong unit)
- "wrong": a different meaning, or a claim that contradicts the documentation

{pairs}

Return JSON only: [{{"id": <id>, "grade": "correct|partial|wrong", "reason": "<= 15 words"}}]
"""

DESCRIPTION_JUDGE = """Compare an automatically generated dataset description with the curator's description of the record it belongs to.

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

1. List the key facts in the curator's description (what, who, where, when, how collected; at most 8).
   For each, say whether the generated description states it.
2. List the generated description's factual claims (at most 10). For each, say whether the curator's
   description, the documentation or the column list supports it.

Return JSON only:
{{"curator_facts": [{{"fact": "...", "covered": true|false}}],
  "generated_claims": [{{"claim": "...", "supported": true|false}}]}}
"""

TOPIC_JUDGE = """Curator subjects for a dataset: {subjects}
Automatically generated topic: {topic}
Does the generated topic match at least one subject (same subject area, possibly broader or narrower)?
Return JSON only: {{"match": true|false, "closest_subject": "<subject or empty>"}}
"""


def _pairs(gold: dict[str, Any], collected: dict[str, Any]) -> tuple[list[dict], int, int]:
    """Documented columns matched to AutoDDG semantic descriptions: (pairs, documented, matched)."""
    pairs, documented = [], 0
    for file_key, columns in (gold.get("dictionary") or {}).items():
        targets = [f["file_key"] for f in gold["files"]] if file_key == "*" else [file_key]
        for target in targets:
            profiled = collected["files"].get(target) or {}
            semantic = profiled.get("semantic_columns") or {}
            by_name = {normalize_name(k): v for k, v in semantic.items()}
            in_file = {normalize_name(k) for k in profiled.get("profiler_columns") or {}}
            for col in columns:
                # Only documented columns that exist in this file ("*" dictionaries
                # aren't tied to one file, and a record's other files don't have them).
                if not col.get("definition") or normalize_name(col["column"]) not in in_file:
                    continue
                documented += 1
                generated = by_name.get(normalize_name(col["column"]))
                if generated:
                    pairs.append({"record": gold["record_id"], "file": target, "column": col["column"],
                                  "definition": col["definition"], "labels": col.get("units", ""),
                                  "generated": generated})
    return pairs, documented, len(pairs)


def judge_columns(pairs: list[dict]) -> list[dict]:
    graded = []
    for start in range(0, len(pairs), COLUMN_BATCH):
        batch = pairs[start:start + COLUMN_BATCH]
        text = "\n\n".join(
            f"id {i}: column `{p['column']}`\n  documentation: {p['definition']}"
            + (f" (values: {p['labels'][:300]})" if p['labels'] else "")
            + f"\n  generated: {p['generated']}" for i, p in enumerate(batch))
        grades = {g["id"]: g for g in ask_json(COLUMN_JUDGE.format(pairs=text))}
        graded += [{**p, "grade": (grades.get(i) or {}).get("grade"), "reason": (grades.get(i) or {}).get("reason")}
                   for i, p in enumerate(batch)]
    return graded


def shuffled(pairs: list[dict]) -> list[dict]:
    """Negative control: each generated text paired with another column's documentation (same file)."""
    rng = random.Random(SEED)
    out = []
    by_file: dict[tuple, list[dict]] = {}
    for p in pairs:
        by_file.setdefault((p["record"], p["file"]), []).append(p)
    for group in by_file.values():
        if len(group) < 2:
            continue
        others = group[1:] + group[:1]
        for p, q in zip(group, others):
            out.append({**p, "definition": q["definition"], "labels": q["labels"], "control": True})
    rng.shuffle(out)
    return out[:150]


def time_agreement(gold: dict[str, Any], collected: dict[str, Any]) -> list[dict]:
    """Profiler time range vs curator dates (years compared)."""
    years = []
    for d in (gold.get("curator") or {}).get("dates") or []:
        for part in str(d.get("date") or "").split("/"):
            if part[:4].isdigit():
                years.append(int(part[:4]))
    if not years:
        return []
    lo, hi = min(years), max(years)
    out = []
    for key, f in collected["files"].items():
        tc = (f or {}).get("temporal_coverage") or {}
        if not tc.get("start"):
            continue
        s, e = int(tc["start"][:4]), int((tc.get("end") or tc["start"])[:4])
        out.append({"record": gold["record_id"], "file": key, "profiler": [s, e], "curator": [lo, hi],
                    "overlap": s <= hi and lo <= e})
    return out


def compare(llm: bool = True, include_unverified: bool = True) -> dict[str, Any]:
    records = []
    for gold_path in sorted(GOLD_DIR.glob("*.json")):
        gold = read_json(gold_path)
        if not include_unverified and not gold.get("verified"):
            continue
        collected_path = COLLECTED_DIR / gold_path.name
        if collected_path.exists():
            records.append((gold, read_json(collected_path)))

    all_pairs, documented = [], 0
    time_rows, description_rows, topic_rows = [], [], []
    for gold, collected in records:
        pairs, n_doc, _ = _pairs(gold, collected)
        all_pairs += pairs
        documented += n_doc
        time_rows += time_agreement(gold, collected)

    result: dict[str, Any] = {
        "date": date.today().isoformat(), "judge": JUDGE_MODEL if llm else None, "records": len(records),
        "columns": {"documented": documented, "with_autoddg_text": len(all_pairs),
                    "coverage": round(len(all_pairs) / documented, 3) if documented else None},
        "time": {"files": len(time_rows), "overlap": sum(r["overlap"] for r in time_rows), "rows": time_rows},
    }
    if llm:
        graded = judge_columns(all_pairs)
        control = judge_columns(shuffled(all_pairs))
        for name, rows in (("column_grades", graded), ("negative_control", control)):
            counts = {g: sum(1 for r in rows if r["grade"] == g) for g in ("correct", "partial", "wrong")}
            result[name] = {"n": len(rows), **counts, "rows": rows}

        for gold, collected in records:
            curator = (gold.get("curator") or {})
            context = "\n".join(f"- {c['column']}: {c.get('definition', '')}"
                                for cols in (gold.get("dictionary") or {}).values() for c in cols[:60])
            for key, f in collected["files"].items():
                if not (f or {}).get("ufd") or not curator.get("description"):
                    continue
                columns = ", ".join(list((f.get("profiler_columns") or {}).keys())[:80])
                verdict = ask_json(DESCRIPTION_JUDGE.format(
                    curator=curator["description"][:6000], context=(context + "\nColumns: " + columns)[:8000],
                    file_key=key, generated=f["ufd"][:6000]))
                facts, claims = verdict.get("curator_facts") or [], verdict.get("generated_claims") or []
                description_rows.append({
                    "record": gold["record_id"], "file": key,
                    "fact_recall": round(sum(1 for x in facts if x.get("covered")) / len(facts), 3) if facts else None,
                    "unsupported_rate": round(sum(1 for x in claims if not x.get("supported")) / len(claims), 3) if claims else None,
                    "facts": facts, "claims": claims})
            topics = {f.get("topic") for f in collected["files"].values() if f and f.get("topic")}
            if curator.get("subjects") and topics:
                for topic in sorted(topics):
                    verdict = ask_json(TOPIC_JUDGE.format(subjects="; ".join(curator["subjects"]), topic=topic))
                    topic_rows.append({"record": gold["record_id"], "topic": topic, **verdict})
        recall = [r["fact_recall"] for r in description_rows if r["fact_recall"] is not None]
        unsupported = [r["unsupported_rate"] for r in description_rows if r["unsupported_rate"] is not None]
        result["descriptions"] = {"files": len(description_rows),
                                  "median_fact_recall": statistics.median(recall) if recall else None,
                                  "median_unsupported_rate": statistics.median(unsupported) if unsupported else None,
                                  "rows": description_rows}
        result["topics"] = {"n": len(topic_rows), "match": sum(1 for r in topic_rows if r.get("match")),
                            "rows": topic_rows}
    return result


def report(r: dict[str, Any]) -> str:
    lines = [f"# AutoDDG vs documentation and curator metadata ({r['date']})", "",
             f"{r['records']} UltraViolet records. Judge: {r['judge'] or 'none (--no-llm)'}. Gold not hand-verified.", "",
             f"- **Column coverage:** {r['columns']['with_autoddg_text']} of {r['columns']['documented']} documented columns "
             f"have an AutoDDG description ({r['columns']['coverage']:.0%}). AutoDDG describes at most the first "
             f"80 columns of a table, so wide surveys lose the rest.",
             f"- **Time range vs curator dates:** {r['time']['overlap']} of {r['time']['files']} files overlap."]
    if "column_grades" in r:
        g, c = r["column_grades"], r["negative_control"]
        lines += [f"- **Column meanings:** {g['correct']} correct, {g['partial']} partial, {g['wrong']} wrong of {g['n']}.",
                  f"- **Negative control (shuffled pairs):** {c['wrong']} of {c['n']} graded wrong "
                  f"({c['wrong'] / c['n']:.0%}; should be near 100%)." if c['n'] else "- Negative control: none",
                  f"- **Descriptions:** {r['descriptions']['files']} files; median recall of the curator's key facts "
                  f"{r['descriptions']['median_fact_recall']}, median share of unsupported claims "
                  f"{r['descriptions']['median_unsupported_rate']}.",
                  f"- **Topic vs subjects:** {r['topics']['match']} of {r['topics']['n']} AutoDDG topics match a curator subject."]
    return "\n".join(lines) + "\n"


def main(include_unverified: bool = True, llm: bool = True) -> None:
    result = compare(llm=llm, include_unverified=include_unverified)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"description-eval-{result['date']}{'' if llm else '-nollm'}"
    write_json(RESULTS_DIR / f"{stem}.json", result)
    (RESULTS_DIR / f"{stem}.md").write_text(report(result))
    print(report(result))
