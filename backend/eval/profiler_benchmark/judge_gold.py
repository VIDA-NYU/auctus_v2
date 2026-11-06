"""A second LLM checks the README gold, calibrated on the hand spot-check.

The gold types were extracted by one LLM (Gemini 2.5 Pro, description_eval gold stage)
and mapped to coarse classes by rules (common.coarse_from_gold). This asks a judge
from a different model family, for every scored column, whether the extraction copies
the documentation and whether the expected class is right. Agreement with the
50-column hand check says how far the judge can be trusted; the judge's verdicts on
all scored columns then estimate the gold's error rate at scale.

  python -m eval.profiler_benchmark.judge_gold <run dir> [--human answers.json] [--model ...]

Responses are cached (description_eval.common.ask_json), so reruns make no new calls.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from eval.description_eval.common import DATA_DIR, ask_json, read_json
from eval.profiler_benchmark.common import COARSE_CLASSES
from eval.profiler_benchmark.score import read_rows

JUDGE_MODEL = "@vertexai/anthropic.claude-sonnet-5-5"  # extraction was Gemini: a different family
BATCH = 25
EXCERPT_BEFORE, EXCERPT_AFTER = 300, 700

PROMPT = """You are checking a ground-truth set of column types for tabular datasets.

For each column below, an LLM read the dataset's documentation and copied the column's
definition, stated type and units/value labels ("extracted"). Rules then mapped that to
one coarse class ("assigned"). The classes are:
- numeric: quantities, counts, measurements, scores
- date: dates, times, years
- categorical: a fixed set of labels or codes (including coded answers like 1=Yes 2=No 3=Maybe)
- boolean: exactly two values meaning yes/no, true/false, present/absent (including 0/1 checkboxes)
- geo_coordinate: latitude / longitude
- place: names or codes of places (countries, states, cities, ZIP codes)
- identifier: IDs, keys, codes unique per row or entity (including ISBN, DOI)
- text: free text

For each column answer:
- "copied": "yes" if the extracted fields match what the documentation excerpt says, "no" if anything is wrong or invented.
- "assigned_ok": "yes" if the assigned class is the right one for this column given the documentation, "no" if not, "unclear" if the documentation can't tell.
- "correct_class": the right class (same as assigned when assigned_ok is yes).
- "reason": a few words.

Judge the column's real meaning, not only the type word the documentation uses: a
documented "number" that is an ISBN is an identifier; a 0/1 checkbox is boolean.

Columns:
{items}

Return JSON only: {{"columns": [{{"n": <n>, "copied": "...", "assigned_ok": "...", "correct_class": "...", "reason": "..."}}]}}
"""


def load_gold() -> dict[tuple[str, str, str], dict]:
    """(record, file, column) -> extracted entry, verified copies first (like run.rescore)."""
    from eval.profiler_benchmark.review_gold import EXTRACTED_DIR, VERIFIED_DIR

    gold = {}
    paths = sorted(EXTRACTED_DIR.glob("*.json")) + (sorted(VERIFIED_DIR.glob("*.json")) if VERIFIED_DIR.exists() else [])
    for path in paths:
        g = read_json(path)
        for file_key, columns in (g.get("dictionary") or {}).items():
            for target in ([f["file_key"] for f in g["files"]] if file_key == "*" else [file_key]):
                for col in columns:
                    gold[(g["record_id"], target, col["column"])] = col
    return gold


def excerpt(record: str, column: str) -> str:
    """The documentation around the column's first mention, from the `run scan` sources."""
    path = DATA_DIR / "sources" / "ultraviolet" / f"{record}.json"
    if not path.exists():
        return ""
    pattern = re.compile(rf"(?<![\w]){re.escape(column)}(?![\w])", re.IGNORECASE)
    for doc in sorted(read_json(path)["docs"], key=lambda d: -d["column_hits"]):
        match = pattern.search(doc["text"])
        if match:
            start = max(0, match.start() - EXCERPT_BEFORE)
            return f"[{doc['name']}] …{doc['text'][start: match.end() + EXCERPT_AFTER]}…"
    return ""


def judge(rows: list[dict], gold: dict, model: str) -> dict[int, dict]:
    items = []
    for n, row in enumerate(rows):
        entry = gold.get((row["record"], row["file"], row["column"]), {})
        items.append((row["record"], n, {
            "n": n, "file": row["file"], "column": row["column"], "assigned": row["expected"],
            "extracted": {k: entry.get(k, "") for k in ("definition", "type", "units")},
            "documentation_excerpt": excerpt(row["record"], row["column"]),
        }))
    verdicts: dict[int, dict] = {}
    by_record: dict[str, list[dict]] = {}
    for record, _, item in items:
        by_record.setdefault(record, []).append(item)
    for record, record_items in sorted(by_record.items()):
        for i in range(0, len(record_items), BATCH):
            batch = record_items[i: i + BATCH]
            reply = ask_json(PROMPT.format(items=json.dumps(batch, ensure_ascii=False, indent=1)), model=model)
            for v in reply.get("columns", []):
                if isinstance(v, dict) and isinstance(v.get("n"), int):
                    verdicts[v["n"]] = v
        print(f"judged {record}: {len(record_items)} columns")
    return verdicts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[c] * right[c] for c in set(left) | set(right)) / len(pairs) ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="benchmark run with types_readme.jsonl")
    parser.add_argument("--human", type=Path, help="hand spot-check: [{record, file, column, verdict Y/N, correct_type}]")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "results" / "gold-judge.json")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.run_dir / "types_readme.jsonl") if r.get("correct") is not None]
    verdicts = judge(rows, load_gold(), args.model)
    judged = []
    for n, row in enumerate(rows):
        v = verdicts.get(n, {})
        correct_class = v.get("correct_class") if v.get("correct_class") in COARSE_CLASSES else None
        judged.append({**{k: row[k] for k in ("record", "file", "column", "expected", "acceptable", "got", "correct")},
                       "copied": v.get("copied"), "assigned_ok": v.get("assigned_ok"),
                       "correct_class": correct_class, "reason": v.get("reason"),
                       # A wrong class only moves the score when the right one isn't accepted anyway.
                       "changes_score": (v.get("assigned_ok") == "no" and correct_class is not None
                                         and correct_class not in row["acceptable"].split("|"))})

    answered = [j for j in judged if j["assigned_ok"] in ("yes", "no")]
    ok = sum(j["assigned_ok"] == "yes" for j in answered)
    summary = {
        "model": args.model, "columns": len(judged), "judged": len(answered),
        "unclear": sum(j["assigned_ok"] == "unclear" for j in judged),
        "missing": sum(j["assigned_ok"] is None for j in judged),
        "assigned_ok": ok, "assigned_ok_rate": ok / len(answered) if answered else None,
        "assigned_ok_ci95": wilson(ok, len(answered)),
        "copied_no": sum(j["copied"] == "no" for j in judged),
        "changes_score": sum(j["changes_score"] for j in judged),
        "wrong_by_class": Counter(f"{j['expected']}->{j['correct_class']}" for j in judged if j["assigned_ok"] == "no"),
    }
    if args.human:
        index = {(j["record"], j["file"], j["column"]): j for j in judged}
        pairs = []
        for h in read_json(args.human):
            j = index.get((h["record"], h["file"], h["column"]))
            if j and h.get("verdict") in ("Y", "N") and j["assigned_ok"] in ("yes", "no"):
                pairs.append(("yes" if h["verdict"] == "Y" else "no", j["assigned_ok"]))
        summary["human"] = {
            "pairs": len(pairs), "agree": sum(a == b for a, b in pairs),
            "kappa": kappa(pairs),
            "human_no_judge_yes": sum(a == "no" and b == "yes" for a, b in pairs),
            "human_yes_judge_no": sum(a == "yes" and b == "no" for a, b in pairs),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "columns": judged}, indent=1, ensure_ascii=False, default=list))
    print(json.dumps(summary, indent=1, default=list))


if __name__ == "__main__":
    main()
