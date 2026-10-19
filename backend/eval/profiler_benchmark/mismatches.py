"""Classify README-vs-profiler type mismatches into failure modes, using the data.

For every scored README column where the profiler's type is outside the acceptable
set, look at the column's actual values and assign a failure mode, plus who it is
on: the profiler, the data/curation, the gold extraction, or genuinely ambiguous.

  python -m eval.profiler_benchmark.run mismatches --out eval/profiler_benchmark/results/<run>

Reads <run>/types_readme.jsonl (from `run types` or `run all`), the gold JSON and
the harvested files; writes <run>/mismatches.jsonl and <run>/mismatches.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
HARVEST_FILES = HERE.parents[3] / "data" / "ultraviolet" / "files"

MISSING_TOKENS = {"na", "n/a", "nan", "null", "none", ".", "-", "--", "?", "missing", "#n/a", "unknown", "nd", "-99", "-999", "999"}
NUMBER_RE = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")
FORMATTED_NUMBER_RE = re.compile(r"^[-+]?[$€£]?\s?\d{1,3}(,\d{3})+(\.\d+)?%?$|^[-+]?[$€£]\s?\d+(\.\d+)?$|^[-+]?\d+(\.\d+)?\s?%$")
YEAR_RE = re.compile(r"^(1[6-9]|20|21)\d\d$")
YESNO = {"yes", "no", "y", "n", "true", "false", "t", "f"}

# (failure mode, who it is on, one-line explanation)
MODES = {
    "not_in_file": ("gold", "README column name not found in the file (extraction error or README drift)"),
    "not_profiled": ("pipeline", "column exists but wasn't profiled (beyond the column cap)"),
    "coded_categories": ("ambiguous", "categories stored as numeric codes (1 = 'Not at all'); the README calls it categorical, the data is integers"),
    "place_false_positive": ("profiler", "geo-classifier labelled a non-place column (codes, flags, numbers) as a place"),
    "formatted_numbers": ("profiler", "numbers with $, % or thousands separators typed as text"),
    "missing_codes": ("data", "missing-value codes written as text (NA, '.', -99, 'NaN') mixed into a numeric or date column, so it gets a weaker type"),
    "dates_as_numbers": ("ambiguous", "dates stored as plain numbers (e.g. YYYYMMDD) or the README's 'date' is a number"),
    "numbers_as_dates": ("profiler", "numeric values typed as dates"),
    "text_vs_categorical": ("ambiguous", "free text vs categorical: a granularity call, not an error"),
    "id_detection": ("profiler", "identifier column not typed as an identifier"),
    "yes_no_flags": ("ambiguous", "a yes/no or 0/1 column read as categorical/numeric vs boolean"),
    "other": ("unclear", "doesn't fit a rule; read the values"),
}


def column_values(record_id: str, file_key: str, column: str) -> list[str] | None:
    path = HARVEST_FILES / record_id / file_key
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            frame = pd.read_excel(path, dtype=str)
        else:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python",
                                encoding_errors="replace", on_bad_lines="skip")
    except Exception:
        return None
    match = next((c for c in frame.columns if str(c).strip().lower() == str(column).strip().lower()), None)
    if match is None:
        return None
    return [str(v).strip() for v in frame[match].tolist()]


def value_profile(values: list[str]) -> dict:
    present = [v for v in values if v != ""]
    lower = [v.lower() for v in present]
    distinct = Counter(present)
    n = max(1, len(present))
    return {
        "rows": len(values), "empty_share": round(1 - len(present) / max(1, len(values)), 3),
        "distinct": len(distinct), "top": [v for v, _ in distinct.most_common(6)],
        "numeric_share": round(sum(bool(NUMBER_RE.match(v)) for v in present) / n, 3),
        "formatted_share": round(sum(bool(FORMATTED_NUMBER_RE.match(v)) for v in present) / n, 3),
        "missing_token_share": round(sum(v in MISSING_TOKENS for v in lower) / n, 3),
        "year_share": round(sum(bool(YEAR_RE.match(v)) for v in present) / n, 3),
        "yesno_share": round(sum(v in YESNO for v in lower) / n, 3),
        "small_int_codes": all(NUMBER_RE.match(v) and float(v).is_integer() and abs(float(v)) < 100 for v in distinct) and len(distinct) <= 12,
    }


def classify(row: dict, values: list[str] | None) -> str:
    expected, got = row["expected"], row["got"]
    if values is None:
        return "not_in_file"
    if got is None:
        return "not_profiled"
    v = value_profile(values)
    if got == "place":
        return "place_false_positive"
    if expected in ("categorical", "boolean") and v["small_int_codes"] and got in ("numeric", "date"):
        return "yes_no_flags" if v["distinct"] <= 2 else "coded_categories"
    if expected == "boolean" or v["yesno_share"] > 0.8:
        return "yes_no_flags"
    if expected == "numeric" and got in ("text", "categorical"):
        if v["formatted_share"] > 0.5:
            return "formatted_numbers"
        if v["numeric_share"] + v["missing_token_share"] > 0.8 or v["numeric_share"] > 0.6:
            return "missing_codes"
    if expected == "date" and got == "numeric":
        return "missing_codes" if v["year_share"] > 0.5 and v["missing_token_share"] > 0 else "dates_as_numbers"
    if expected == "numeric" and got == "date":
        return "numbers_as_dates"
    if {expected, got} == {"text", "categorical"}:
        return "text_vs_categorical"
    if expected == "identifier":
        return "id_detection"
    return "other"


def analyze(run_dir: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run_dir / "types_readme.jsonl").read_text().splitlines()]
    out = []
    for row in rows:
        if row.get("correct") is not False:
            continue
        values = column_values(row["record"], row["file"], row["column"])
        mode = classify(row, values)
        out.append({**row, "mode": mode, "on": MODES[mode][0],
                    "values": value_profile(values) if values is not None else None})
    # Columns the README names but that aren't in the file are unscored (got None); report them too.
    for row in rows:
        if row.get("correct") is None and row.get("expected") and row.get("got") is None:
            if column_values(row["record"], row["file"], row["column"]) is None:
                out.append({**row, "mode": "not_in_file", "on": "gold", "values": None})
    return out


def report(items: list[dict], run_dir: Path) -> str:
    scored = [json.loads(l) for l in (run_dir / "types_readme.jsonl").read_text().splitlines()]
    n_scored = sum(1 for r in scored if r.get("correct") is not None)
    n_wrong = sum(1 for r in scored if r.get("correct") is False)
    by_mode = Counter(i["mode"] for i in items)
    by_on = Counter(i["on"] for i in items if i["mode"] != "not_in_file")
    lines = ["# README type mismatches by failure mode", "",
             f"{n_wrong} of {n_scored} scored README columns disagree with the profiler; "
             f"{by_mode.get('not_in_file', 0)} more README columns don't exist in their file (unscored).", "",
             "| failure mode | columns | on | what it is |", "|---|---|---|---|"]
    for mode, count in by_mode.most_common():
        lines.append(f"| {mode} | {count} | {MODES[mode][0]} | {MODES[mode][1]} |")
    lines += ["", "Scored mismatches by attribution: " + ", ".join(f"{k} {v}" for k, v in by_on.most_common()), ""]
    groups = defaultdict(list)
    for i in items:
        groups[i["mode"]].append(i)
    for mode, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## {mode} ({len(group)})", "", "| dataset | file | column | README type | profiler | values seen |", "|---|---|---|---|---|---|"]
        for i in group[:12]:
            v = i["values"]
            seen = ", ".join(v["top"][:5]) if v else "-"
            lines.append(f"| {i['record']} | {i['file'][:40]} | {i['column']} | {(i.get('gold_type') or '')[:30]} | "
                         f"{i.get('got')} | {seen[:70]} |")
        if len(group) > 12:
            lines.append(f"| ... | {len(group) - 12} more | | | | |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(run_dir: Path) -> None:
    items = analyze(run_dir)
    (run_dir / "mismatches.jsonl").write_text("".join(json.dumps(i, default=str) + "\n" for i in items))
    (run_dir / "mismatches.md").write_text(report(items, run_dir))
    print((run_dir / "mismatches.md").read_text()[:3000])
