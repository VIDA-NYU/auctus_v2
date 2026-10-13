"""Hand-verify the README data dictionaries used as type ground truth.

The dictionaries were extracted from UltraViolet READMEs by an LLM
(eval/description_eval gold stage), so each must be checked by a person before
the README type score is reported. Only column names, types and units matter
for the type benchmark; definitions don't need checking.

  python -m eval.profiler_benchmark.review_gold list
  python -m eval.profiler_benchmark.review_gold sheet <record_id>    # writes a review page
  python -m eval.profiler_benchmark.review_gold verify <record_id>   # after fixing the JSON

Workflow per record:
1. `sheet` writes gold/review/<record>.md: the extracted table (columns that don't
   exist in the actual file are flagged) followed by the README.
2. Open it next to the JSON it names. Fix wrong names/types/units, delete invented
   rows, add missed ones (same keys: column, definition, type, units).
3. `verify` marks it verified and copies it to gold/ultraviolet/, which the
   benchmark reads first. Re-run `sheet` any time to re-check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXTRACTED_DIR = HERE.parent / "description_eval" / "data" / "gold"
VERIFIED_DIR = HERE / "gold" / "ultraviolet"
REVIEW_DIR = HERE / "gold" / "review"
HARVEST_DIR = HERE.parents[3] / "data" / "ultraviolet"


def gold_path(record_id: str) -> Path:
    """The file to edit: the verified copy if one exists, else the extraction."""
    verified = VERIFIED_DIR / f"{record_id}.json"
    return verified if verified.exists() else EXTRACTED_DIR / f"{record_id}.json"


def file_headers(record_id: str, file_key: str) -> list[str] | None:
    path = HARVEST_DIR / "files" / record_id / file_key
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return [str(c) for c in pd.read_excel(path, nrows=0).columns]
        return [str(c) for c in pd.read_csv(path, nrows=0, sep=None, engine="python", encoding_errors="replace").columns]
    except Exception:
        return None


def readme_text(record_id: str, readme_file: str | None) -> str:
    if not readme_file:
        return "(no README)"
    path = HARVEST_DIR / "files" / record_id / readme_file
    return path.read_text(errors="replace") if path.exists() else f"(README not downloaded: {readme_file})"


def cmd_list() -> None:
    print(f"{'record':<14}{'columns':>8}  {'verified':<9} readme")
    for path in sorted(EXTRACTED_DIR.glob("*.json")):
        gold = json.loads(gold_path(path.stem).read_text())
        columns = sum(len(v) for v in gold["dictionary"].values())
        if columns:
            print(f"{path.stem:<14}{columns:>8}  {str(gold.get('verified', False)):<9} {gold.get('readme_file')}")


def cmd_sheet(record_id: str) -> None:
    source = gold_path(record_id)
    gold = json.loads(source.read_text())
    files = [f["file_key"] for f in gold["files"]]
    lines = [f"# Review: {record_id}", "",
             f"Edit: `{source}`", f"Then: `python -m eval.profiler_benchmark.review_gold verify {record_id}`", "",
             "Check each row's **column**, **type** and **units** against the README below. "
             "⚠ marks a column name that isn't in the file's header (likely an extraction error, or README drift).", ""]
    for file_key, columns in gold["dictionary"].items():
        targets = files if file_key == "*" else [file_key]
        headers = {h.strip().lower() for t in targets for h in (file_headers(record_id, t) or [])}
        lines += [f"## {file_key}" + ("  (applies to every file)" if file_key == "*" else ""), "",
                  "| # | column | type | units | in file? |", "|---|---|---|---|---|"]
        for i, col in enumerate(columns, 1):
            found = "?" if not headers else ("yes" if str(col["column"]).strip().lower() in headers else "⚠ no")
            cell = lambda v: str(v or "").replace("|", "/").replace("\n", " ")[:60]
            lines.append(f"| {i} | {cell(col['column'])} | {cell(col.get('type'))} | {cell(col.get('units'))} | {found} |")
        lines.append("")
    if gold.get("extraction_notes"):
        lines += [f"Extraction notes: {gold['extraction_notes']}", ""]
    lines += ["---", "", f"## README ({gold.get('readme_file')})", "", "```", readme_text(record_id, gold.get("readme_file")), "```"]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{record_id}.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


def cmd_verify(record_id: str) -> None:
    source = gold_path(record_id)
    gold = json.loads(source.read_text())
    gold["verified"] = True
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIED_DIR / f"{record_id}.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n")
    print(f"verified {record_id} -> {VERIFIED_DIR / f'{record_id}.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "sheet", "verify"])
    parser.add_argument("record_id", nargs="?")
    args = parser.parse_args()
    if args.command != "list" and not args.record_id:
        sys.exit("record_id required")
    {"list": lambda: cmd_list(), "sheet": lambda: cmd_sheet(args.record_id),
     "verify": lambda: cmd_verify(args.record_id)}[args.command]()


if __name__ == "__main__":
    main()
