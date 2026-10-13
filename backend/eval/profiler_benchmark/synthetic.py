"""Synthetic test tables with known column types (ground truth by construction).

Each generated column has an expected coarse class and, where the answer is
genuinely ambiguous (e.g. a 0/1 column), a set of acceptable classes. A second
"format" suite writes one table in several file encodings/layouts to test reading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 400


def _columns(rng: np.random.Generator, n: int) -> dict[str, tuple[list, str, set[str]]]:
    """name -> (values, expected class, acceptable classes)."""
    days = pd.date_range("2015-01-01", periods=n, freq="D")
    words = ["harbor", "transit", "survey", "river", "zoning", "school", "permit", "station", "tree", "noise"]
    with_gaps = rng.normal(100, 15, n).round(1).astype(object)
    with_gaps[rng.random(n) < 0.1] = ""
    return {
        "int_count": (rng.integers(0, 500, n).tolist(), "numeric", {"numeric"}),
        "float_measure": (rng.normal(20, 5, n).round(3).tolist(), "numeric", {"numeric"}),
        "float_missing": (with_gaps.tolist(), "numeric", {"numeric"}),
        "thousands_sep": ([f"{x:,}" for x in rng.integers(1000, 900000, n)], "numeric", {"numeric"}),
        "percent": ([f"{x:.1f}%" for x in rng.uniform(0, 100, n)], "numeric", {"numeric"}),
        "currency": ([f"${x:,.2f}" for x in rng.uniform(1, 5000, n)], "numeric", {"numeric"}),
        "date_iso": (days.strftime("%Y-%m-%d").tolist(), "date", {"date"}),
        "date_us": (days.strftime("%m/%d/%Y").tolist(), "date", {"date"}),
        "date_eu": (days.strftime("%d/%m/%Y").tolist(), "date", {"date"}),
        "datetime_iso": ((days + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")).strftime("%Y-%m-%dT%H:%M:%S").tolist(), "date", {"date"}),
        "year": (rng.integers(1950, 2025, n).tolist(), "date", {"date", "numeric"}),
        "latitude": (rng.uniform(40.5, 40.9, n).round(6).tolist(), "geo_coordinate", {"geo_coordinate"}),
        "longitude": (rng.uniform(-74.25, -73.7, n).round(6).tolist(), "geo_coordinate", {"geo_coordinate"}),
        "zip_code": ([f"{x:05d}" for x in rng.integers(501, 99950, n)], "place", {"place", "identifier"}),
        "us_state": (rng.choice(["New York", "Texas", "Ohio", "California", "Florida", "Maine"], n).tolist(), "place", {"place"}),
        "country": (rng.choice(["France", "Brazil", "Japan", "Kenya", "Canada", "India", "Peru"], n).tolist(), "place", {"place"}),
        "city": (rng.choice(["Chicago", "Houston", "Boston", "Denver", "Seattle", "Atlanta"], n).tolist(), "place", {"place"}),
        "bool_tf": (rng.choice(["true", "false"], n).tolist(), "boolean", {"boolean"}),
        "bool_yn": (rng.choice(["Y", "N"], n).tolist(), "boolean", {"boolean", "categorical"}),
        "bool_01": (rng.integers(0, 2, n).tolist(), "boolean", {"boolean", "categorical", "numeric"}),
        "category_small": (rng.choice(["low", "medium", "high"], n).tolist(), "categorical", {"categorical"}),
        "category_codes": (rng.choice(["A1", "B2", "C3", "D4"], n).tolist(), "categorical", {"categorical"}),
        "free_text": ([" ".join(rng.choice(words, 9)) for _ in range(n)], "text", {"text"}),
        "uuid": ([f"{rng.integers(0, 2**32):08x}-{rng.integers(0, 2**16):04x}-{rng.integers(0, 2**16):04x}" for _ in range(n)], "identifier", {"identifier"}),
        "record_code": ([f"R{i:06d}X" for i in range(n)], "identifier", {"identifier"}),
        "row_number": (list(range(1, n + 1)), "identifier", {"identifier", "numeric"}),
    }


def write_suite(out_dir: Path, seed: int = 0) -> list[dict]:
    """Write the type suite and the format suite; return the gold manifest."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = _columns(rng, ROWS)
    frame = pd.DataFrame({name: values for name, (values, _, _) in columns.items()})
    gold = {name: {"expected": exp, "acceptable": sorted(acc)} for name, (_, exp, acc) in columns.items()}

    cases = []
    def add(name: str, write, note: str) -> None:
        path = out_dir / name
        write(path)
        cases.append({"case": name, "path": str(path), "note": note, "gold": gold})

    add("types.csv", lambda p: frame.to_csv(p, index=False), "baseline: comma, UTF-8")
    add("types_semicolon.csv", lambda p: frame.to_csv(p, index=False, sep=";"), "semicolon delimiter")
    add("types.tsv", lambda p: frame.to_csv(p, index=False, sep="\t"), "tab delimiter")
    add("types_cp1252.csv", lambda p: frame.assign(free_text=frame["free_text"] + " café").to_csv(p, index=False, encoding="cp1252"), "Windows-1252 encoding")
    add("types_bom.csv", lambda p: frame.to_csv(p, index=False, encoding="utf-8-sig"), "UTF-8 with BOM")
    add("types.xlsx", lambda p: frame.to_excel(p, index=False), "Excel XLSX")
    add("types_preamble.csv",
        lambda p: p.write_text("Survey of things, 2015-2016\nSource: synthetic\n\n" + frame.to_csv(index=False)),
        "title lines above the header (known gap: header assumed on row 1)")
    (out_dir / "manifest.json").write_text(json.dumps(cases, indent=1))
    return cases


def performance_frame(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    """A table of the given shape cycling through the type suite's columns."""
    rng = np.random.default_rng(seed)
    base = _columns(rng, rows)
    names = list(base)
    return pd.DataFrame({f"{names[i % len(names)]}_{i}": base[names[i % len(names)]][0] for i in range(cols)})
