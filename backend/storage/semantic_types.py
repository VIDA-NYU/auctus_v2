"""Semantic column types from an LLM that sees only column names and sample values.

atlas-profiler says what a column is stored as (Integer, Text, Enumeration); this
says what it is ("US ZIP code", "agreement scale 1-5", "participant ID"). One call
per table on the NYU gateway, from the same CSV sample AutoDDG reads.

On UltraViolet the labels matched the README's definition for 75% of 568 documented
columns, against 66% for AutoDDG's own column descriptions (2026-11-09, eval
description_eval.semtypes). Feeding them to AutoDDG didn't change its dataset
descriptions, so they go to search and to the per-column metadata instead:
  - columns[i]["semantic_type_label"]  shown per column, searchable
  - attribute_keywords                 search reads it (api/search.py)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

SEMANTIC_TYPES_MODEL = os.getenv("SEMANTIC_TYPES_MODEL", os.getenv("AUTODDG_MODEL", "@vertexai/gemini-2.5-flash"))
# Columns typed per call. Every column gets typed; wide tables take several calls. One call
# for 186-251 columns hit max_tokens (Gemini counts its thinking) on 4 of 8 wide tables and
# lost all their labels; 80 per call never did (2026-11-27).
COLUMNS_PER_CALL = 80
SAMPLE_VALUES = 8
MAX_LABEL_WORDS = 8

PROMPT = """Give each column of this table a semantic type: what the values ARE, in 2 to 6 words,
as specific as the evidence allows (e.g. "US ZIP code", "agreement scale 1-5", "ISO country code",
"participant ID", "survey completion date", "household size count", "free-text comment").
Use only the column name, the table's title and the sample values. If the values don't
say, give the most specific type you can defend, not a guess about the study.

Table: {title}
{context}Columns (name: sample of distinct values):
{columns}

Return JSON only: {{"<column name>": "<semantic type>", ...}}
"""


def column_listing(sample_csv: str, names: list[str] | None = None) -> str:
    frame = pd.read_csv(io.StringIO(sample_csv), dtype=str, keep_default_na=False)
    lines = []
    for name in (list(frame.columns) if names is None else names):
        values = [v for v in dict.fromkeys(frame[name].tolist()) if v.strip()][:SAMPLE_VALUES]
        lines.append(f"- {name}: {', '.join(v[:40] for v in values)}")
    return "\n".join(lines)


def parse_reply(text: str) -> dict[str, str]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    text = fenced.group(1) if fenced else text
    start = text.find("{")
    if start < 0:
        return {}
    reply = json.JSONDecoder(strict=False).raw_decode(text[start:])[0]
    return {str(k): " ".join(str(v).split()[:MAX_LABEL_WORDS])
            for k, v in reply.items() if isinstance(v, (str, int, float)) and str(v).strip()}


RETRY = """Your keys didn't match the table's columns. Use exactly these column names as keys, one per
column, typing the column itself (if the table is a data dictionary, its columns are things like
"variable name" or "value labels", not the variables it describes):
{names}
"""


def _norm(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def infer_semantic_types(client, sample_csv: str, title: str, model: str = SEMANTIC_TYPES_MODEL) -> dict[str, str]:
    """Labels keyed by column name, for every column: COLUMNS_PER_CALL columns per call.

    A batch that fails is skipped (its columns stay unlabelled), the others still count.
    In a wide table each call also gets the other column names, so it knows what table it
    is typing."""
    columns = list(pd.read_csv(io.StringIO(sample_csv), dtype=str, keep_default_na=False, nrows=0).columns)
    labels: dict[str, str] = {}
    for start in range(0, len(columns), COLUMNS_PER_CALL):
        batch = columns[start:start + COLUMNS_PER_CALL]
        others = [c for c in columns if c not in batch]
        context = (f"Other columns of the table (for context, don't type them): {', '.join(others)}\n"
                   if others else "")
        try:
            labels.update(_infer_batch(client, sample_csv, title, batch, context, model))
        except Exception as exc:
            if len(columns) <= COLUMNS_PER_CALL:
                raise
            LOGGER.warning("Semantic types failed for columns %d-%d of %r: %r", start + 1, start + len(batch), title, exc)
    return labels


def _infer_batch(client, sample_csv: str, title: str, columns: list[str], context: str, model: str) -> dict[str, str]:
    """One call; a reply whose keys mostly miss the columns (it typed the rows of a data
    dictionary instead) gets one retry with the names spelled out."""
    messages = [{"role": "user", "content": PROMPT.format(title=title, context=context,
                                                          columns=column_listing(sample_csv, columns))}]
    labels = _ask(client, model, messages)
    wanted = {_norm(c) for c in columns}
    if columns and sum(_norm(k) in wanted for k in labels) < len(columns) / 2:
        messages += [{"role": "assistant", "content": json.dumps(labels)},
                     {"role": "user", "content": RETRY.format(names=json.dumps(columns, ensure_ascii=False))}]
        labels = _ask(client, model, messages)
    return labels


def _ask(client, model: str, messages: list[dict[str, str]]) -> dict[str, str]:
    response = client.chat.completions.create(model=model, messages=messages,
                                              # Gemini 2.5 counts its thinking against max_tokens:
                                              # 4000 cut 49 of 591 replies off mid-JSON.
                                              temperature=0, max_tokens=16000)
    return parse_reply(response.choices[0].message.content or "")


def apply_semantic_types(record: dict[str, Any], labels: dict[str, str]) -> int:
    """Put the labels on the record's columns and into attribute_keywords; returns how many matched."""
    pm = record.get("profiler_metadata")
    if not isinstance(pm, dict) or not labels:
        return 0
    by_name = {_norm(k): v for k, v in labels.items()}
    matched = []
    for column in pm.get("columns") or []:
        if isinstance(column, dict):
            label = by_name.get(_norm(column.get("name", "")))
            if label:
                column["semantic_type_label"] = label
                matched.append(label)
    if matched:
        keywords = list(pm.get("attribute_keywords") or [])
        pm["attribute_keywords"] = keywords + [m for m in dict.fromkeys(matched) if m not in keywords]
    return len(matched)


def attach_semantic_types(record: dict[str, Any], client) -> dict[str, Any]:
    """Best effort: a failed call leaves the record as it was."""
    if client is None or not record.get("sample"):
        return record
    try:
        labels = infer_semantic_types(client, record["sample"], record.get("title", "") or "")
        LOGGER.info("Semantic types for %s: %d column(s)", record.get("id"), apply_semantic_types(record, labels))
    except Exception as exc:
        LOGGER.warning("Semantic types failed for %s: %r", record.get("id"), exc)
    return record
