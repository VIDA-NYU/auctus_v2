"""Standard identifier columns (ISBN, ISSN, DOI, ORCID), recognized by their check digits.

atlas-profiler types these as generic identifiers or text. Their schemes have check
digits or a fixed syntax, so a lookup is exact where the classifier can only guess:
a column whose values mostly validate gets the scheme recorded, and becomes
searchable by it ("ISBN").
"""

from __future__ import annotations

import re
from typing import Any

IDENTIFIER = "http://schema.org/identifier"
# Share of a column's distinct non-empty values that must validate.
MIN_VALID_SHARE = 0.8
MIN_VALUES = 3
MAX_CHECKED = 2000


def _isbn(value: str) -> bool:
    digits = re.sub(r"[\s-]", "", value.upper().removeprefix("ISBN").strip(": "))
    if re.fullmatch(r"\d{9}[\dX]", digits):
        total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(digits))
        return total % 11 == 0
    if re.fullmatch(r"97[89]\d{10}", digits):
        total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(digits))
        return total % 10 == 0
    return False


def _issn(value: str) -> bool:
    digits = re.sub(r"[\s-]", "", value.upper().removeprefix("ISSN").strip(": "))
    if not re.fullmatch(r"\d{7}[\dX]", digits):
        return False
    total = sum((8 - i) * int(c) for i, c in enumerate(digits[:7]))
    check = (11 - total % 11) % 11
    return digits[7] == ("X" if check == 10 else str(check))


def _doi(value: str) -> bool:
    return bool(re.fullmatch(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.\d{4,9}/\S+", value.strip(), re.IGNORECASE))


def _orcid(value: str) -> bool:
    match = re.fullmatch(r"(?:https?://orcid\.org/)?(\d{4})-(\d{4})-(\d{4})-(\d{3}[\dX])", value.strip().upper())
    if not match:
        return False
    digits = "".join(match.groups())
    total = 0
    for c in digits[:15]:
        total = (total + int(c)) * 2
    check = (12 - total % 11) % 11
    return digits[15] == ("X" if check == 10 else str(check))


# ISSN before ISBN: an 8-character ISSN never passes the ISBN check, and order only
# matters for the label when both could apply.
SCHEMES = (("ORCID", _orcid), ("DOI", _doi), ("ISSN", _issn), ("ISBN", _isbn))


def identifier_scheme(values: list[str]) -> str | None:
    """The scheme most of the distinct values validate against, or None.

    Placeholders such as "No ISBN" don't count against the column: values without a
    single digit are ignored.
    """
    distinct = sorted({v.strip() for v in values if v and any(ch.isdigit() for ch in v)})[:MAX_CHECKED]
    if len(distinct) < MIN_VALUES:
        return None
    for scheme, valid in SCHEMES:
        if sum(1 for v in distinct if valid(v)) >= MIN_VALID_SHARE * len(distinct):
            return scheme
    return None


def mark_identifier_columns(profiler_output: dict[str, Any], rows: list[dict[str, str]]) -> int:
    """Record the scheme of standard identifier columns (from the first rows); returns how many."""
    marked = 0
    for column in profiler_output.get("columns") or []:
        name = column.get("name")
        scheme = identifier_scheme([str(r.get(name) or "") for r in rows])
        if not scheme:
            continue
        column["identifier_scheme"] = scheme
        semantic = list(column.get("semantic_types") or [])
        if IDENTIFIER not in semantic:
            semantic.append(IDENTIFIER)
        column["semantic_types"] = semantic
        marked += 1
    return marked
