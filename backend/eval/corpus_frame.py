"""The round's corpus id list — one reader, shared by every tool that needs
to know what "the corpus" is.

Extracted for the same reason ``field_scores`` and ``classify_query`` were:
two modules answering the same question from two independent reads drift, and
this particular question already drifted once. The live index held 144
documents against a 100-dataset corpus (44 stale from an earlier round) and
no retrieval or judging code noticed, because every path asked the index how
many documents it had instead of asking the round what its corpus was
(plan-drift-audit.md finding 1). A document count cannot distinguish "the
corpus" from "whatever is currently indexed"; the recorded id list can.

The frame file is the sampling frame's own output, tracked in git, and names
the ids the round drew — so reading it is also what makes a run reproducible
after the index moves on.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CORPUS_FRAME = Path("eval/frame/post_spatial_swap_corpus_2026-08-17.json")


def load_corpus_ids(corpus_path: Path = DEFAULT_CORPUS_FRAME) -> list[str]:
    """Return the round's corpus ids, sorted.

    Sorted rather than file-order so every caller sees the same sequence:
    ``judge_qrels`` slices this into fixed chunk membership, and a chunk
    assignment that depended on how the frame file happened to be written
    would be reproducible only by accident.
    """
    doc = json.loads(corpus_path.read_text(encoding="utf-8"))
    return sorted(doc["corpus_ids"])
