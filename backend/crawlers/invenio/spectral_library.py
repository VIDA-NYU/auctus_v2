"""Suggest materials for FTIR spectra by matching them to an open reference library.

The library is Open Specy's FTIR set (636 spectra, 247 materials: plastics, cellulose
derivatives, rubbers, fibres, resins; https://osf.io/x7dpz, Cowger et al. 2021,
doi:10.1021/acs.analchem.1c00123). Its data license on OSF reads "Other", so the
compact file built from it is not committed; build it once:

  python -m crawlers.invenio.spectral_library build ftir_library.rds ftir_metadata.rds

(the two files from https://osf.io/x7dpz; needs `pip install pyreadr`). Without the
file, spectra are indexed as before, with no suggestions.

Matching follows Open Specy's approach: both spectra on one wavenumber grid, first
derivative (removes baseline offsets), standardized, Pearson correlation; the CO2
doublet is left out. The scores are suggestions for a conservator, not identifications.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

LIBRARY_PATH = Path(os.getenv("SPECTRAL_LIBRARY", Path(__file__).resolve().parents[2] / "data" / "spectral"
                         / "openspecy_ftir.npz"))
GRID = np.arange(700.0, 3800.0, 4.0)
CO2 = (2280.0, 2400.0)
MIN_COVERAGE = 0.6  # share of the grid the spectrum must cover
CONFIDENT, POSSIBLE = 0.8, 0.6


def preprocess(xs, ys) -> np.ndarray | None:
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    order = np.argsort(x)
    v = np.interp(GRID, x[order], y[order], left=np.nan, right=np.nan)
    covered = ~np.isnan(v)
    if covered.sum() < MIN_COVERAGE * len(GRID):
        return None
    v[~covered] = np.nanmin(v)
    v = np.gradient(v)
    v = (v - v.mean()) / (v.std() or 1.0)
    v[(GRID > CO2[0]) & (GRID < CO2[1])] = 0.0
    return v


@lru_cache(maxsize=1)
def load_library(path: str = str(LIBRARY_PATH)) -> tuple[np.ndarray, list[str]] | None:
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["spectra"], [str(n) for n in data["names"]]


def suggest(xs, ys, top: int = 3, library: tuple[np.ndarray, list[str]] | None = None) -> list[tuple[str, float]]:
    """Up to `top` distinct materials with their correlation, best first ([] without a library)."""
    library = library or load_library()
    vector = preprocess(xs, ys) if library else None
    if vector is None:
        return []
    spectra, names = library
    scores = spectra @ vector / len(GRID)
    out, seen = [], set()
    for i in np.argsort(-scores):
        key = names[i].lower()
        if key not in seen:
            seen.add(key)
            out.append((names[i], round(float(scores[i]), 3)))
        if len(out) == top:
            break
    return out


def confidence(score: float) -> str:
    return "confident" if score >= CONFIDENT else "possible" if score >= POSSIBLE else "no confident match"


def build(library_rds: str, metadata_rds: str, out: str = str(LIBRARY_PATH)) -> dict[str, Any]:
    import pyreadr

    lib = next(iter(pyreadr.read_r(library_rds).values()))
    meta = next(iter(pyreadr.read_r(metadata_rds).values()))
    names = dict(zip(meta["sample_name"], meta["spectrum_identity"].astype(str).str.strip()))
    rows, labels = [], []
    for sample, group in lib.groupby("sample_name"):
        vector = preprocess(group["wavenumber"].values, group["intensity"].values)
        if vector is not None:
            rows.append(vector.astype(np.float32))
            labels.append(names.get(sample, str(sample)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spectra=np.vstack(rows), names=np.array(labels))
    return {"spectra": len(rows), "materials": len(set(labels)), "path": out}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        print(build(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
