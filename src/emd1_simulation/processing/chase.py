"""Read the Zhao et al. arsenic time course, antioxidant perturbations,
transcript chases and m6A microarray from the source workbook (PMID 38142659)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..paths import RAW_DATA_DIR

XLSX = RAW_DATA_DIR / "papers" / "Zhao-PMID-38142659.xlsx"

ANTIOX = ["SOD1", "SOD2", "TXN", "CAT", "GPX1", "PRDX5"]
STAGES = ["8w", "22w", "22+12w"]          # column blocks, left to right


def _grid(sheet_name: str) -> list[list]:
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    try:
        return [list(r) for r in wb[sheet_name].iter_rows(values_only=True)]
    finally:
        wb.close()


def _num(x):
    return float(x) if isinstance(x, (int, float)) else np.nan


def parse_fig3() -> pd.DataFrame:
    """Antioxidant expression across the transformation time course.

    Per gene: 0 uM vs 1 uM arsenite, 3 replicates each, at weeks 0/1/4/8/22
    and '+12' (12 further weeks of maintenance; 1 uM arm only).
    """
    g = _grid("Fig. 3 and Fig. S3")
    rows, gene = [], None
    for r in g:
        if not r:
            continue
        c0 = r[0]
        # Supplementary panels use the same columns for different assays.
        if isinstance(c0, str) and re.match(r"Fig\.\s*S", c0):
            break
        if isinstance(c0, str) and c0 in ANTIOX and any(
                isinstance(c, str) and "arsenite" in c for c in r[1:6]):
            gene = c0
            continue
        if gene and (isinstance(c0, (int, float)) or (isinstance(c0, str) and c0.strip() == "+12")):
            week = "+12" if isinstance(c0, str) else str(int(c0))
            for j, dose in enumerate(["0uM", "1uM"]):
                for k in range(3):
                    v = _num(r[1 + j * 3 + k]) if len(r) > 1 + j * 3 + k else np.nan
                    if np.isfinite(v):
                        rows.append(dict(gene=gene, week=week, dose=dose, rep=k + 1, value=v))
        elif isinstance(c0, str) and c0 not in ANTIOX:
            gene = None
    return pd.DataFrame(rows)


ARM_ORDER = ("siControl", "siMETTL3", "siFTO")


def _stage_header(g: list) -> dict:
    """Map each chase block to its exposure stage using the Fig. 5B headers."""
    for i, r in enumerate(g):
        if r and isinstance(r[0], str) and r[0].strip() == "Fig. 5B":
            for rr in g[i + 1:i + 4]:
                if not rr:
                    continue
                cols = {j: str(c).replace(" ", "")
                        for j, c in enumerate(rr)
                        if isinstance(c, str) and c.strip()}
                if set(cols.values()) == set(STAGES):
                    return cols
            break
    raise ValueError("Fig. 5B stage header row not found; sheet layout has changed")


def parse_fig5b() -> pd.DataFrame:
    """Read the Fig. 5B actinomycin chases at 0, 2, 4 and 6 hours.

    Each exposure stage (8w, 22w and 22+12w) includes siControl, siMETTL3
    and siFTO in triplicate. Fractions are normalised to the 0-hour value.
    """
    g = _grid("Fig. 5 and S5")
    stage_by_col = _stage_header(g)
    rows = []
    for i, r in enumerate(g):
        if not r or not isinstance(r[0], str) or r[0] not in ANTIOX:
            continue
        if not any(isinstance(c, str) and c == "siControl" for c in r):
            continue
        starts = [j for j, c in enumerate(r) if isinstance(c, str) and c == "siControl"]
        if len(starts) != 3:
            continue          # Fig. S5c reuses "siControl" but is not a chase
        gene = r[0]
        for s in starts:
            stage = stage_by_col.get(s - 1)
            if stage is None:
                raise ValueError(
                    f"{gene}: chase block at column {s} has no stage header at "
                    f"column {s - 1}; expected one of {STAGES}")
            for a, si in enumerate(ARM_ORDER):
                label = r[s + a * 3] if len(r) > s + a * 3 else None
                if label != si:
                    raise ValueError(
                        f"{gene} {stage}: expected '{si}' at column "
                        f"{s + a * 3}, found {label!r}. Arm order in the "
                        "workbook has changed; decay ratios would be wrong.")
            for rr in g[i + 1:i + 6]:
                if not rr or len(rr) <= s:
                    break
                hr = _num(rr[s - 1])
                if not np.isfinite(hr):
                    break
                for a, si in enumerate(ARM_ORDER):
                    for k in range(3):
                        col = s + a * 3 + k
                        v = _num(rr[col]) if len(rr) > col else np.nan
                        if np.isfinite(v):
                            rows.append(dict(gene=gene, stage=stage, sirna=si,
                                             hour=hr, rep=k + 1, frac=v))
    return pd.DataFrame(rows)


# Panels S5b and S5c report fold changes relative to each treatment’s own control.
PERTURBATIONS = {
    "STM2457": ("pharmacological", ["Control", "STM2457 (METTL3i)", "FB23-2 (FTOi)"]),
    "siMETTL3": ("genetic", ["siControl", "siMETTL3", "siFTO"]),
}


def parse_perturbation() -> pd.DataFrame:
    """Antioxidant expression under writer/eraser perturbation, per stage."""
    g = _grid("Fig. 5 and S5")
    rows = []
    for i, r in enumerate(g):
        if not r or not isinstance(r[0], str) or r[0] not in ANTIOX:
            continue
        marker = next((m for m in PERTURBATIONS if any(
            isinstance(c, str) and c == m for c in r)), None)
        if marker is None:
            continue
        kind, arms = PERTURBATIONS[marker]
        if len([j for j, c in enumerate(r) if isinstance(c, str) and c == "siControl"]) == 3:
            continue          # that is a Fig. 5B chase block, not a perturbation block
        gene = r[0]
        for rr in g[i + 1:i + 5]:
            if not rr or rr[0] is None:
                break
            stage = "+12" if isinstance(rr[0], str) else str(int(rr[0]))
            if stage not in ("8", "22", "+12"):
                break         # perturbation blocks are labelled by week, not hour
            for a, tr in enumerate(arms):
                for k in range(3):
                    v = _num(rr[1 + a * 3 + k]) if len(rr) > 1 + a * 3 + k else np.nan
                    if np.isfinite(v):
                        rows.append(dict(gene=gene, stage=stage, kind=kind,
                                         treatment=tr, rep=k + 1, fold=v))
    return pd.DataFrame(rows)


def parse_microarray() -> pd.Series:
    """Transcriptome-wide m6A microarray, log2 fold change, one value per gene."""
    g = _grid("Fig. S3b-S3d-m6A microarray")
    vals: dict[str, list[float]] = {}
    for r in g[1:]:
        if r and isinstance(r[0], str) and isinstance(r[1], (int, float)):
            vals.setdefault(r[0], []).append(float(r[1]))
    # A gene may have more than one probe.
    return pd.Series({k: float(np.mean(v)) for k, v in vals.items()}, name="log2FC_m6A")


def chase_decay_rates() -> pd.DataFrame:
    """Read the per-series decay constants from the bundled SQLite database."""
    from .store import read_table
    return read_table("chase_decay_constants")


def decay_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series first-order decay constant k (1/h) from ln(frac) vs hour."""
    rows = []
    for (gene, stage, si, rep), d in df.groupby(["gene", "stage", "sirna", "rep"]):
        d = d[d.frac > 0].sort_values("hour")
        if len(d) < 3:
            continue
        k, _ = np.polyfit(d.hour.values, np.log(d.frac.values), 1)
        rows.append(dict(gene=gene, stage=stage, sirna=si, rep=rep, k=-k,
                         half_life=np.log(2) / -k if k < 0 else np.inf))
    return pd.DataFrame(rows)
