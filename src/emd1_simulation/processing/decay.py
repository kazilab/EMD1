"""Load original chase observations for decay-model fitting."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pandas as pd

from ..paths import RAW_DATA_DIR
from .chase import parse_fig5b

NEDD4L_WORKBOOK = RAW_DATA_DIR / "papers" / "NatCommun2021_NEDD4L_SourceData.xlsx"


def load_antioxidant_decay() -> pd.DataFrame:
    """Read the Zhao chase after the fixed, normalised 0-hour observation."""
    d = parse_fig5b()
    return d[d.hour > 0].reset_index(drop=True)


def load_nedd4l_decay(workbook: str | Path | None = None) -> pd.DataFrame:
    """Read NEDD4L chases from panels 3i, 3m and 5a of the Cui workbook.

    These panels retain observed variation at 0 hours.
    """
    import openpyxl
    p = Path(workbook) if workbook is not None else NEDD4L_WORKBOOK
    with closing(openpyxl.load_workbook(p, data_only=True)) as wb:
        rows = []
        for sheet, panel, arms in [
                ("Fig.3", "3i", ["Control", "As-T"]),
                ("Fig.3", "3m", ["WT", "FTO KO"]),
                ("Fig.5", "5a", ["WT+shNC", "FTO KO+shNC", "WT+shMETTL14", "FTO KO+shMETTL14"])]:
            g = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
            start = next(i for i, r in enumerate(g)
                         if any(isinstance(c, str) and c == panel for c in r))
            hour, rep = None, 0
            for r in g[start + 1:]:
                if not r or all(c is None for c in r):
                    break
                lab = next((c for c in r[:4] if isinstance(c, str) and c.endswith("h")), None)
                if lab is not None:
                    hour, rep = float(lab[:-1]), 0
                if hour is None:
                    continue
                vals = [c for c in r if isinstance(c, (int, float))]
                if len(vals) < len(arms):
                    continue
                rep += 1
                for a, v in zip(arms, vals[-len(arms):]):
                    rows.append(dict(panel=panel, arm=a, hour=hour, rep=rep, frac=float(v)))
    return pd.DataFrame(rows)

