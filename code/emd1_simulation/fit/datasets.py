"""Loaders and module definitions for the two GEO series.

Both series ship NCBI-pipeline gene-level counts covering IP *and* input
samples, on the same GRCh38.p13 annotation, so they are directly joinable and
no raw/SRA data is needed at this resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
ANNOT = DATA / "GSE144620" / "Human.GRCh38.p13.annot.tsv.gz"

# --- sample maps ---------------------------------------------------------
# Order is the column order of the counts matrices, verified against
# !Sample_geo_accession / !Sample_title in each series matrix.
GSE145923_SAMPLES = [  # HaCaT keratinocytes
    ("GSM4339741", "As",  "input", 1), ("GSM4339742", "As",  "input", 2),
    ("GSM4339743", "AsT", "input", 1), ("GSM4339744", "AsT", "input", 2),
    ("GSM4339745", "Ctl", "input", 1), ("GSM4339746", "Ctl", "input", 2),
    ("GSM4339747", "As",  "IP", 1),    ("GSM4339748", "As",  "IP", 2),
    ("GSM4339749", "AsT", "IP", 1),    ("GSM4339750", "AsT", "IP", 2),
    ("GSM4339751", "Ctl", "IP", 1),    ("GSM4339752", "Ctl", "IP", 2),
]

_G620 = []
for i, cond in enumerate(["KO", "mut", "vec"]):          # ALKBH5 KO / overexpr / vector
    for j, ros in enumerate(["minus", "plus"]):          # -/+ ROS
        for k, kind in enumerate(["input", "IP"]):
            for rep in (1, 2):
                acc = "GSM%d" % (4292376 + i * 8 + j * 4 + k * 2 + rep - 1)
                _G620.append((acc, f"{cond}_{ros}", kind, rep))
GSE144620_SAMPLES = _G620

# GSE145924 -- METTL14 / DDB2 / global genome repair, same lab and same HaCaT
# line as GSE145923 (the accessions are consecutive with it). This is the
# EMD1 -> KCC3 edge measured directly: METTL14 writes m6A on DDB2, YTHDF1 reads
# it, DDB2 protein drives global genome repair. PMID 34452996.
GSE145924_SAMPLES = [
    ("GSM4339753", "shNC",  "input", 1), ("GSM4339754", "shNC",  "input", 2),
    ("GSM4339755", "shM14", "input", 1), ("GSM4339756", "shM14", "input", 2),
    ("GSM4339757", "shNC",  "IP", 1),    ("GSM4339758", "shNC",  "IP", 2),
    ("GSM4339759", "shM14", "IP", 1),    ("GSM4339760", "shM14", "IP", 2),
    ("GSM4773964", "UVB",   "input", 1), ("GSM4773965", "UVB",   "input", 2),
    ("GSM4773966", "UVB",   "IP", 1),    ("GSM4773967", "UVB",   "IP", 2),
]

# --- transcript / module definitions -------------------------------------
# Single transcripts the model names explicitly.
SINGLE = {"M_a3b": "APOBEC3B", "M_ned": "NEDD4L"}

# The antioxidant genes named in the arsenic redox-homeostasis work.
ANTIOX_CORE = ["SOD1", "SOD2", "CAT", "TXN", "GPX1", "PRDX5"]

# GO-derived sets, as a check that the core lists are not cherry-picked.
GO_SETS = {
    "repair_GO": ("GOProcess", r"DNA repair"),
    "antiox_GO": ("GOProcess", r"response to oxidative stress"),
}


@dataclass
class Series:
    name: str
    counts: pd.DataFrame          # gene symbol x sample label
    meta: pd.DataFrame            # sample label -> condition, kind, rep
    cell: str


def _annotation() -> pd.DataFrame:
    return pd.read_csv(ANNOT, sep="\t",
                       usecols=["GeneID", "Symbol", "GOProcess", "GOFunction"])


def repair_gene_symbols() -> set[str]:
    """Use the annotation snapshot or its bundled, annotation-only gene list."""
    if ANNOT.exists():
        ann = _annotation()
        return set(ann.loc[ann["GOProcess"].fillna("").str.contains("DNA repair", case=False),
                           "Symbol"].dropna())
    path = Path(__file__).resolve().parents[3] / "derived/repair_gene_symbols.csv"
    return set(pd.read_csv(path)["Symbol"].dropna())


def load_series(gse: str, samples: list, cell: str) -> Series:
    raw = pd.read_csv(DATA / gse / f"{gse}_raw_counts_GRCh38.p13_NCBI.tsv.gz",
                      sep="\t", index_col=0)
    ann = _annotation().set_index("GeneID")
    labels = [f"{c}_{k}{r}" for _, c, k, r in samples]
    accs = [a for a, *_ in samples]
    if list(raw.columns) != accs:
        raise ValueError(f"{gse}: column order does not match the sample map")
    raw.columns = labels
    sym = ann["Symbol"].reindex(raw.index)
    counts = raw.groupby(sym.values).sum()
    meta = pd.DataFrame([(l, c, k, r) for l, (_, c, k, r) in zip(labels, samples)],
                        columns=["label", "condition", "kind", "rep"]).set_index("label")
    return Series(gse, counts, meta, cell)


def module_genes(expressed: pd.Index) -> dict[str, list[str]]:
    """Module gene sets, restricted to genes actually expressed in the data."""
    ann = _annotation()
    out: dict[str, list[str]] = {k: [v] for k, v in SINGLE.items()}
    out["M_aox"] = [g for g in ANTIOX_CORE if g in expressed]
    for name, (col, pat) in GO_SETS.items():
        hit = ann[col].fillna("").str.contains(pat, case=False, regex=True)
        gs = sorted(set(ann.loc[hit, "Symbol"]) & set(expressed))
        out[name] = gs
    out["M_rep"] = out["repair_GO"]
    return out


def cpm(counts: pd.DataFrame) -> pd.DataFrame:
    return counts / counts.sum() * 1e6


def enrichment(series: Series, min_cpm: float = 5.0, centre: bool = True) -> pd.DataFrame:
    """Per-replicate log2(IP/input), one column per condition-replicate.

    Median-centred within each replicate by default. Centring asks "does this
    transcript gain m6A *relative to the transcriptome*", which is the quantity
    the model's site-specific-vs-global contrast needs, and it removes
    replicate-to-replicate differences in IP efficiency. Set centre=False to
    look at global shifts -- but then IP efficiency is a live confounder.
    """
    c = cpm(series.counts)
    m = series.meta
    cols = {}
    for cond in m["condition"].unique():
        for rep in sorted(m["rep"].unique()):
            ip = m[(m.condition == cond) & (m.kind == "IP") & (m.rep == rep)].index
            inp = m[(m.condition == cond) & (m.kind == "input") & (m.rep == rep)].index
            if len(ip) != 1 or len(inp) != 1:
                continue
            i, n = c[ip[0]], c[inp[0]]
            keep = n > min_cpm
            e = pd.Series(np.nan, index=c.index)
            e[keep] = np.log2((i[keep] + 0.1) / (n[keep] + 0.1))
            if centre:
                e -= e.median()
            cols[f"{cond}_r{rep}"] = e
    return pd.DataFrame(cols)


def expression(series: Series, min_cpm: float = 5.0) -> pd.DataFrame:
    """Per-replicate log2 CPM from the input samples only."""
    c = cpm(series.counts)
    m = series.meta
    cols = {}
    for cond in m["condition"].unique():
        for rep in sorted(m["rep"].unique()):
            inp = m[(m.condition == cond) & (m.kind == "input") & (m.rep == rep)].index
            if len(inp) != 1:
                continue
            n = c[inp[0]]
            cols[f"{cond}_r{rep}"] = np.log2(n + 0.1).where(n > min_cpm)
    return pd.DataFrame(cols)
