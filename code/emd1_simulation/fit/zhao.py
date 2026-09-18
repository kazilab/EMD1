"""Parse the arsenic redox-homeostasis source data (PMID 38142659).

This workbook is the most model-relevant of the three: it carries a genuine
chronic-arsenic time course (0/1/4/8/22 weeks, plus 12 weeks of maintenance),
the antioxidant module measured gene by gene across it, actinomycin-chase decay
series under siMETTL3 / siFTO at three stages of transformation, METTL3- and
FTO-inhibitor arms, and a transcriptome-wide m6A microarray.

The sheets are hand-laid-out, so blocks are located by their header strings
rather than by fixed offsets.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

XLSX = (Path(__file__).resolve().parent.parent / "data" / "papers"
        / "Zhao-PMID-38142659.xlsx")

ANTIOX = ["SOD1", "SOD2", "TXN", "CAT", "GPX1", "PRDX5"]
STAGES = ["8w", "22w", "22+12w"]          # column blocks, left to right


def _grid(sheet_name: str) -> list[list]:
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    return [list(r) for r in wb[sheet_name].iter_rows(values_only=True)]


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
        # The sheet holds Fig. 3 (absolute quantification) followed by Fig. S3l
        # (log2 ratios, so negative) and Fig. S3m (ELISA standard curves) in the
        # same column layout. Only the first section is this readout.
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
    """Column -> stage, read from the row under the 'Fig. 5B' marker.

    The stage row sits above the per-gene header rows and labels each block's
    left edge ('8 w', '22 w', '22+12 w' at the columns that hold the chase
    hours). Reading it makes the left-to-right stage order a verified fact
    rather than a positional assumption.
    """
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
    """Actinomycin-chase decay: remaining fraction at 0/2/4/6 h.

    Three column blocks (8w, 22w, 22+12w); within each, siControl / siMETTL3 /
    siFTO in triplicate. Values are normalised to t = 0.

    Every positional assumption here is checked against the sheet's own labels
    rather than assumed. The block order used to be taken on trust: the parser
    located "siControl" and then read the next two triplets as siMETTL3 and
    siFTO without looking at their headers, and mapped blocks to stages purely
    left-to-right. Both happen to be correct in this workbook, but a silent
    swap of the two knockdown arms would have inverted the central antioxidant
    result while every downstream number still looked plausible. It now raises.
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
            # The stage comes from the sheet's own header, not from block order.
            stage = stage_by_col.get(s - 1)
            if stage is None:
                raise ValueError(
                    f"{gene}: chase block at column {s} has no stage header at "
                    f"column {s - 1}; expected one of {STAGES}")
            # The arm of each triplet comes from its own label.
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


# Two independent handles on the same edge: a pharmacological pair (Fig. S5b)
# and a genetic pair (Fig. S5c). Both are fold change vs their own control,
# measured at each stage of transformation.
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
    # several probes per symbol; average rather than let the last one win
    return pd.Series({k: float(np.mean(v)) for k, v in vals.items()}, name="log2FC_m6A")


DERIVED_DECAY_CSV = "chase_decay_constants.csv"


def _derived_decay_candidates() -> list[Path]:
    """Where a shipped copy of the derived decay constants may live.

    Working repository first, then the public-archive layout where `derived/`
    sits beside `code/`.
    """
    here = Path(__file__).resolve()
    return [here.parents[1] / "data" / "derived" / DERIVED_DECAY_CSV,
            here.parents[3] / "derived" / DERIVED_DECAY_CSV]


def chase_decay_rates() -> pd.DataFrame:
    """Per-series chase decay constants, from the workbook or the shipped CSV.

    The source workbook is closed-access and is not redistributed (see
    data/THIRD_PARTY.md), but the constants derived from it are. Anything that
    needs only the rates therefore keeps working without it, which is what lets
    the audit and the full simulation run from the public archive alone.
    """
    if XLSX.exists():
        return decay_rates(parse_fig5b())
    for candidate in _derived_decay_candidates():
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        f"neither {XLSX} nor a derived {DERIVED_DECAY_CSV} was found; "
        f"looked in {', '.join(str(c) for c in _derived_decay_candidates())}")


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


# ---------------------------------------------------------------------------
# Analysis: what these data say about the model's antioxidant / feedback arm
# ---------------------------------------------------------------------------

def analyse() -> dict[str, pd.DataFrame]:
    from scipy.stats import ttest_ind

    out = {}

    # --- 1. does m6A stabilise the antioxidant transcripts? ---------------
    # The model couples M_aox to antioxidant capacity through a YTHDF1
    # *translation* gain and gives these transcripts a fixed decay rate. If the
    # chase shows decay changing with writer/eraser status, that coupling is
    # in the wrong place.
    k = decay_rates(parse_fig5b())
    rows = []
    for (gene, stage), d in k.groupby(["gene", "stage"]):
        ctl = d[d.sirna == "siControl"].k
        for arm in ["siMETTL3", "siFTO"]:
            a = d[d.sirna == arm].k
            if len(ctl) < 2 or len(a) < 2:
                continue
            rows.append(dict(gene=gene, stage=stage, arm=arm,
                             k_ctl=ctl.mean(), k_arm=a.mean(),
                             ratio=a.mean() / ctl.mean(),
                             p=ttest_ind(a, ctl, equal_var=False).pvalue))
    out["decay"] = pd.DataFrame(rows)

    # --- 2. writer/eraser perturbation vs antioxidant level ---------------
    p = parse_perturbation()
    ctrl_names = {"Control", "siControl"}
    rows = []
    for (kind, treatment, stage), d in p.groupby(["kind", "treatment", "stage"]):
        if treatment in ctrl_names:
            continue
        rows.append(dict(kind=kind, treatment=treatment, stage=stage,
                         n=len(d), median_fold=d.fold.median(),
                         lo=d.fold.quantile(.25), hi=d.fold.quantile(.75)))
    out["perturb"] = pd.DataFrame(rows)

    # --- 3. antioxidant induction across the transformation time course ---
    # The model-relevant quantity is the arsenic effect: 1 uM against the
    # paired 0 uM control at the SAME week, so drift in the untreated line does
    # not masquerade as induction. The '+12' stage has no paired control.
    f3 = parse_fig3()
    m = f3.groupby(["gene", "week", "dose"]).value.mean().unstack("dose")
    ref22 = m.xs("22", level="week")["0uM"]
    rows = []
    for (gene, week), r in m.iterrows():
        den = r.get("0uM", np.nan)
        paired = np.isfinite(den)
        if not paired:
            den = ref22.get(gene, np.nan)
        rows.append(dict(gene=gene, week=week, fold=r["1uM"] / den, paired=paired))
    out["timecourse"] = pd.DataFrame(rows)

    # --- 4. module-level m6A on the independent microarray platform -------
    from .datasets import ANTIOX_CORE, _annotation
    ma = parse_microarray()
    ann = _annotation()
    hit = ann["GOProcess"].fillna("").str.contains("DNA repair", case=False)
    repair = sorted(set(ann.loc[hit, "Symbol"]) & set(ma.index))
    rows = []
    for label, genes in [("antioxidant core", [g for g in ANTIOX_CORE if g in ma.index]),
                         ("DNA-repair module", repair)]:
        v = ma.reindex(genes).dropna()
        from scipy.stats import mannwhitneyu
        rest = ma.drop(v.index, errors="ignore")
        pv = mannwhitneyu(v, rest, alternative="two-sided").pvalue if len(v) >= 5 else np.nan
        rows.append(dict(module=label, n=len(v), median_log2FC=v.median(),
                         fold=2 ** v.median(), p=pv))
    rows.append(dict(module="all genes (platform baseline)", n=len(ma),
                     median_log2FC=ma.median(), fold=2 ** ma.median(), p=np.nan))
    out["microarray"] = pd.DataFrame(rows)
    return out


def main() -> int:
    from ..model import Params
    r = analyse()
    P = Params()

    print("=" * 82)
    print("1. mRNA DECAY under writer/eraser knockdown  (actinomycin chase, k in 1/h)")
    print("=" * 82)
    d = r["decay"]
    for arm in ["siMETTL3", "siFTO"]:
        a = d[d.arm == arm]
        faster = (a.ratio > 1).sum()
        print(f"\n  {arm}:  decay faster than siControl in {faster}/{len(a)} gene x stage cells"
              f"   median k ratio = {a.ratio.median():.2f}")
        for _, x in a.sort_values(["gene", "stage"]).iterrows():
            flag = "*" if x.p < 0.05 else " "
            print(f"    {x.gene:6s} {x.stage:7s} k {x.k_ctl:6.3f} -> {x.k_arm:6.3f}"
                  f"   ratio {x.ratio:5.2f} {flag}  p={x.p:.3f}")

    print("\n" + "=" * 82)
    print("2. ANTIOXIDANT LEVEL under writer/eraser perturbation (fold vs own control)")
    print("=" * 82)
    p = r["perturb"].sort_values(["kind", "treatment", "stage"])
    print(f"  {'handle':16s}{'treatment':20s}{'stage':8s}{'median':>9s}{'IQR':>18s}")
    for _, x in p.iterrows():
        print(f"  {x.kind:16s}{x.treatment:20s}{x.stage:8s}{x.median_fold:9.2f}"
              f"   [{x.lo:.2f}, {x.hi:.2f}]")

    print("\n" + "=" * 82)
    print("3. ANTIOXIDANT INDUCTION across transformation (fold vs week 0, 1 uM arm)")
    print("=" * 82)
    t = r["timecourse"].pivot(index="gene", columns="week", values="fold")
    t = t[[c for c in ["0", "1", "4", "8", "22", "+12"] if c in t.columns]]
    print(t.round(2).to_string())
    print("  ('+12' has no paired 0 uM arm; referenced to the week-22 control)")
    late = t[[c for c in ["22", "+12"] if c in t.columns]].median(axis=1)
    print(f"\n  median late-stage induction, 5 induced genes: "
          f"{late.drop('PRDX5').median():.2f}x   PRDX5: {late['PRDX5']:.2f}x")
    print(f"  model assumes a single uniform antioxidant pool rising to 1.90x")

    print("\n" + "=" * 82)
    print("4. MODULE m6A on the microarray (independent platform, not MeRIP)")
    print("=" * 82)
    m = r["microarray"]
    print(f"  {'module':32s}{'n':>7s}{'median log2FC':>15s}{'fold':>8s}{'p':>11s}")
    for _, x in m.iterrows():
        pv = "" if not np.isfinite(x.p) else f"{x.p:.2e}"
        print(f"  {x.module:32s}{x.n:7d}{x.median_log2FC:15.3f}{x.fold:8.2f}{pv:>11s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
