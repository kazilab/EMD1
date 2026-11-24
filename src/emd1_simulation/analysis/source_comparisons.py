"""Summarise antioxidant chase and expression data from the original assays.

Requires the source workbooks and annotation under data/raw/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..processing.chase import (
    decay_rates, parse_fig3, parse_fig5b, parse_microarray, parse_perturbation,
)


def analyse() -> dict[str, pd.DataFrame]:
    out = {}

    # Estimate each decay contrast within gene and exposure stage.
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
                             n_ctl=len(ctl), n_arm=len(a),
                             sd_ctl=ctl.std(ddof=1), sd_arm=a.std(ddof=1)))
    out["decay"] = pd.DataFrame(rows)

    # Compare antioxidant levels within treatment and stage.
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

    # Use the same-week untreated control, or the week-22 control for +12.
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
    from ..processing.datasets import ANTIOX_CORE, _annotation
    ma = parse_microarray()
    ann = _annotation()
    hit = ann["GOProcess"].fillna("").str.contains("DNA repair", case=False)
    repair = sorted(set(ann.loc[hit, "Symbol"]) & set(ma.index))
    rows = []
    for label, genes in [("antioxidant core", [g for g in ANTIOX_CORE if g in ma.index]),
                         ("DNA-repair module", repair)]:
        v = ma.reindex(genes).dropna()
        rows.append(dict(module=label, n=len(v), median_log2FC=v.median(),
                         fold=2 ** v.median()))
    rows.append(dict(module="all genes (platform baseline)", n=len(ma),
                     median_log2FC=ma.median(), fold=2 ** ma.median()))
    out["microarray"] = pd.DataFrame(rows)
    return out


def main() -> int:
    r = analyse()

    print("mRNA decay under writer or eraser knockdown (actinomycin chase, k in 1/h)")
    d = r["decay"]
    for arm in ["siMETTL3", "siFTO"]:
        a = d[d.arm == arm]
        faster = (a.ratio > 1).sum()
        print(f"\n  {arm}:  decay faster than siControl in {faster}/{len(a)} gene x stage cells"
              f"   median k ratio = {a.ratio.median():.2f}")
        for _, x in a.sort_values(["gene", "stage"]).iterrows():
            print(f"    {x.gene:6s} {x.stage:7s} k {x.k_ctl:6.3f} -> {x.k_arm:6.3f}"
                  f"   ratio {x.ratio:5.2f}  series n={x.n_ctl}/{x.n_arm}")

    print("\nAntioxidant levels under writer or eraser perturbation (fold vs matched control)")
    p = r["perturb"].sort_values(["kind", "treatment", "stage"])
    print(f"  {'handle':16s}{'treatment':20s}{'stage':8s}{'median':>9s}{'IQR':>18s}")
    for _, x in p.iterrows():
        print(f"  {x.kind:16s}{x.treatment:20s}{x.stage:8s}{x.median_fold:9.2f}"
              f"   [{x.lo:.2f}, {x.hi:.2f}]")

    print("\nAntioxidant induction (1 µM arsenic vs same-week 0 µM control)")
    t = r["timecourse"].pivot(index="gene", columns="week", values="fold")
    t = t[[c for c in ["0", "1", "4", "8", "22", "+12"] if c in t.columns]]
    print(t.round(2).to_string())
    print("The +12 stage uses the week-22 control because it lacks a paired untreated arm.")
    late = t[[c for c in ["22", "+12"] if c in t.columns]].median(axis=1)
    print(f"\n  median late-stage induction, 5 induced genes: "
          f"{late.drop('PRDX5').median():.2f}x   PRDX5: {late['PRDX5']:.2f}x")

    print("\nModule m6A on the independent microarray platform")
    m = r["microarray"]
    print("The distributions summarise variation across genes in each module.")
    print(f"  {'module':32s}{'n':>7s}{'median log2FC':>15s}{'fold':>8s}")
    for _, x in m.iterrows():
        print(f"  {x.module:32s}{x.n:7d}{x.median_log2FC:15.3f}{x.fold:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
