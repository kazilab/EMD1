"""Prespecified and exploratory site-level m6A from GSE145923 tracks.

Gene-level IP/input averages the whole gene body and, for APOBEC3B and NEDD4L,
gave results that contradicted both the model and the source papers. Both
papers' claims are about specific peaks, so this reads the deposited bigwigs
directly and quantifies enrichment in the peak window rather than across the
locus.

Note the tracks are **hg19**, while the counts matrices and the NCBI annotation
are GRCh38.p13 — coordinates here come from the UCSC hg19 assembly and must not
be taken from the annotation file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BW = (Path(__file__).resolve().parent.parent / "data" / "GSE145923" / "bw")

# filename tag -> condition.  chrAs = chronic arsenic, mouseAs = tumorigenic
# (xenograft-derived), noAs = control.
TAG = {"noAs": "Ctl", "chrAs": "As", "mouseAs": "AsT"}

# hg19, from the UCSC assembly (hgnc track).
LOCI = {
    "APOBEC3B": ("chr22", 39378352, 39388809, "+"),
    "NEDD4L":   ("chr18", 55711458, 56068772, "+"),
    "SOD1":     ("chr21", 33031979, 33041244, "+"),
    "CAT":      ("chr11", 34460481, 34493607, "+"),
}

# The source paper prints the GRCh38 exon-1 interval
# NC_000018.10:58221610-58221760 (Fig. 3f). Its hg19 lift is the 150-nt
# interval below. This coordinate is external to the deposited track signal,
# so it is the primary NEDD4L estimator and avoids winner's-curse selection on
# control IP/input. The sliding-window function below remains exploratory.
PUBLISHED_FEATURES = {
    "NEDD4L": ("chr18", 55712458, 55712608,
                "Cui et al. Nat Commun 2021 Fig. 3f; GRCh38 interval lifted to hg19"),
}


@dataclass
class Track:
    path: Path
    cond: str
    kind: str      # "IP" or "input"
    rep: int
    total: float   # total signal, for library normalisation


def tracks() -> list[Track]:
    out = []
    for p in sorted(BW.glob("*.bw")):
        stem = p.stem                      # GSM..._As-ip-chrAs1
        kind = "IP" if "-ip-" in stem else "input"
        tag = next(t for t in TAG if stem.endswith(t + "1") or stem.endswith(t + "2"))
        rep = int(stem[-1])
        import pyBigWig
        bw = pyBigWig.open(str(p))
        total = float(bw.header()["sumData"])
        bw.close()
        out.append(Track(p, TAG[tag], kind, rep, total))
    return out


def binned(tr: Track, chrom: str, start: int, end: int, binsize: int) -> np.ndarray:
    """Library-normalised mean coverage per bin (signal per billion)."""
    import pyBigWig
    n = max(1, (end - start) // binsize)
    bw = pyBigWig.open(str(tr.path))
    v = bw.stats(chrom, start, start + n * binsize, nBins=n, type="mean")
    bw.close()
    a = np.array([0.0 if x is None else float(x) for x in v])
    return a / tr.total * 1e9


def interval_signal(tr: Track, chrom: str, start: int, end: int) -> float:
    """Exact library-normalised signal over one prespecified half-open interval."""
    import pyBigWig
    bw = pyBigWig.open(str(tr.path))
    value = bw.stats(chrom, start, end, type="sum", exact=True)[0]
    bw.close()
    return (0.0 if value is None else float(value)) / tr.total * 1e9


def published_feature_enrichment(gene: str) -> dict:
    """Pooled IP/input contrasts on an independently published coordinate.

    Same-index ratios are emitted only as descriptive diagnostics: GEO does not
    document these cultures as paired, so the pooled contrast is primary.
    """
    if gene not in PUBLISHED_FEATURES:
        raise KeyError(f"no prespecified published feature for {gene}")
    chrom, start, end, source = PUBLISHED_FEATURES[gene]
    ts = tracks()
    values = {
        (tr.cond, tr.kind, tr.rep): interval_signal(tr, chrom, start, end)
        for tr in ts
    }
    out = dict(gene=gene, status="ok", estimator="prespecified published interval",
               chrom=chrom, start_0based=start, end_0based=end,
               interval_nt=end - start, coordinate_source=source)
    for cond in ("Ctl", "As", "AsT"):
        reps = sorted(rep for c, kind, rep in values if c == cond and kind == "IP")
        ip = np.asarray([values[(cond, "IP", rep)] for rep in reps])
        ino = np.asarray([values[(cond, "input", rep)] for rep in reps])
        out[f"{cond}_enr"] = float(ip.sum() / ino.sum()) if ino.sum() > 0 else np.nan
        out[f"{cond}_input"] = float(ino.mean())
        for rep, rep_ip, rep_in in zip(reps, ip, ino):
            out[f"{cond}_rep{rep}_enr"] = float(rep_ip / rep_in) if rep_in > 0 else np.nan
    for cond in ("As", "AsT"):
        out[f"{cond}/Ctl"] = out[f"{cond}_enr"] / out["Ctl_enr"]
        for rep in (1, 2):
            out[f"{cond}/Ctl_index_ratio_rep{rep}"] = (
                out[f"{cond}_rep{rep}_enr"] / out[f"Ctl_rep{rep}_enr"])
    return out


def profiles(gene: str, binsize: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    chrom, start, end, _ = LOCI[gene]
    ts = tracks()
    cols = {}
    for t in ts:
        cols[f"{t.cond}_{t.kind}{t.rep}"] = binned(t, chrom, start, end, binsize)
    df = pd.DataFrame(cols)
    pos = start + np.arange(len(df)) * binsize
    return df, pos


def peak_enrichment(gene: str, binsize: int = 100, peak_nt: int = 300,
                    min_input: float = 1.0) -> dict:
    """Exploratory enrichment in a control-selected peak window.

    This discovery utility is not the NEDD4L calibration estimator: maximising
    control IP/input can still create winner's-curse bias even though the window
    is not re-chosen per treatment. Use :func:`published_feature_enrichment` for
    NEDD4L. The window here is defined on the **control** condition only, so it
    is at least not re-chosen per condition.
    Bins with negligible input coverage are excluded: an IP/input ratio built on
    near-zero exonic signal is noise, not methylation.
    """
    df, pos = profiles(gene, binsize)
    conds = ("Ctl", "As", "AsT")

    def mean_of(cond, kind):
        return df[[c for c in df.columns if c.startswith(f"{cond}_{kind}")]].mean(axis=1)

    # Eligibility is a discovery-stage decision and therefore uses control
    # input only. Including As/AsT input leaks the evaluated conditions into
    # feature selection, even when the peak score itself is control-defined.
    expressed = mean_of("Ctl", "input") > min_input
    if expressed.sum() < 5:
        return dict(gene=gene, status="input coverage too low", n_bins=int(expressed.sum()))

    ctl_enr = np.log2((mean_of("Ctl", "IP") + .01) / (mean_of("Ctl", "input") + .01))

    # A real m6A peak is a CONTIGUOUS run of a few hundred nucleotides. Taking
    # the top-k bins wherever they fall returns scattered exonic noise and will
    # manufacture a "peak" in any locus, so require genomic adjacency and all
    # bins in the window to be expressed.
    w = max(3, int(peak_nt // binsize))
    e = ctl_enr.to_numpy()
    ok = expressed.to_numpy()
    best, best_i = -np.inf, None
    for i in range(len(e) - w + 1):
        if not ok[i:i + w].all():
            continue
        m = e[i:i + w].mean()
        if m > best:
            best, best_i = m, i
    if best_i is None:
        return dict(gene=gene, status="no contiguous expressed window", n_bins=int(ok.sum()))
    win = pd.Index(range(best_i, best_i + w))

    # How much does the window stand out from the rest of the expressed locus?
    bg = np.nanmedian(e[ok & ~np.isin(np.arange(len(e)), win)])
    out = dict(gene=gene, status="ok", n_expressed_bins=int(expressed.sum()),
               n_peak_bins=w, peak_over_background=float(2 ** (best - bg)),
               peak_kb=f"{(pos[best_i]-pos[0])/1000:.1f}")
    for c in conds:
        ip, ino = mean_of(c, "IP")[win].sum(), mean_of(c, "input")[win].sum()
        out[f"{c}_enr"] = float(ip / ino) if ino > 0 else np.nan
        out[f"{c}_input"] = float(mean_of(c, "input")[expressed].mean())
    for c in ["As", "AsT"]:
        out[f"{c}/Ctl"] = out[f"{c}_enr"] / out["Ctl_enr"] if out["Ctl_enr"] else np.nan
    return out


def region_table(gene: str, binsize: int = 100, utr_kb: float = 2.0,
                 min_input: float = 1.0) -> pd.DataFrame:
    """Enrichment per condition in three regions of the locus.

    Reporting only the single best-enriched window is misleading: for NEDD4L the
    best-enriched window and the canonical 3'-terminal window disagree, and
    which one you quote decides the answer. All three are shown.
    """
    df, pos = profiles(gene, binsize)
    chrom, start, end, strand = LOCI[gene]

    def mean_of(c, k, rep=None):
        cols = ([x for x in df.columns if x.startswith(f"{c}_{k}")] if rep is None
                else [f"{c}_{k}{rep}"])
        return df[cols].mean(axis=1)

    # Keep region eligibility independent of the evaluated conditions.
    expressed = (mean_of("Ctl", "input") > min_input).to_numpy()

    best = peak_enrichment(gene, binsize=binsize, min_input=min_input)
    regions = {}
    if best.get("status") == "ok":
        i0 = int(round(float(best["peak_kb"]) * 1000 / binsize))
        sel = np.zeros(len(df), bool); sel[i0:i0 + best["n_peak_bins"]] = True
        regions[f"best window (+{best['peak_kb']} kb)"] = sel
    # 3'-terminal window: canonical m6A territory, strand-aware
    nb = int(utr_kb * 1000 / binsize)
    sel = np.zeros(len(df), bool)
    if strand == "+":
        sel[-nb:] = True
    else:
        sel[:nb] = True
    regions[f"3' terminal {utr_kb:g} kb"] = sel
    regions["whole gene (expressed)"] = expressed

    rows = []
    if gene in PUBLISHED_FEATURES:
        fixed = published_feature_enrichment(gene)
        for c in ("Ctl", "As", "AsT"):
            rows.append(dict(
                gene=gene,
                region=(f"published exon-1 {fixed['start_0based']}-"
                        f"{fixed['end_0based']}"),
                cond=c,
                IP=np.nan,
                input=fixed[f"{c}_input"],
                enr=fixed[f"{c}_enr"],
                vs_ctl=np.nan if c == "Ctl" else fixed[f"{c}/Ctl"],
                rep1=fixed[f"{c}_rep1_enr"],
                rep2=fixed[f"{c}_rep2_enr"],
            ))
    for name, sel in regions.items():
        base = None
        for c in ["Ctl", "As", "AsT"]:
            ip, ino = mean_of(c, "IP")[sel].sum(), mean_of(c, "input")[sel].sum()
            e = ip / ino if ino > 0 else np.nan
            if c == "Ctl":
                base = e
            reps = []
            for r in (1, 2):
                a, b = mean_of(c, "IP", r)[sel].sum(), mean_of(c, "input", r)[sel].sum()
                reps.append(a / b if b > 0 else np.nan)
            rows.append(dict(gene=gene, region=name, cond=c, IP=ip, input=ino,
                             enr=e, vs_ctl=np.nan if c == "Ctl" else e / base,
                             rep1=reps[0], rep2=reps[1]))
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 96)
    print("Prespecified + exploratory m6A enrichment, GSE145923 tracks (hg19, n=2)")
    print("=" * 96)
    for g in LOCI:
        t = region_table(g)
        print(f"\n{g}  ({LOCI[g][0]}:{LOCI[g][1]}-{LOCI[g][2]}, {LOCI[g][3]} strand)")
        print(f"  {'region':42s}{'cond':6s}{'input':>9s}{'IP/input':>10s}"
              f"{'vs Ctl':>9s}{'rep1':>8s}{'rep2':>8s}")
        for region in t.region.unique():
            sub = t[t.region == region]
            for _, r in sub.iterrows():
                rel = "" if not np.isfinite(r.vs_ctl) else f"{r.vs_ctl:9.2f}"
                lab = region if r.cond == "Ctl" else ""
                print(f"  {lab:42s}{r.cond:6s}{r.input:9.1f}{r.enr:10.2f}{rel:>9s}"
                      f"{r.rep1:8.2f}{r.rep2:8.2f}")
    fixed = published_feature_enrichment("NEDD4L")
    out = Path("figures/output/nedd4l_published_exon1.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fixed]).to_csv(out, index=False)
    print(f"\nPrimary NEDD4L estimator: As/Ctl={fixed['As/Ctl']:.3f} on the "
          f"prespecified {fixed['interval_nt']}-nt published exon-1 interval.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
