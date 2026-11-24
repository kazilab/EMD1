"""Quantify regional m6A enrichment from GSE145923 bigWig tracks.

Use hg19 coordinates for these tracks. The gene-count annotation uses GRCh38.p13.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ..paths import OUTPUT_DIR, RAW_DATA_DIR
from .store import source_status
from .verified_features import load_verified_feature

BW = RAW_DATA_DIR / "GSE145923" / "bw"

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

# NEDD4L exon 1 requires a verified hg38-to-hg19 mapping.
PUBLISHED_FEATURES: dict[str, tuple] = {}


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
        match = re.search(r"-(ip|input)-([A-Za-z]+)([12])$", stem)
        if match is None or match.group(2) not in TAG:
            raise ValueError(f"Unrecognised GSE145923 track identity: {p.name}")
        kind = "IP" if match.group(1) == "ip" else "input"
        tag, rep = match.group(2), int(match.group(3))
        import pyBigWig
        bw = pyBigWig.open(str(p))
        total = float(bw.header()["sumData"])
        bw.close()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid bigWig normalization total: {p}")
        out.append(Track(p, TAG[tag], kind, rep, total))
    if not out:
        raise FileNotFoundError(
            f"No bigWig source tracks under {BW}. Acquire and checksum-verify "
            "sources first with python -m emd1_simulation.acquisition.download.")
    keys = [(tr.cond, tr.kind, tr.rep) for tr in out]
    expected = {(c, k, r) for c in TAG.values() for k in ("IP", "input") for r in (1, 2)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "GSE145923 requires both input and IP replicates 1 and 2 in every condition; "
            f"missing={sorted(expected - set(keys))}, unexpected={sorted(set(keys) - expected)}")
    return out


def binned(tr: Track, chrom: str, start: int, end: int, binsize: int) -> np.ndarray:
    """Average exact signal over each bin’s full width, counting uncovered bases as zero."""
    import pyBigWig
    if end <= start or binsize <= 0:
        raise ValueError("Binned coverage requires a positive interval and bin size")
    bw = pyBigWig.open(str(tr.path))
    try:
        values = []
        for lo in range(start, end, binsize):
            hi = min(lo + binsize, end)
            value = bw.stats(chrom, lo, hi, type="sum", exact=True)[0]
            values.append((0.0 if value is None else float(value)) / (hi - lo))
    finally:
        bw.close()
    a = np.array(values)
    return a / tr.total * 1e9


def interval_signal(tr: Track, chrom: str, start: int, end: int) -> float:
    """Exact library-normalised signal over one prespecified half-open interval."""
    import pyBigWig
    bw = pyBigWig.open(str(tr.path))
    value = bw.stats(chrom, start, end, type="sum", exact=True)[0]
    bw.close()
    return (0.0 if value is None else float(value)) / tr.total * 1e9


def published_feature_enrichment(gene: str, *, feature_file: str | Path | None = None) -> dict:
    """Pooled IP/input contrasts after verifying source-coordinate evidence.

    Same-index ratios are emitted only as descriptive diagnostics: GEO does not
    document these cultures as paired, so the pooled contrast is primary.
    """
    if feature_file is None:
        raise ValueError(
            f"{gene} source-coordinate mapping is unresolved. Supply an explicitly verified "
            "feature JSON with checksummed assembly-chain and transcript-annotation evidence.")
    feature = load_verified_feature(feature_file, gene)
    chrom, start, end, source = (
        feature["chrom"], feature["start_0based"], feature["end_0based"], feature["source"])
    ts = tracks()
    values = {
        (tr.cond, tr.kind, tr.rep): interval_signal(tr, chrom, start, end)
        for tr in ts
    }
    if any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Source interval has nonfinite or negative coverage signal")
    out = dict(gene=gene, status="ok", estimator="source interval with verified mapping",
               chrom=chrom, start_0based=start, end_0based=end,
               interval_nt=end - start, coordinate_source=source,
               transcript=feature["transcript"],
               feature_record_sha256=feature["feature_record_sha256"],
               assembly_chain_sha256=feature["assembly_chain_sha256"],
               target_refgene_sha256=feature["target_refgene_sha256"])
    for cond in ("Ctl", "As", "AsT"):
        reps = sorted(rep for c, kind, rep in values if c == cond and kind == "IP")
        ip = np.asarray([values[(cond, "IP", rep)] for rep in reps])
        ino = np.asarray([values[(cond, "input", rep)] for rep in reps])
        out[f"{cond}_enr"] = float(ip.sum() / ino.sum()) if ino.sum() > 0 else np.nan
        out[f"{cond}_input"] = float(ino.mean())
        out[f"{cond}_min_input_density"] = float(ino.min() / (end - start))
        if out[f"{cond}_min_input_density"] < 1.0:
            out["status"] = "low_input_coverage_quantification_unreliable"
        for rep, rep_ip, rep_in in zip(reps, ip, ino):
            out[f"{cond}_rep{rep}_enr"] = float(rep_ip / rep_in) if rep_in > 0 else np.nan
    if not np.isfinite(out["Ctl_enr"]) or out["Ctl_enr"] <= 0:
        raise ValueError("NEDD4L control IP/input is not quantifiable on the verified interval")
    for cond in ("As", "AsT"):
        out[f"{cond}/Ctl"] = out[f"{cond}_enr"] / out["Ctl_enr"]
        for rep in (1, 2):
            baseline = out[f"Ctl_rep{rep}_enr"]
            out[f"{cond}/Ctl_index_ratio_rep{rep}"] = (
                out[f"{cond}_rep{rep}_enr"] / baseline
                if np.isfinite(baseline) and baseline > 0 else np.nan)
    return out


def profiles(gene: str, binsize: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    chrom, start, end, _ = LOCI[gene]
    ts = tracks()
    cols = {}
    for t in ts:
        cols[f"{t.cond}_{t.kind}{t.rep}"] = binned(t, chrom, start, end, binsize)
    df = pd.DataFrame(cols)
    pos = start + np.arange(len(df)) * binsize
    df.attrs["bin_widths"] = np.minimum(binsize, end - pos)
    return df, pos


def peak_enrichment(gene: str, binsize: int = 100, peak_nt: int = 300,
                    min_input: float = 1.0) -> dict:
    """Score a contiguous window selected in the untreated control. Selecting the
    highest control enrichment can bias comparisons with other conditions.
    """
    df, pos = profiles(gene, binsize)
    widths = np.asarray(df.attrs.get("bin_widths", np.full(len(df), binsize)))
    conds = ("Ctl", "As", "AsT")

    def mean_of(cond, kind):
        return df[[c for c in df.columns if c.startswith(f"{cond}_{kind}")]].mean(axis=1)

    # Coverage eligibility is set using the untreated control alone.
    expressed = mean_of("Ctl", "input") > min_input
    if expressed.sum() < 5:
        return dict(gene=gene, status="input coverage too low", n_bins=int(expressed.sum()))

    ctl_enr = np.log2((mean_of("Ctl", "IP") + .01) / (mean_of("Ctl", "input") + .01))

    # Score only contiguous windows with sufficient input throughout.
    w = max(3, int(peak_nt // binsize))
    e = ctl_enr.to_numpy()
    ok = expressed.to_numpy()
    best, best_i = -np.inf, None
    for i in range(len(e) - w + 1):
        if not ok[i:i + w].all():
            continue
        m = np.average(e[i:i + w], weights=widths[i:i + w])
        if m > best:
            best, best_i = m, i
    if best_i is None:
        return dict(gene=gene, status="no contiguous expressed window", n_bins=int(ok.sum()))
    win = pd.Index(range(best_i, best_i + w))

    bg = np.nanmedian(e[ok & ~np.isin(np.arange(len(e)), win)])
    out = dict(gene=gene, status="ok", n_expressed_bins=int(expressed.sum()),
               n_peak_bins=w, peak_over_background=float(2 ** (best - bg)),
               peak_start_bin=int(best_i),
               peak_start_0based=int(pos[best_i]),
               peak_end_0based=int(pos[best_i + w - 1] + widths[best_i + w - 1]),
               peak_kb=f"{(pos[best_i]-pos[0])/1000:.1f}")
    for c in conds:
        ip = float((mean_of(c, "IP")[win] * widths[win]).sum())
        ino = float((mean_of(c, "input")[win] * widths[win]).sum())
        out[f"{c}_enr"] = float(ip / ino) if ino > 0 else np.nan
        out[f"{c}_input"] = float(mean_of(c, "input")[expressed].mean())
    for c in ["As", "AsT"]:
        out[f"{c}/Ctl"] = out[f"{c}_enr"] / out["Ctl_enr"] if out["Ctl_enr"] else np.nan
    return out


def region_table(gene: str, binsize: int = 100, utr_kb: float = 2.0,
                 min_input: float = 1.0, *, feature_file: str | Path | None = None) -> pd.DataFrame:
    """Compare enrichment in the control-selected peak, 3′ terminal window and expressed gene body."""
    df, pos = profiles(gene, binsize)
    widths = np.asarray(df.attrs.get("bin_widths", np.full(len(df), binsize)))
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
        # Rounded kilobases are display text, never a genomic locator.
        i0 = best["peak_start_bin"]
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
    if gene == "NEDD4L" and feature_file is not None:
        fixed = published_feature_enrichment(gene, feature_file=feature_file)
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
            ip = float((mean_of(c, "IP")[sel] * widths[sel]).sum())
            ino = float((mean_of(c, "input")[sel] * widths[sel]).sum())
            e = ip / ino if ino > 0 else np.nan
            if c == "Ctl":
                base = e
            reps = []
            for r in (1, 2):
                a = float((mean_of(c, "IP", r)[sel] * widths[sel]).sum())
                b = float((mean_of(c, "input", r)[sel] * widths[sel]).sum())
                reps.append(a / b if b > 0 else np.nan)
            rows.append(dict(gene=gene, region=name, cond=c, IP=ip, input=ino,
                             enr=e, vs_ctl=np.nan if c == "Ctl" else e / base,
                             rep1=reps[0], rep2=reps[1]))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-nedd4l-feature", type=Path,
                        help="Feature JSON with checksummed assembly-chain and first-exon evidence")
    args = parser.parse_args()
    print("=" * 96)
    print("Exploratory m6A enrichment, GSE145923 tracks (hg19, n=2)")
    if args.verified_nedd4l_feature is None:
        print("NEDD4L published-feature measurement is unresolved and has no default target.")
    print("=" * 96)
    for g in LOCI:
        t = region_table(g, feature_file=args.verified_nedd4l_feature)
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
    outdir = OUTPUT_DIR / "source_reanalysis"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.verified_nedd4l_feature is not None:
        fixed = published_feature_enrichment("NEDD4L", feature_file=args.verified_nedd4l_feature)
        out = outdir / "nedd4l_published_exon1.csv"
        pd.DataFrame([fixed]).to_csv(out, index=False)
        print(f"\nVerified source interval: As/Ctl={fixed['As/Ctl']:.3f}; status={fixed['status']}.")
        print(f"wrote {out}")
    else:
        (outdir / "nedd4l_published_exon1.csv").unlink(missing_ok=True)
        out = outdir / "nedd4l_source_status.json"
        out.write_text(json.dumps(source_status()["sources"]["nedd4l_exon1"], indent=2) + "\n")
        print(f"wrote unresolved source status: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
