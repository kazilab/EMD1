"""Measure MeRIP enrichment on the first three annotated exons (Yang et al., Fig. 4A).

For each gene, use the longest spliced hg19 RefSeq NM_ isoform and sum
coverage over its first three exons in transcriptional order. Apply the same
rule to DDB2, XPC, the repair set and 600 sampled background transcripts.
Inclusion requires adequate signal in every control input replicate. Pool
library-normalised IP and input within each condition. Same-index replicate
ratios are descriptive because GEO does not document matched cultures.
"""

from __future__ import annotations

import glob
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DATA, repair_gene_symbols
from ..paths import OUTPUT_DIR

REFGENE = DATA / "hg19_refGene.txt.gz"
REFGENE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
MAIN_CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X"]}

# XPC is a same-pathway comparator. The background transcripts define specificity.
FOCUS = ("DDB2", "XPC")
N_BACKGROUND = 600
SEED = 20260825
N_FEATURE_EXONS = 3
MIN_CONTROL_INPUT_DENSITY = 1.0
INPUT_QC_BIN_NT = 100
MIN_CONTROL_INPUT_BINS = 3
LIBRARY_SCALE = 1e9


@dataclass(frozen=True)
class SeriesSpec:
    gse: str
    tags: dict          # filename tag -> condition label
    ctl: str
    order: list
    expected_replicates: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class TranscriptRegion:
    """A predeclared first-three-exon feature in transcriptional order."""

    accession: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]
    transcript_nt: int

    @property
    def start(self) -> int:
        return min(start for start, _ in self.exons)

    @property
    def end(self) -> int:
        return max(end for _, end in self.exons)

    @property
    def exon_nt(self) -> int:
        return sum(end - start for start, end in self.exons)


SERIES = (
    SeriesSpec("GSE145923", {"noAs": "Ctl", "chrAs": "As", "mouseAs": "AsT"},
               "Ctl", ["Ctl", "As", "AsT"]),
    SeriesSpec("GSE145924", {"NC": "shNC", "SH": "shMETTL14", "UVB": "UVB"},
               "shNC", ["shNC", "shMETTL14", "UVB"]),
)


def _transcriptional_first_exons(starts: list[int], ends: list[int],
                                 strand: str) -> tuple[tuple[int, int], ...]:
    """Return the first three exons in 5'-to-3' transcript order."""
    genomic = tuple(zip(starts, ends))
    if len(genomic) < N_FEATURE_EXONS:
        return ()
    if strand == "+":
        return genomic[:N_FEATURE_EXONS]
    if strand == "-":
        return tuple(reversed(genomic[-N_FEATURE_EXONS:]))
    raise ValueError(f"invalid refGene strand: {strand!r}")


def _refgene() -> dict[str, TranscriptRegion]:
    """Select the longest spliced hg19 RefSeq NM_ isoform with at least three exons."""
    if not REFGENE.is_file():
        raise FileNotFoundError(
            f"Required source annotation is missing: {REFGENE}. "
            "Acquire and checksum-verify sources first with "
            "python -m emd1_simulation.acquisition.download.")
    candidates: dict[str, list[TranscriptRegion]] = {}
    with gzip.open(REFGENE, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            acc, chrom, strand, sym = f[1], f[2], f[3], f[12]
            if not acc.startswith("NM_") or chrom not in MAIN_CHROMS:
                continue
            starts = [int(x) for x in f[9].rstrip(",").split(",")]
            ends = [int(x) for x in f[10].rstrip(",").split(",")]
            if len(starts) != len(ends) or any(end <= start for start, end in zip(starts, ends)):
                continue
            exons = _transcriptional_first_exons(starts, ends, strand)
            if not exons:
                continue
            candidates.setdefault(sym, []).append(TranscriptRegion(
                accession=acc, chrom=chrom, strand=strand, exons=exons,
                transcript_nt=sum(end - start for start, end in zip(starts, ends))))

    # Break equal-length ties by RefSeq accession.
    return {
        symbol: min(regions, key=lambda r: (-r.transcript_nt, r.accession))
        for symbol, regions in candidates.items()
    }


def _open_tracks(spec: SeriesSpec):
    import pyBigWig
    out = []
    for path in sorted(glob.glob(str(DATA / spec.gse / "bw" / "*.bw"))):
        stem = Path(path).name
        kind = "IP" if "-ip-" in stem else "input"
        m = re.search(r"-(?:ip|input)-([A-Za-z0-9]+?)(\d)\.bw$", stem)
        if m is None or m.group(1) not in spec.tags:
            continue
        bw = pyBigWig.open(path)
        total = float(bw.header()["sumData"])
        if not np.isfinite(total) or total <= 0:
            bw.close()
            raise ValueError(f"invalid bigWig sumData in {path}: {total}")
        out.append((bw, spec.tags[m.group(1)], kind, int(m.group(2)), total))
    if not out:
        raise FileNotFoundError(
            f"No bigWig source tracks for {spec.gse} under {DATA}. "
            "Acquire and checksum-verify sources first with "
            "python -m emd1_simulation.acquisition.download.")
    # Both input and IP tracks are required for each experimental replicate.
    keys = [(cond, kind, rep) for _, cond, kind, rep, _ in out]
    if len(keys) != len(set(keys)):
        for bw, *_ in out:
            bw.close()
        raise ValueError(f"duplicate bigWig condition/kind/replicate in {spec.gse}")
    for cond in spec.order:
        ip_reps = {rep for _, c, kind, rep, _ in out if c == cond and kind == "IP"}
        in_reps = {rep for _, c, kind, rep, _ in out if c == cond and kind == "input"}
        if ip_reps != set(spec.expected_replicates) or in_reps != set(spec.expected_replicates):
            for bw, *_ in out:
                bw.close()
            raise ValueError(
                f"incomplete IP/input source replicates in {spec.gse} {cond}: "
                f"IP={sorted(ip_reps)}, input={sorted(in_reps)}, "
                f"expected={list(spec.expected_replicates)}")
    return out


def _normalised_interval_signal(bw, total: float, chrom: str,
                                start: int, end: int) -> float:
    """Exact library-normalised signal in one half-open genomic interval."""
    if chrom not in bw.chroms():
        return np.nan
    value = bw.stats(chrom, start, end, type="sum", exact=True)[0]
    return (0.0 if value is None else float(value)) / total * LIBRARY_SCALE


def _normalised_region_signal(bw, total: float, region: TranscriptRegion) -> float:
    """Exact library-normalised bigWig sum over the annotated exon union."""
    signal = 0.0
    for start, end in region.exons:
        value = _normalised_interval_signal(bw, total, region.chrom, start, end)
        if not np.isfinite(value):
            return np.nan
        signal += value
    return signal


def _control_input_qc(bw, total: float, region: TranscriptRegion,
                      min_density: float) -> tuple[int, float]:
    """Count exon bins passing the control input threshold and return the mean density across all bins."""
    densities = []
    for exon_start, exon_end in region.exons:
        for start in range(exon_start, exon_end, INPUT_QC_BIN_NT):
            end = min(start + INPUT_QC_BIN_NT, exon_end)
            signal = _normalised_interval_signal(bw, total, region.chrom, start, end)
            densities.append(signal / (end - start))
    arr = np.asarray(densities, dtype=float)
    if not np.isfinite(arr).all():
        return 0, np.nan
    return int(np.count_nonzero(arr >= min_density)), float(arr.mean())


def first_three_exon_ratio(bws, spec: SeriesSpec, region: TranscriptRegion,
                           min_input_density: float = MIN_CONTROL_INPUT_DENSITY
                           ) -> dict | None:
    """Calculate pooled IP/input fold changes for features passing QC in every control input replicate."""
    signals = {
        (cond, kind, rep): _normalised_region_signal(bw, total, region)
        for bw, cond, kind, rep, total in bws
    }
    ctl_qc = [
        _control_input_qc(bw, total, region, min_input_density)
        for bw, cond, kind, _, total in bws
        if cond == spec.ctl and kind == "input"
    ]
    ctl_passing_bins = np.asarray([passing for passing, _ in ctl_qc], dtype=int)
    ctl_mean_densities = np.asarray([density for _, density in ctl_qc], dtype=float)
    if (ctl_passing_bins.size == 0 or not np.isfinite(ctl_mean_densities).all()
            or np.any(ctl_passing_bins < MIN_CONTROL_INPUT_BINS)):
        return None

    enrichments: dict[str, float] = {}
    replicate_enrichments: dict[tuple[str, int], float] = {}
    for cond in spec.order:
        reps = sorted(rep for c, kind, rep in signals if c == cond and kind == "IP")
        ip = np.asarray([signals[(cond, "IP", rep)] for rep in reps], dtype=float)
        ino = np.asarray([signals[(cond, "input", rep)] for rep in reps], dtype=float)
        if not np.isfinite(ip).all() or not np.isfinite(ino).all() or ino.sum() <= 0:
            enrichments[cond] = np.nan
            continue
        # Tracks are individually library-size normalised above, then pooled.
        # With matched replicate counts this is also the ratio of their means.
        enrichments[cond] = ip.sum() / ino.sum()
        for rep, rep_ip, rep_in in zip(reps, ip, ino):
            replicate_enrichments[(cond, rep)] = rep_ip / rep_in if rep_in > 0 else np.nan

    base = enrichments.get(spec.ctl, np.nan)
    if not np.isfinite(base) or base <= 0:
        return None

    out = {
        "accession": region.accession,
        "chrom": region.chrom,
        "strand": region.strand,
        # The exon intervals specify the measured bases within the genomic span.
        "region_bounding_start_0based": region.start,
        "region_bounding_end_0based": region.end,
        "exon_intervals_0based": ";".join(
            f"{start}-{end}" for start, end in region.exons),
        "region_exon_nt": region.exon_nt,
        "input_qc_bin_nt": INPUT_QC_BIN_NT,
        "min_control_input_bins": MIN_CONTROL_INPUT_BINS,
        "min_control_input_density": min_input_density,
        "control_input_passing_bins_min": int(ctl_passing_bins.min()),
        "control_input_mean_density_min": float(ctl_mean_densities.min()),
    }
    ctl_reps = sorted(rep for cond, kind, rep in signals
                      if cond == spec.ctl and kind == "IP")
    for cond in spec.order:
        value = enrichments.get(cond, np.nan)
        out[cond] = value / base if np.isfinite(value) else np.nan
        for rep in ctl_reps:
            ctl_rep = replicate_enrichments.get((spec.ctl, rep), np.nan)
            cond_rep = replicate_enrichments.get((cond, rep), np.nan)
            out[f"{cond}__rep{rep}"] = (
                cond_rep / ctl_rep
                if np.isfinite(cond_rep) and np.isfinite(ctl_rep) and ctl_rep > 0
                else np.nan)
    return out


def quantify(spec: SeriesSpec, loci: dict[str, TranscriptRegion], repair: list,
             background: list) -> pd.DataFrame:
    bws = _open_tracks(spec)
    try:
        rows = []
        for group, genes in (("repair", repair), ("background", background), ("focus", FOCUS)):
            for g in genes:
                if g not in loci:
                    continue
                r = first_three_exon_ratio(bws, spec, loci[g])
                if r:
                    rows.append({"gene": g, "set": group, **r})
    finally:
        for bw, *_ in bws:
            bw.close()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    """Normalise each contrast to its background median and report the 5th–95th
    background percentiles. Same-index diagnostics use replicate-specific medians.
    """
    rows = []
    for cond in spec.order[1:]:
        d = df.dropna(subset=[cond])
        bg = d[d.set == "background"][cond]
        med = bg.median()
        lo, hi = np.percentile(bg / med, [5, 95])
        module = d[d.set == "repair"][cond].median() / med
        replicate_cols = sorted(col for col in d.columns if col.startswith(f"{cond}__rep"))
        for gene in FOCUS:
            hit = d[(d.gene == gene) & (d.set == "focus")]
            if hit.empty:
                continue
            hit = hit.iloc[0]
            val = float(hit[cond])
            row = dict(
                series=spec.gse, contrast=f"{cond}/{spec.ctl}", gene=gene,
                pooled_raw_ratio=val, background_raw_median=float(med),
                effect=val / med,
                effect_definition="pooled_raw_ratio / background_raw_median",
                pct=float((bg < val).mean() * 100),
                null_lo=lo, null_hi=hi, module=module,
                n_background_drawn=N_BACKGROUND, n_background_used=int(bg.size),
                n_repair=int((d.set == "repair").sum()),
                accession=hit["accession"], chrom=hit["chrom"], strand=hit["strand"],
                region_bounding_start_0based=int(hit["region_bounding_start_0based"]),
                region_bounding_end_0based=int(hit["region_bounding_end_0based"]),
                exon_intervals_0based=hit["exon_intervals_0based"],
                region_exon_nt=int(hit["region_exon_nt"]),
                input_qc_bin_nt=int(hit["input_qc_bin_nt"]),
                min_control_input_bins=int(hit["min_control_input_bins"]),
                min_control_input_density=float(hit["min_control_input_density"]),
                control_input_passing_bins_min=int(hit["control_input_passing_bins_min"]),
                control_input_mean_density_min=float(hit["control_input_mean_density_min"]))
            for col in replicate_cols:
                rep_bg = d.loc[d.set == "background", col].dropna()
                rep_med = rep_bg.median()
                suffix = col.rsplit("__rep", 1)[1]
                row[f"index_raw_ratio_rep{suffix}"] = (
                    float(hit[col]) if np.isfinite(hit[col]) else np.nan)
                row[f"index_background_normalized_ratio_rep{suffix}"] = (
                    float(hit[col]) / rep_med
                    if rep_med > 0 and np.isfinite(hit[col]) else np.nan)
                row[f"index_ratio_rep{suffix}"] = (
                    float(hit[col]) / rep_med
                    if rep_med > 0 and np.isfinite(hit[col]) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    loci = _refgene()
    repair = sorted(repair_gene_symbols() & set(loci))
    rng = np.random.default_rng(SEED)
    pool = sorted(set(loci) - set(repair) - set(FOCUS))
    background = list(rng.choice(pool, size=N_BACKGROUND, replace=False))

    print("=" * 94)
    print("MeRIP enrichment in the first three exons of DDB2 and XPC")
    print(f"hg19 features in {len(repair)} repair genes and "
          f"{len(background)} sampled background transcripts.")
    print("The 'n used' column gives the background transcripts passing QC for each contrast.")
    ddb2_region = loci["DDB2"]
    exon_text = ", ".join(f"{start}-{end}" for start, end in ddb2_region.exons)
    print(f"DDB2: {ddb2_region.accession} {ddb2_region.chrom} {ddb2_region.strand}; "
          f"first-three-exon union [{exon_text}] ({ddb2_region.exon_nt} exonic nt).")
    print(f"Feature selection is annotation-only; every control input replicate must have >= "
          f"{MIN_CONTROL_INPUT_BINS} fixed ~{INPUT_QC_BIN_NT}-nt bins at density >= "
          f"{MIN_CONTROL_INPUT_DENSITY:g}.")
    print("=" * 94)

    out = []
    for spec in SERIES:
        df = quantify(spec, loci, repair, background)
        out.append(summarise(df, spec))
    summary = pd.concat(out, ignore_index=True)

    print(f"\n  {'contrast':26s}{'gene':7s}{'effect':>8s}{'pct':>7s}"
          f"{'null (5-95%)':>18s}{'n used':>8s}{'module':>8s}{'index ratios':>20s}")
    print("  " + "-" * 92)
    for _, r in summary.iterrows():
        flag = "  <-- outside null" if not (r.null_lo <= r.effect <= r.null_hi) else ""
        indexed = ",".join(
            f"r{col.removeprefix('index_ratio_rep')}={r[col]:.2f}"
            for col in summary.columns
            if col.startswith("index_ratio_rep") and pd.notna(r[col]))
        print(f"  {r.series + ' ' + r.contrast:26s}{r.gene:7s}{r.effect:8.2f}"
              f"{r.pct:6.1f}%   [{r.null_lo:.2f}, {r.null_hi:.2f}]"
              f"{r.n_background_used:8d}{r.module:8.2f}{indexed:>20s}{flag}")

    print("The background range is the 5th–95th percentile across transcripts.")
    print("Same-index replicate ratios are descriptive. The GEO records do not document pairing.")

    out_csv = OUTPUT_DIR / "source_reanalysis" / "repair_edge_first_three_exons.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
