"""Read first-three-exon DDB2 enrichment from the curated database.

Each effect is relative to its contrast-specific background median. The
5th–95th percentile range describes variation among background transcripts.
"""
from .store import read_table

_REPAIR = read_table("repair_edge_summary")
_CONTRASTS = {"arsenic": "As/Ctl", "transformed": "AsT/Ctl",
              "writer_kd": "shMETTL14/shNC", "uvb": "UVB/shNC"}


def _repair_row(key: str, gene: str = "DDB2"):
    rows = _REPAIR.loc[(_REPAIR["contrast"] == _CONTRASTS[key])
                       & (_REPAIR["gene"] == gene)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {gene}/{key} evidence row, found {len(rows)}")
    return rows.iloc[0]


REPAIR_NULL_BY_CONTRAST = {
    key: (float(_repair_row(key).null_lo), float(_repair_row(key).null_hi))
    for key in _CONTRASTS}
REPAIR_NULL_BAND = REPAIR_NULL_BY_CONTRAST["arsenic"]
REPAIR_N_QUANTIFIED = {key: int(_repair_row(key).n_background_used) for key in _CONTRASTS}
REPAIR_N_REPAIR_QUANTIFIED = {key: int(_repair_row(key).n_repair) for key in _CONTRASTS}
REPAIR_DDB2 = {key: (float(_repair_row(key).effect), float(_repair_row(key).pct))
               for key in _CONTRASTS}
REPAIR_XPC = {key: float(_repair_row(key, "XPC").effect) for key in _CONTRASTS}
REPAIR_MODULE = {key: float(_repair_row(key).module) for key in _CONTRASTS}
REPAIR_N_BACKGROUND = int(_repair_row("arsenic").n_background_drawn)

# Coordinate convention: hg19, zero-based half-open exon intervals.
_FEATURE = _repair_row("arsenic")
REPAIR_FEATURE_ACCESSION = str(_FEATURE.accession)
REPAIR_FEATURE_CHROM = str(_FEATURE.chrom)
REPAIR_FEATURE_STRAND = str(_FEATURE.strand)
REPAIR_FEATURE_EXONS_0BASED = tuple(
    tuple(map(int, interval.split("-")))
    for interval in _FEATURE.exon_intervals_0based.split(";"))
REPAIR_INPUT_QC_BIN_NT = int(_FEATURE.input_qc_bin_nt)
REPAIR_MIN_CONTROL_INPUT_BINS = int(_FEATURE.min_control_input_bins)
REPAIR_MIN_CONTROL_INPUT_DENSITY = float(_FEATURE.min_control_input_density)

