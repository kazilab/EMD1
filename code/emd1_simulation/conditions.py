"""Simulated conditions and the published results each one is checked against.

The four plotted conditions are the ones the design calls for:

    control -> arsenic -> arsenic + FTO inhibition -> arsenic + METTL3 knockdown

The remaining entries are perturbations the arsenic papers actually performed
(ALKBH5 knockdown, antioxidant-reader knockdown, APOBEC3B knockdown, NEDD4L
rescue). None is used to tune a numerical parameter. APOBEC3B knockdown and
NEDD4L rescue have out-of-calibration qualitative sign checks, although their
edge directions come from the same literature and are not blinded prospective
validations. ALKBH5 and antioxidant-reader knockdown are retained as unvalidated
model projections and are not counted as agreement with data.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Intervention, Params

DOSE = 1.0   # dimensionless chronic arsenic dose
# V11: writer KD must accelerate antioxidant decay; model ~1.80, measured ~2.2.
AOX_DECAY_RATIO_MIN = 1.5

# --- "what if arsenic DID drive the repair site?" ------------------------
# ONE change. The edge itself is always live (gam_f1r > 0 in Params); the only
# arsenic-specific claim in the model is w_rep_writer = 0, so flipping that
# single parameter is the whole counterfactual.
#
# The previous version bundled three changes (a_B = 0.6, w_rep_writer = 1,
# gam_f1r = 2) and reported their joint effect as though it isolated the edge.
# It did not: taken singly they give M_rep 1.48x, 1.56x and 1.00x respectively
# against 2.05x combined, and the gam_f1r component moves M_rep not at all. The
# height of that trace was set by how arsenic reached the site, not by the edge
# coupling it was labelled with.
REPAIR_ARSENIC_WRITER_INPUT = {"w_rep_writer": 1.00}

# --- the EMD1 -> KCC3 evidence, at the resolution the mechanism lives at ----
# Reproduce with ``python -m emd1_simulation.fit.repair_edge``.
#
# The formerly deleted edge used to rest on two MODULE-level nulls (an earlier
# MeRIP repair set and a 271-gene m6A microarray set). The current reproducible
# annotation-only pipeline selects 301 repair transcripts with at least three
# exons, of which 258-262 pass control-input QC per contrast. A repair-set
# summary is the wrong resolution for this edge. GSE145924
# -- same lab, same HaCaT line, accessions consecutive with the arsenic series --
# reports a single-transcript mechanism: METTL14 writes m6A on DDB2, YTHDF1
# reads it, and DDB2 drives global genome repair (PMID 34452996). Our deposited-
# track reanalysis does not independently reproduce the reported METTL14-KD
# decrease: on the externally predeclared, annotated first-three-exon feature,
# DDB2 is 1.31x and remains inside its cross-transcript null. The earlier 0.33x
# result was treatment leakage plus de novo peak selection outside that feature.
# The repair-set median remains uninformative (0.97-1.03), including where
# transformed-line DDB2 is 0.39x.
#
# These are first-three-exon DDB2 enrichments, divided by the background median
# of 600 transcripts quantified identically (ratios drift transcriptome-wide
# between conditions, so a raw ratio is not an effect).
# 5th-95th percentile of the effect seen across background transcripts. This is
# a CROSS-TRANSCRIPT null distribution -- how much an arbitrary transcript moves
# in this contrast -- not a measurement-error interval and not a formal
# detection limit. It is the right null for a specificity claim ("does DDB2 move
# more than transcripts in general?") and should not be described as anything
# stronger.
# Each contrast has its OWN background distribution; they are close but not the
# same, and a measured value must be judged against the band from its own
# contrast rather than against the arsenic one.
REPAIR_NULL_BY_CONTRAST = {
    "arsenic":     (0.5438026232668061, 1.8642027242120363),
    "transformed": (0.48604201613716175, 2.1270301588898786),
    "writer_kd":   (0.5375148806488801, 1.8581768348092378),
    "uvb":         (0.5976183974129734, 1.74630172922041),
}
REPAIR_NULL_BAND = REPAIR_NULL_BY_CONTRAST["arsenic"]
# Transcripts actually quantified per contrast after per-control-replicate input
# QC. 600 background transcripts were sampled; these are the retained counts.
REPAIR_N_QUANTIFIED = {"arsenic": 336, "writer_kd": 334,
                       "transformed": 334, "uvb": 336}
REPAIR_N_REPAIR_QUANTIFIED = {"arsenic": 262, "writer_kd": 258,
                              "transformed": 261, "uvb": 258}
REPAIR_DDB2 = {
    # label:              (effect, background percentile)
    "arsenic":            (0.9083368458716881, 37.797619047619044),
    "writer_kd":          (1.3137988452814229, 78.1437125748503),
    "transformed":        (0.3896584042785915, 2.3952095808383236),
    "uvb":                (0.9531399619179151, 42.55952380952381),
}
# XPC is a same-pathway specificity control. It moves with DDB2 under both
# chronic arsenic and METTL14 knockdown, reinforcing that neither deposited-
# track contrast shows a transcript-specific DDB2 outlier. It diverges from
# DDB2 in the arsenic-transformed line, where DDB2 falls below its null.
REPAIR_XPC = {"arsenic": 0.9554652495028385,
              "writer_kd": 1.589504615700011,
              "transformed": 0.7206576588193051,
              "uvb": 1.0605404883864882}
# Repair-set median, flat in every contrast and therefore uninformative about a
# single-transcript mechanism.
REPAIR_MODULE = {"arsenic": 1.0103423552487114,
                 "writer_kd": 0.966458614525688,
                 "transformed": 1.0268016973771406,
                 "uvb": 0.9923436455332798}
REPAIR_N_BACKGROUND = 600

# Exact DDB2 feature and eligibility rule used by fit/repair_edge.py.  Keeping
# these in the simulation provenance makes the generated JSON self-contained;
# the regression test verifies they remain identical to the reproduced CSV.
REPAIR_FEATURE_ACCESSION = "NM_000107"
REPAIR_FEATURE_CHROM = "chr11"
REPAIR_FEATURE_STRAND = "+"
REPAIR_FEATURE_EXONS_0BASED = (
    (47236524, 47236814),
    (47237886, 47238023),
    (47238408, 47238600),
)
REPAIR_INPUT_QC_BIN_NT = 100
REPAIR_MIN_CONTROL_INPUT_BINS = 3
REPAIR_MIN_CONTROL_INPUT_DENSITY = 1.0


def antioxidant_decay_rate(M_aox: float, p: Params | None = None,
                           aox_reader: float = 1.0) -> float:
    """Effective first-order decay constant for the antioxidant pool."""
    p = p or Params()
    return p.d_aox / (1.0 + p.gam_ia * aox_reader * M_aox)


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    E: float
    iv: Intervention
    plotted: bool = False


CONDITIONS = [
    Condition("control", "Control", 0.0, Intervention(), plotted=True),
    Condition("as", "Arsenic", DOSE, Intervention(), plotted=True),
    Condition("as_ftoi", "Arsenic + FTO inhibition", DOSE,
              Intervention(fto=0.40), plotted=True),
    Condition("as_m3kd", "Arsenic + METTL3 knockdown", DOSE,
              Intervention(mettl3=0.35), plotted=True),
    # --- held out of the figure; two validated, two unvalidated projections ---
    Condition("as_a5kd", "Arsenic + ALKBH5 knockdown", DOSE,
              Intervention(alkbh5=0.30)),
    Condition("as_aoxkd", "Arsenic + antioxidant-reader knockdown", DOSE,
              Intervention(aox_reader=0.30)),
    Condition("as_a3bkd", "Arsenic + APOBEC3B knockdown", DOSE,
              Intervention(a3b_expr=0.30)),
    Condition("as_nedresc", "Arsenic + NEDD4L rescue", DOSE,
              Intervention(ned_expr=2.5)),
]

BY_KEY = {c.key: c for c in CONDITIONS}
PLOTTED = [c for c in CONDITIONS if c.plotted]
OUT_OF_CALIBRATION_CHECK_KEYS = frozenset({"as_a3bkd", "as_nedresc"})


# --- model-check register -------------------------------------------------
# Each check is (id, edge, description, callable(endpoints) -> bool, source).
# `endpoints` maps condition key -> observable -> value at the end of the run,
# on the model's native scale. Checks form their own arm ratios/differences.

@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


# Passing a check means the implementation reproduces the stated constraint;
# it does not automatically mean independent validation. V5 and V9 use
# interventions withheld from numerical parameter tuning and covered by
# external data, but their signs are biologically encoded from that literature.
CHECK_ROLES = {
    **{f"V{i}": "calibration/encoding consistency" for i in (1, 2, 3, 4, 7, 10, 11)},
    "V5": "out-of-calibration qualitative check",
    "V6": "structural regression",
    "V8": "data concordance",
    "V9": "out-of-calibration qualitative check",
    "V12": "data concordance",
    "V13": "structural regression",
}


def _gt(a, b, margin=1.05):
    return a > b * margin


def _lt(a, b, margin=0.95):
    return a < b * margin


CHECKS = [
    Check("V1", "KCC5 -> EMD1",
          "Arsenic raises ROS and FTO activity",
          lambda e: _gt(e["as"]["R"], e["control"]["R"]) and _gt(e["as"]["Fo"], e["control"]["Fo"]),
          "As-induced ROS; FTO protein stabilised by impaired autophagy (NC 6, JBC 3A)"),

    Check("V2", "EMD1 (site-specific)",
          "At nominal calibration the four m6A sites do NOT move together: A3B "
          "and NEDD4L fall while the antioxidant module rises; the three-sign "
          "conjunction is audited separately across stress-test draws",
          lambda e: (_lt(e["as"]["M_a3b"], e["control"]["M_a3b"])
                     and _lt(e["as"]["M_ned"], e["control"]["M_ned"])
                     and _gt(e["as"]["M_aox"], e["control"]["M_aox"])),
          "prespecified published exon-1 NEDD4L interval 0.61x; m6A microarray antioxidant "
          "core 1.28x over platform baseline"),

    Check("V3", "EMD1 -> KCC2",
          "Arsenic raises APOBEC3B and mutational burden",
          lambda e: _gt(e["as"]["A3B"], e["control"]["A3B"]) and _gt(e["as"]["N"], e["control"]["N"]),
          "JBC 1B/1G: As raises A3B protein and mutation count 2 -> 8"),

    Check("V4", "EMD1 -> KCC2 (rescue)",
          "FTO inhibition reverses the A3B rise and lowers mutational burden",
          lambda e: _lt(e["as_ftoi"]["A3B"], e["as"]["A3B"]) and _lt(e["as_ftoi"]["N"], e["as"]["N"]),
          "JBC 3K: A3B mRNA 0.12x under FB23-2; JBC 3Q: mutations 9 -> 4"),

    Check("V5", "EMD1 -> KCC2 (epistasis)",
          "APOBEC3B knockdown lowers burden without lowering ROS",
          lambda e: (_lt(e["as_a3bkd"]["N"], e["as"]["N"])
                     and not _lt(e["as_a3bkd"]["R"], e["as"]["R"])),
          "JBC 1G: A3B shRNA returns mutation count toward baseline; JBC 3I rescue"),

    Check("V6", "EMD1 -> KCC3  (ASSUMED ZERO ARSENIC INPUT)",
          "The DDB2/NER proxy stays flat under the nominal zero-input assumption. "
          "The edge is not absent -- V13 shows it responds to a writer "
          "perturbation -- and the measurement cannot exclude a moderate effect",
          lambda e: abs(e["as"]["Q"] / e["control"]["Q"] - 1.0) < 0.05,
          "First-three-exon DDB2 sits at 0.91x, background percentile 38 among "
          "336 quantified transcripts, with the same-pathway control XPC also "
          "inside the null. The repair-set "
          "median this check originally rested on had no power: it stays "
          "within 0.97-1.03 even where transformed-line DDB2 is 0.39x. See "
          "fit/repair_edge.py."),

    Check("V7", "EMD1 -> KCC10",
          "At nominal calibration arsenic lowers NEDD4L, de-represses WNT and "
          "raises clonal expansion; the expansion sign is sensitivity-dependent",
          lambda e: (_lt(e["as"]["NEDD4L"], e["control"]["NEDD4L"])
                     and _gt(e["as"]["WNT"], e["control"]["WNT"])
                     and e["as"]["lnP"] > e["control"]["lnP"]),
          "NC 3e: NEDD4L 0.68x; NC 1e: proliferation 1.5x"),

    Check("V8", "EMD1 -> KCC10 (rescue)",
          "FTO inhibition restores NEDD4L and suppresses expansion",
          lambda e: (_gt(e["as_ftoi"]["NEDD4L"], e["as"]["NEDD4L"])
                     and e["as_ftoi"]["lnP"] < e["as"]["lnP"]),
          "NC 3l: NEDD4L 1.66x in FTO KO; NC Fig 2: FTO KO reduces tumour growth"),

    Check("V9", "EMD1 -> KCC10 (independent handle)",
          "Direct NEDD4L re-expression suppresses expansion with FTO untouched",
          lambda e: e["as_nedresc"]["lnP"] < e["as"]["lnP"],
          "NC Fig 4: NEDD4L rescue"),

    Check("V10", "feedback loop",
          "At nominal calibration m6A-dependent antioxidant adaptation holds the "
          "redox balance and writer disruption lowers expansion; the writer-KD "
          "expansion direction is not robust across the parameter stress test",
          lambda e: (_gt(e["as"]["A"], e["control"]["A"])
                     and _lt(e["as_m3kd"]["A"], e["as"]["A"])
                     and e["as_m3kd"]["lnP"] < e["as"]["lnP"]),
          "Zhao Fig 5B/S5b/S5c: siMETTL3 0.61-0.65x and STM2457 0.58-0.66x "
          "antioxidant level, by two independent handles"),

    Check("V11", "EMD1 -> antioxidant (mechanism)",
          "m6A STABILISES the antioxidant transcripts: writer knockdown "
          "raises their effective decay rate by at least "
          f"{AOX_DECAY_RATIO_MIN:.1f}x (model ~1.8x; measured ~2.2x)",
          lambda e: (
              antioxidant_decay_rate(e["as_m3kd"]["M_aox"])
              / antioxidant_decay_rate(e["as"]["M_aox"])
              >= AOX_DECAY_RATIO_MIN
          ),
          "Zhao Fig 5B chase: siMETTL3 raises the decay constant ~2.2x in "
          "16 of 18 gene x stage cells. This is why the coupling is a decay "
          "term and not the translation gain first assumed."),

    Check("V12", "reader-specific divergence (CENTRAL CLAIM)",
          "One FTO perturbation moves APOBEC3B and NEDD4L in OPPOSITE "
          "directions, because a decay reader binds one and a stabilising "
          "reader the other",
          lambda e: ((e["as_ftoi"]["M_a3b"] > e["as"]["M_a3b"])
                     and (e["as_ftoi"]["M_ned"] > e["as"]["M_ned"])
                     and (e["as_ftoi"]["A3B"] < e["as"]["A3B"])
                     and (e["as_ftoi"]["NEDD4L"] > e["as"]["NEDD4L"])),
          "Both sides now measured directly: JBC 3E (FTO KD -> 3.3x FASTER "
          "A3B decay) vs NC 3m (FTO KO -> SLOWER NEDD4L decay). Same "
          "perturbation, opposite outcome. Opposite signs alone do not exclude a shared scalar with reader-specific responses."),

    # V13 is not part of CHECKS because run.py supplies the direct writer-KD
    # outputs separately to check_repair_edge after computing fold changes.
]


def ordinal(n: float) -> str:
    """1 -> 1st, 42 -> 42nd. A plain "%dth" reads as "3th" and "42th"."""
    i = int(round(n))
    suf = "th" if 11 <= i % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def check_repair_edge(m_rep_as: float, q_as: float,
                      m_rep_wkd: float, q_wkd: float) -> tuple[bool, str]:
    """V13: the DDB2/NER edge is live; nominal arsenic writer input is zero.

    This tests the simulation, on quantities the simulation computes. An earlier
    version compared two hard-coded observations to each other and returned True
    for any model output whatsoever -- including 0 and 999 -- so it validated
    nothing while being counted among the checks that passed.

      (a) EDGE LIVE. A direct writer perturbation must move the repair site and
          the DDB2/NER proxy Q. If it does not, the edge has been deleted rather
          than retained with assumed-zero arsenic input, which is what
          `gam_f1r = 0` used to do: it pinned Q at 1.000 even in arms where
          M_rep moved by a factor of two.
      (b) ARSENIC INPUT ASSUMED ZERO. The arsenic arm must leave Q flat and the
          repair site inside the cross-transcript DDB2 null. That is a check of
          the encoded assumption `w_rep_writer = 0`, not a detection of
          biological non-engagement.

    WHAT THIS IS AND IS NOT. It is a structural consistency check: it confirms
    the edge is wired up and that the nominal exposure does not drive it, and it
    fails if either half is broken. The 0.95 thresholds are conventional, not
    derived. It does NOT test the counterfactual, and it does not test
    quantitative agreement with a matched data contrast. Most importantly it is
    not evidence that arsenic leaves this site alone: the counterfactual in
    which arsenic DOES drive the site substantially overlaps the measured
    specificity band, so the data cannot cleanly separate the two.

    The published METTL14-DDB2-YTHDF1 mechanism supports retaining the coupling,
    but the first-three-exon reanalysis of the deposited tracks does not reproduce
    a DDB2 decrease under METTL14 knockdown (1.31x, inside its null, and XPC also
    inside). The modelled direct-writer response is therefore a structural
    implication, not a validation against that reanalysis or an aggregate DNA-repair assay.
    """
    lo, hi = REPAIR_NULL_BAND
    present = (m_rep_wkd < 0.95) and (q_wkd < 0.95)
    arsenic_input_assumed_zero = (lo <= m_rep_as <= hi) and abs(q_as - 1.0) < 0.05
    return present and arsenic_input_assumed_zero, (
        f"edge live: writer knockdown moves the site to {m_rep_wkd:.2f}x and the "
        f"DDB2/NER proxy Q to {q_wkd:.2f}x; arsenic input assumed zero: "
        f"arsenic leaves the site at {m_rep_as:.2f}x (inside the measured "
        f"cross-transcript null [{lo:.2f}, {hi:.2f}]) and Q at {q_as:.2f}x. "
        f"Structural check of the encoding, not a detection of biological "
        f"non-engagement -- the null is wide enough to contain the "
        f"counterfactual too")
