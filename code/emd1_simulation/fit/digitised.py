"""Values digitised from published figure panels.

These are the endpoints that exist only as figure panels — no source data, no
GEO deposit. Everything here was read off published figures, so treat the
numbers as approximate: bar heights and curve points are accurate to roughly
+/-10% of the axis range, and anything marked `qualitative` is a gel or blot
with no axis at all.

Provenance for each entry is the paper and panel it came from, so any value can
be re-checked against the figure.

Source figures live in ``data/papers/figs/`` (extracted from the JBC supplement).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Datum:
    paper: str
    panel: str
    quantity: str
    values: dict          # condition -> value
    unit: str
    method: str           # how it was read
    note: str = ""
    p: dict = field(default_factory=dict)


JBC = "Wei et al., J Biol Chem 298:101563 (2022)  PMID 34998823"

APOBEC3B_ARM = [
    Datum(JBC, "3D", "APOBEC3B m6A level, FTO knockdown",
          {"NC siRNA": 1.0, "FTO siRNA-1": 7.5, "FTO siRNA-2": 36.5},
          "fold vs NC", "bar heights, y-axis 0-50, clear gridlines",
          "Direct evidence that FTO is the A3B eraser: knocking it down raises "
          "A3B m6A by 7.5-36x. Two independent siRNAs, same direction, very "
          "different magnitude -- treat the magnitude as order-of-magnitude only.",
          {"FTO siRNA-1": 0.0214, "FTO siRNA-2": 0.001}),

    Datum(JBC, "3E", "APOBEC3B mRNA decay constant, FTO knockdown",
          {"NC siRNA": 0.055, "FTO siRNA-1": 0.181, "FTO siRNA-2": 0.183},
          "1/h", "colour-detected curve points at 0/2/4/8 h, slope of ln(pct)",
          "FTO loss -> more m6A on A3B -> ~3.3x FASTER decay, i.e. m6A "
          "destabilises A3B. This is the YTHDF2 coupling the model calls gam_d2, "
          "measured directly. Half-life 12.6 h -> ~3.8 h. A multiplicative "
          "calibration error shifts the intercept, not the slope, so k is robust "
          "even though the digitised percentages are not."),

    Datum(JBC, "3K", "APOBEC3B mRNA level, FTO inhibition",
          {"DMSO": 1.0, "FB23-2": 0.12},
          "fold vs DMSO", "bar heights, y-axis 0-1.5",
          "Pharmacological confirmation of 3D/3E: inhibiting FTO collapses A3B.",
          {"FB23-2": 0.0233}),

    Datum(JBC, "3M", "APOBEC3B m6A level, FTO inhibition",
          {"DMSO": 1.0, "FB23-2": 4.7},
          "fold vs DMSO", "bar heights, y-axis 0-6",
          "Same direction as 3D by a second, pharmacological handle.",
          {"FB23-2": 0.0032}),

    Datum(JBC, "3B", "APOBEC3B mRNA bound by FTO (RIP)",
          {"IgG": 1.0, "FTO": 3.0},
          "fold vs IgG", "bar heights, y-axis 0-4",
          "FTO physically binds the A3B transcript.", {"FTO": 0.037}),

    Datum(JBC, "1G", "mutation count by substitution type",
          {"Ctrl shRNA": 2, "Ctrl shRNA +As": 8, "A3B shRNA +As": 3},
          "max count across substitution types",
          "tallest bar per group, y-axis 0-10, integer counts",
          "KCC2 endpoint. Arsenic raises the mutation load ~4x; knocking down "
          "A3B returns it most of the way to baseline, placing A3B downstream "
          "of arsenic in the mutagenesis chain."),

    Datum(JBC, "3Q", "mutation count by substitution type, FTO inhibition",
          {"DMSO +As": 9, "FB23-2 +As": 4},
          "max count across substitution types",
          "tallest bar per group, y-axis 0-10, integer counts",
          "FTO inhibition roughly halves arsenic-induced mutations -- the "
          "KCC2 rescue the model predicts for its as_ftoi arm."),

    Datum(JBC, "1B / 3G / 3H", "A3B protein, arsenic and FTO manipulation",
          {}, "-", "western blot, no axis",
          "qualitative: As raises A3B protein; FTO siRNA abolishes the rise; "
          "FLAG-A3B re-expression restores it."),

    Datum(JBC, "3I", "A3B deaminase activity, epistasis",
          {}, "-", "activity gel, no axis",
          "qualitative: under As, FTO siRNA reduces deaminase product and "
          "FLAG-A3B rescues it -- A3B is downstream of FTO."),

    Datum(JBC, "3A", "writer/eraser protein vs arsenic dose",
          {}, "-", "western blot, no axis",
          "qualitative: FTO protein rises with As dose (0-2 uM) while ALKBH5, "
          "METTL3, METTL14, METTL16 and YTHDF2 are comparatively flat. This is "
          "the protein-level evidence for the model's Fo induction, and it is "
          "why FTO mRNA being flat in GSE145923 is not a contradiction."),
]


NC = "Cui et al., Nat Commun 12:2183 (2021)  PMID 33846348"

NEDD4L_ARM = [
    Datum(NC, "3f", "NEDD4L m6A peak coordinate",
          {}, "-", "IGV track label read directly off the panel",
          "NC_000018.10:58221610-58221760 (GRCh38), transcript NM_001144966.3, "
          "and panels 3h/3k label it 'm6A on Exon 1'. Lifted to hg19 this is "
          "chr18:~55,712,458 -- the same window the bigwig extraction found at "
          "+1.0 kb. The 5'-proximal location I flagged as atypical is the "
          "paper's own functional site, so that caveat is withdrawn."),

    Datum(NC, "3h", "NEDD4L exon-1 m6A, arsenic transformation",
          {"Control": 1.0, "As-T": 0.12},
          "fold vs control", "bar heights, y-axis 0-1.2",
          "Targeted m6A IP-qPCR. Panel 3f shows the peak present in control and "
          "absent in As-T. My bigwig extraction gave 0.91 for As-T, which is "
          "wrong: NEDD4L mRNA collapses to 0.19x there, so the IP/input ratio "
          "is built on almost no coverage. Trust the targeted assay.",
          {"As-T": 0.0002}),

    Datum(NC, "3k / 3j", "NEDD4L m6A, FTO knockout",
          {"WT": 1.0, "FTO KO (exon 1)": 2.8, "FTO KO (transcript)": 3.55},
          "fold vs WT", "bar heights, y-axes 0-4 and 0-5",
          "FTO removal raises NEDD4L m6A, confirming FTO as the eraser.",
          {"exon 1": 0.0003, "transcript": 0.019}),

    Datum(NC, "3e", "NEDD4L mRNA across transformation",
          {"Control": 1.0, "As": 0.68, "As-T": 0.19},
          "fold vs control", "bar heights, y-axis 0-1.5",
          "The model assumes 0.74 for its arsenic arm; the As stage measures "
          "0.68. Close. As-T is far lower, but that line is xenograft-derived.",
          {"As": 0.021, "As-T": 0.0006}),

    Datum(NC, "3l", "NEDD4L mRNA, FTO knockout",
          {"WT": 1.0, "FTO KO-1": 1.67, "FTO KO-2": 1.65},
          "fold vs WT", "bar heights, y-axis 0-2.5",
          "Two independent knockouts agree.", {"KO-1": 0.013, "KO-2": 0.006}),

    Datum(NC, "3i / 3m", "NEDD4L mRNA decay",
          {"Control 6h": 0.80, "As-T 6h": 0.60, "WT 6h": 0.47, "FTO KO 6h": 0.79},
          "fraction remaining", "curve points at 0/3/6 h ActD",
          "m6A stabilises NEDD4L: losing it (As-T) speeds decay, removing the "
          "eraser (FTO KO) slows it. Opposite sign to APOBEC3B, from the same "
          "FTO event -- the reader-specific divergence the model is built on.",
          {"3i": 0.019, "3m": 0.0125}),
]

GLOBAL_AND_KCC10 = [
    Datum(NC, "1a / 1d / 1h", "global m6A under arsenic",
          {}, "-", "m6A dot blot with methylene-blue loading -- NO axis",
          "Semi-quantitative only. Signal falls with As dose (0/0.1/0.2 uM), is "
          "lower in As and As-T lines, and is lower in human arsenical keratosis "
          "than normal skin. Direction is clear and consistent; magnitude is not "
          "recoverable from a dot blot. The NAR paper's LC-QqQ-MS/MS values are "
          "behind Oxford's CDN and were not retrievable here.",
          {}),

    Datum(NC, "1e", "proliferation across transformation",
          {"Control": 2.8, "As": 4.2, "As-T": 7.8},
          "fold cell number at endpoint", "curve endpoints, y-axis 0-9",
          "KCC10 endpoint. As/Control = 1.5x, As-T/Control = 2.8x. The model's "
          "arsenic arm predicts log2 +1.39, i.e. 2.6x -- it overstates the "
          "chronic-As stage and roughly matches the transformed stage.",
          {"As": 0.0116, "As-T": 0.0005}),

    Datum(NC, "1c", "tumour volume, day 34",
          {"Control": 0.0, "As": 1450.0}, "mm^3",
          "curve endpoint, y-axis 0-2000",
          "Per-replicate values for this are already in the Source Data "
          "workbook, so the digitised reading is only a cross-check.",
          {"day 34": 0.021}),
]


def summary() -> str:
    out = []
    for title, group in [("APOBEC3B / KCC2 arm", APOBEC3B_ARM),
                         ("NEDD4L / KCC10 arm", NEDD4L_ARM),
                         ("global m6A and KCC10 endpoints", GLOBAL_AND_KCC10)]:
        out += ["", "=" * 78, f"Digitised figure endpoints -- {title}", "=" * 78]
        out += _fmt(group)
    return "\n".join(out)


def _fmt(group) -> list:
    out = []
    for d in group:
        out.append(f"\n[{d.panel}] {d.quantity}")
        if d.values:
            vs = "  ".join(f"{k} = {v:g}" for k, v in d.values.items())
            out.append(f"   {vs}   ({d.unit})")
            if d.p:
                out.append("   p: " + ", ".join(f"{k} {v}" for k, v in d.p.items()))
        out.append(f"   read: {d.method}")
        if d.note:
            for line in d.note.split(". "):
                if line.strip():
                    out.append(f"   - {line.strip().rstrip('.')}.")
    return out


if __name__ == "__main__":
    print(summary())
