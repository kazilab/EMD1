"""Approximate readings of published panels, identified by paper and figure.

Gels and blots without a quantitative axis are marked qualitative. The
readings lack per-value uncertainty estimates. Gao panels 1G and 3Q report
TP53 substitution-category maxima for descriptive comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def simulation_endpoint_values(panel: str) -> dict[str, float]:
    """Read approximate values for a Gao panel from the bundled database."""
    from .store import read_table

    rows = read_table("digitised_endpoints")
    rows = rows.loc[rows["panel"] == panel]
    if rows.empty or rows["condition"].duplicated().any():
        raise ValueError(f"Missing or duplicated digitised endpoints for panel {panel}")
    values = {str(row.condition): float(row.value) for row in rows.itertuples()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(f"Panel {panel} requires finite positive endpoints")
    return values


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


JBC = "Gao et al., J Biol Chem 298:101563 (2022)  PMID 34998823"

APOBEC3B_ARM = [
    Datum(JBC, "3D", "APOBEC3B m6A level, FTO knockdown",
          {"NC siRNA": 1.0, "FTO siRNA-1": 7.5, "FTO siRNA-2": 36.5},
          "fold vs NC", "bar heights, y-axis 0-50, clear gridlines",
          "FTO knockdown increased APOBEC3B m6A in both siRNA groups. The estimated "
          "fold change differed substantially between groups.",
          {"FTO siRNA-1": 0.0214, "FTO siRNA-2": 0.001}),

    Datum(JBC, "3E", "APOBEC3B mRNA decay constant, FTO knockdown",
          {"NC siRNA": 0.055, "FTO siRNA-1": 0.181, "FTO siRNA-2": 0.183},
          "1/h", "colour-detected curve points at 0/2/4/8 h, slope of ln(pct)",
          "Approximate half-life fell from 12.6 to 3.8 h following FTO knockdown. The "
          "change supports the direction of reader-associated decay. The four time "
          "points provide limited precision for estimating its magnitude."),

    Datum(JBC, "3K", "APOBEC3B mRNA level, FTO inhibition",
          simulation_endpoint_values("3K"),
          "fold vs DMSO", "bar heights, y-axis 0-1.5",
          "FB23-2 reduced APOBEC3B relative to DMSO. The paper does not specify arsenic "
          "exposure for this panel. Inhibitor strength and assay duration lack "
          "calibrated model equivalents.",
          {"FB23-2": 0.0233}),

    Datum(JBC, "3M", "APOBEC3B m6A level, FTO inhibition",
          {"DMSO": 1.0, "FB23-2": 4.7},
          "fold vs DMSO", "bar heights, y-axis 0-6",
          "FTO inhibition increased APOBEC3B m6A, consistent with the siRNA comparison "
          "in Fig. 3D.",
          {"FB23-2": 0.0032}),

    Datum(JBC, "3B", "APOBEC3B mRNA bound by FTO (RIP)",
          {"IgG": 1.0, "FTO": 3.0},
          "fold vs IgG", "bar heights, y-axis 0-4",
          "APOBEC3B mRNA was enriched in the FTO immunoprecipitate.", {"FTO": 0.037}),

    Datum(JBC, "1G", "mutation count by substitution type",
          simulation_endpoint_values("1G"),
          "max count across substitution types",
          "tallest bar per group, y-axis 0-10, integer counts",
          "Largest substitution-category count among cloned TP53 3D-PCR products. An "
          "estimate of total mutation burden or rate would require an assay "
          "denominator."),

    Datum(JBC, "3Q", "mutation count by substitution type, FTO inhibition",
          simulation_endpoint_values("3Q"),
          "max count across substitution types",
          "tallest bar per group, y-axis 0-10, integer counts",
          "The maximum substitution-category count among cloned TP53 3D-PCR products "
          "fell from about 9 to 4. The ODE has no observation model for this statistic."),

    Datum(JBC, "1B / 3G / 3H", "A3B protein, arsenic and FTO manipulation",
          {}, "-", "western blot, no axis",
          "Qualitative blot. Arsenic increased APOBEC3B protein, FTO siRNA suppressed "
          "the increase, and FLAG-APOBEC3B restored it."),

    Datum(JBC, "3I", "A3B deaminase activity, epistasis",
          {}, "-", "activity gel, no axis",
          "Qualitative gel. Under arsenic exposure, FTO siRNA reduced deaminase product "
          "and FLAG-APOBEC3B restored it."),

    Datum(JBC, "3A", "writer/eraser protein vs arsenic dose",
          {}, "-", "western blot, no axis",
          "Qualitative blot. FTO protein increased with arsenic dose (0–2 µM). ALKBH5, "
          "METTL3, METTL14, METTL16 and YTHDF2 showed little change."),
]


NC = "Cui et al., Nat Commun 12:2183 (2021)  PMID 33846348"

NEDD4L_ARM = [
    Datum(NC, "3f", "NEDD4L m6A peak coordinate",
          {}, "-", "IGV track label read directly off the panel",
          "The published hg38 interval is NC_000018.10:58221610-58221760 on "
          "NM_001144966.3. Panels 3h and 3k identify exon 1. Its mapping to the "
          "deposited hg19 tracks remains unresolved."),

    Datum(NC, "3h", "NEDD4L exon-1 m6A, arsenic transformation",
          {"Control": 1.0, "As-T": 0.12},
          "fold vs control", "bar heights, y-axis 0-1.2",
          "Targeted m6A IP-qPCR in the xenograft-derived As-T line. Comparisons with "
          "the coverage tracks require a verified exon-1 mapping and adequate input "
          "coverage.",
          {"As-T": 0.0002}),

    Datum(NC, "3k / 3j", "NEDD4L m6A, FTO knockout",
          {"WT": 1.0, "FTO KO (exon 1)": 2.8, "FTO KO (transcript)": 3.55},
          "fold vs WT", "bar heights, y-axes 0-4 and 0-5",
          "FTO knockout increased NEDD4L m6A at exon 1 and across the transcript.",
          {"exon 1": 0.0003, "transcript": 0.019}),

    Datum(NC, "3e", "NEDD4L mRNA across transformation",
          {"Control": 1.0, "As": 0.68, "As-T": 0.19},
          "fold vs control", "bar heights, y-axis 0-1.5",
          "As and xenograft-derived As-T are distinct experimental conditions. The "
          "cross-sectional comparison has no calibrated model time.",
          {"As": 0.021, "As-T": 0.0006}),

    Datum(NC, "3l", "NEDD4L mRNA, FTO knockout",
          {"WT": 1.0, "FTO KO-1": 1.67, "FTO KO-2": 1.65},
          "fold vs WT", "bar heights, y-axis 0-2.5",
          "Both knockout lines increased NEDD4L mRNA abundance.", {"KO-1": 0.013, "KO-2": 0.006}),

    Datum(NC, "3i / 3m", "NEDD4L mRNA decay",
          {"Control 6h": 0.80, "As-T 6h": 0.60, "WT 6h": 0.47, "FTO KO 6h": 0.79},
          "fraction remaining", "curve points at 0/3/6 h ActD",
          "NEDD4L decayed faster in As-T cells and slower after FTO knockout. APOBEC3B "
          "showed the opposite response in a separate experimental background.",
          {"3i": 0.019, "3m": 0.0125}),
]

GLOBAL_AND_KCC10 = [
    Datum(NC, "1a / 1d / 1h", "global m6A under arsenic",
          {}, "-", "m6A dot blot with methylene-blue loading control, no quantitative axis",
          "The dot blot showed lower signal with increasing arsenic dose (0–0.2 µM), in "
          "the As and As-T lines, and in human arsenical keratosis compared with normal "
          "skin. A numerical fold change cannot be read from this blot.",
          {}),

    Datum(NC, "1e", "proliferation across transformation",
          {"Control": 2.8, "As": 4.2, "As-T": 7.8},
          "fold cell number at endpoint", "curve endpoints, y-axis 0-9",
          "Approximate endpoint cell-number ratios were 1.5 for As/control and 2.8 for "
          "As-T/control. These measurements describe population growth. Clonal "
          "selection and the experimental-to-model time scale remain undetermined.",
          {"As": 0.0116, "As-T": 0.0005}),

    Datum(NC, "1c", "tumour volume, day 34",
          {"Control": 0.0, "As": 1450.0}, "mm^3",
          "curve endpoint, y-axis 0-2000",
          "Replicate measurements are available in the source workbook.",
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
            out.append(f"   {d.note}")
    return out


if __name__ == "__main__":
    print(summary())
