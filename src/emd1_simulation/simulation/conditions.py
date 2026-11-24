"""Exposure and intervention arms."""
from dataclasses import dataclass
from .model import Intervention

DOSE = 1.0  # dimensionless chronic arsenic input

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
    # Additional hypothesis probes; source protocols and magnitudes are unmatched.
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
# Deprecated arm grouping.
OUT_OF_CALIBRATION_CHECK_KEYS = frozenset({"as_a3bkd", "as_nedresc"})

# Gao Figures 3E and 3K do not specify arsenic exposure. Their comparators
# use zero exposure. Timing, intervention strength and cell state are unmatched.
SOURCE_PROTOCOL_CONDITIONS = (
    Condition("control", "Unexposed source-context control", 0.0, Intervention()),
    Condition("ftoi", "Unexposed FTO inhibition", 0.0, Intervention(fto=0.40)),
    Condition("fto_complete_loss", "Unexposed complete FTO loss bound probe", 0.0,
              Intervention(fto=0.0)),
)

