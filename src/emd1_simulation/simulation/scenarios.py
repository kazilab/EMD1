"""Compare structural assumptions about repair-site input and lesion removal."""
from dataclasses import replace
import numpy as np
from .conditions import BY_KEY
from .model import Params, Intervention, control_steady_state, simulate, trace

REPAIR_SCENARIOS = {
    "no_exposure_input": {},
    "partial_writer_input": {"w_rep_writer": 0.5},
    "full_writer_input": {"w_rep_writer": 1.0},
    "fto_sensitive_site": {"w_rep_fto": 1.0},
    "writer_and_fto_input": {"w_rep_writer": 1.0, "w_rep_fto": 1.0},
    "legacy_universal_repair_transfer": {"repair_proxy_weight": 1.0},
    "writer_input_with_assumed_repair_transfer": {
        "w_rep_writer": 1.0, "repair_proxy_weight": 1.0},
}


def repair_scenarios(p: Params, t: np.ndarray) -> dict:
    results = {}
    defaults = Params()
    custom_input = any(getattr(p, name) != getattr(defaults, name)
                       for name in ("w_rep_writer", "w_rep_fto", "a_B"))
    for name, changes in REPAIR_SCENARIOS.items():
        label = "configured_baseline" if name == "no_exposure_input" and custom_input else name
        q = replace(p, **changes)
        y0 = control_steady_state(q)
        ctl = trace(simulate(q, Intervention(), 0., t, y0), q, Intervention())
        arms = {}
        for key in ("as", "as_ftoi", "as_m3kd"):
            c = BY_KEY[key]
            tr = trace(simulate(q, c.iv, c.E, t, y0), q, c.iv)
            arms[key] = {o: float(tr[o][-1] / ctl[o][-1])
                         for o in ("M_rep", "Q", "Q_lesion", "N")}
        results[label] = {"parameters": changes, "endpoint_folds_vs_own_control": arms}
    return {"interpretation": "Unweighted structural scenarios, not confidence or credible intervals. "
            "Q is a DDB2/NER proxy; Q_lesion is an assumed transfer to aggregate repair.",
            "scenarios": results}
