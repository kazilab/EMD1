"""Nominal and counterfactual integration protocols."""
from dataclasses import replace
import numpy as np
from .model import Intervention, Params, control_steady_state, normalise_params, simulate, simulate_protocol, trace
from .conditions import CONDITIONS, DOSE, SOURCE_PROTOCOL_CONDITIONS
from .settings import COUNTERFACTUAL_SEED

REPAIR_ARSENIC_WRITER_INPUT = {"w_rep_writer": 1.0}

def run_nominal(p: Params, t: np.ndarray, intervention_start: float = 0.0) -> dict:
    p = normalise_params(p)
    y0 = control_steady_state(p)
    runs = {}
    for c in CONDITIONS:
        y = simulate_protocol(p, c.iv, c.E, t, intervention_start, y0)
        tr = trace(y, p, c.iv)
        if intervention_start > 0:
            # A switch at or beyond the endpoint leaves all readouts untreated.
            before = (np.ones(len(t), dtype=bool) if intervention_start >= t[-1]
                      else t < intervention_start)
            pre = trace(y[:, before], p, Intervention())
            for key in tr:
                tr[key][before] = pre[key]
        runs[c.key] = tr
    return runs


def run_source_protocols(p: Params, t: np.ndarray) -> dict:
    """Simulate FTO perturbations on the unexposed comparison background."""
    p = normalise_params(p)
    y0 = control_steady_state(p)
    return {
        c.key: trace(simulate_protocol(p, c.iv, c.E, t, 0.0, y0), p, c.iv)
        for c in SOURCE_PROTOCOL_CONDITIONS
    }

def _cf_fold(q: Params, t: np.ndarray) -> dict:
    """Normalise a counterfactual trajectory to its own unexposed control."""
    y0 = control_steady_state(q)
    ctl = trace(simulate(q, Intervention(), 0.0, t, y0=y0), q, Intervention())
    ars = trace(simulate(q, Intervention(), DOSE, t, y0=y0), q, Intervention())
    out = {}
    for o in ("M_rep", "Q", "N"):
        # N starts at zero in both arms with the same initial fixation rate.
        r = np.ones_like(ars[o], dtype=float)
        np.divide(ars[o], ctl[o], out=r, where=np.abs(ctl[o]) > 1e-12)
        out[o] = r
    return out

def run_counterfactual(p: Params, t: np.ndarray, n: int = 0, seed: int = COUNTERFACTUAL_SEED) -> dict:
    """Apply full arsenic-driven writer input at the repair site.

    Bands span the 5th to 95th percentiles of parameter draws for this scenario.
    Comparisons with cross-transcript variation are descriptive.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    p = normalise_params(p)
    out = _cf_fold(replace(p, **REPAIR_ARSENIC_WRITER_INPUT), t)
    if n:
        from .ensemble import sample_params
        rng = np.random.default_rng(seed)
        draws = {o: [] for o in ("M_rep", "Q")}
        for _ in range(n):
            try:
                d = _cf_fold(replace(sample_params(rng, base=p),
                                     **REPAIR_ARSENIC_WRITER_INPUT), t)
            except Exception:
                continue
            for o in draws:
                draws[o].append(d[o])
        for o, v in draws.items():
            if not v:
                raise RuntimeError("no valid counterfactual draws")
            a = np.asarray(v)
            out[o + "_lo"], out[o + "_hi"] = np.percentile(a, [5, 95], axis=0)
            out[o + "_n"] = len(v)
    return out
