"""Sensitivity analysis using independent log-normal parameter perturbations.

The assumed perturbation widths describe sensitivity around the selected
parameters. Ensemble percentiles have no sampling or posterior interpretation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import Params, control_steady_state, normalise_params, simulate, trace

# Log-scale standard deviations. Reader identities and coupling signs are fixed.
FREE_SIGMA = {
    "a_W": 0.30, "a_F": 0.30,
    "K_W": 0.25, "K_F": 0.25,
    "tau_W": 0.30, "tau_F": 0.30, "tau_M": 0.30,
    "gam_d2": 0.35, "gam_i2": 0.35, "gam_ia": 0.35,
    # The DDB2 translation gain has a wider assumed range than the other reader
    # gains because its magnitude lacks a matched measurement.
    "gam_f1r": 0.50,
    "M0_a3b": 0.20, "M0_ned": 0.20, "M0_rep": 0.20, "M0_aox": 0.20,
    "w_aox_fto": 0.30, "nrf2": 0.30,
    "k_ROS": 0.30, "k_A3B": 0.30, "k_rep": 0.30, "K_L": 0.30,
    "k_fix": 0.30,
    "k_E": 0.25, "k_A": 0.25,
    "K_N": 0.15, "h_N": 0.20, "K_Rd": 0.12,
    "k_prol": 0.20, "k_apo": 0.30, "k_dam": 0.30,
}

# With zero chronic ALKBH5 response, K_B and tau_B have no effect. The basal
# death rate d0 cancels in each population contrast against control.
def _renormalise(p: Params) -> Params:
    """Recalculate synthesis rates and the resting ROS threshold for each draw."""
    return normalise_params(p)


def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    base = base or Params()
    draw = {}
    for name, sigma in FREE_SIGMA.items():
        val = getattr(base, name) * float(np.exp(rng.normal(0.0, sigma)))
        if name.startswith("M0_") or name == "w_aox_fto":
            val = float(np.clip(val, 0.05, 0.90))   # occupancies/weights are fractions
        draw[name] = val
    return _renormalise(replace(base, **draw))


def run_ensemble(conditions, observables, t_eval, n: int = 400, seed: int = 20260822,
                 audit_checks: bool = False, base: Params | None = None,
                 check_evaluator=None, check_roles: dict | None = None):
    """Return {condition: {observable: array (n_ok, n_t)}} plus the accepted count.

    Each draw is centred on the supplied parameters and normalised to its own
    unexposed control. Optional checks evaluate each draw on its native scale.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer; got {n}")

    if (check_evaluator is None) != (check_roles is None):
        raise ValueError("Provide check_evaluator and check_roles together")
    if not audit_checks and check_evaluator is not None:
        raise ValueError("Enable audit_checks when supplying a check evaluator")
    rng = np.random.default_rng(seed)
    acc = {c.key: {o: [] for o in observables} for c in conditions}
    audit_counts = None
    if audit_checks:
        if check_evaluator is None or check_roles is None:
            from ..analysis.checks import CHECKS, CHECK_ROLES, evaluate_draw_checks
            check_evaluator = check_evaluator or evaluate_draw_checks
            check_roles = check_roles or {
                cid: CHECK_ROLES[cid] for cid in (*[check.cid for check in CHECKS], "V13")}
        audit_counts = {cid: 0 for cid in check_roles}
    ok = 0
    for _ in range(n):
        p = sample_params(rng, base=base)
        try:
            y0 = control_steady_state(p)
            runs = {c.key: trace(simulate(p, c.iv, c.E, t_eval, y0=y0), p, c.iv)
                    for c in conditions}
        except Exception:
            continue
        ctl = runs["control"]
        bad = False
        for c in conditions:
            for o in observables:
                v = runs[c.key][o]
                if o == "lnP":
                    val = (v - ctl["lnP"]) / np.log(2.0)
                else:
                    ref = ctl[o]
                    val = np.ones_like(v, dtype=float)
                    np.divide(v, ref, out=val, where=np.abs(ref) > 1e-12)
                if not np.all(np.isfinite(val)):
                    bad = True
                acc[c.key][o].append(val)
        if bad:
            for c in conditions:
                for o in observables:
                    acc[c.key][o].pop()
            continue
        ok += 1
        if audit_checks:
            scores = check_evaluator(p, runs)
            for cid in audit_counts:
                audit_counts[cid] += int(bool(scores.get(cid, False)))

    arrays = {k: {o: np.asarray(v) for o, v in d.items()} for k, d in acc.items()}
    if ok == 0:
        raise RuntimeError("no valid ensemble draws; no sensitivity band can be reported")
    if not audit_checks:
        return arrays, ok
    audit = {
        "interpretation": (
            "Fraction of independent parameter stress-test draws retaining each "
            "nominal model-check sign; not a validation probability, confidence "
            "level, or posterior probability."),
        "draws": ok,
        "checks": {
            cid: {
                "role": check_roles[cid],
                "passes": count,
                "pass_fraction": count / ok if ok else None,
            }
            for cid, count in audit_counts.items()
        },
    }
    return arrays, ok, audit
