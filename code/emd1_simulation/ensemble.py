"""Uncertainty propagation.

The public datasets differ in cell type, dose and duration, MeRIP-seq reports
relative regional enrichment rather than absolute site occupancy, and no single
study spans the whole chain. Point estimates of these gains would therefore be
false precision. Instead the free effect sizes receive independent log-normal
stress-test perturbations and qualitative predictions are reported as heuristic
sensitivity bands: a prediction only counts if it survives the spread. These
draws are not a fitted joint distribution, posterior, confidence interval, or
credible interval. In particular, the five lesion parameters were calibrated
jointly but are perturbed independently here, so many draws no longer reproduce
both calibration ratios closely.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import Params, control_steady_state, normalise_params, simulate, trace

# Free effect sizes and their log-scale sigma. Structural choices (which reader
# binds which transcript, the sign of each edge) are NOT sampled -- those are
# the model's hypothesis, not its uncertainty.
FREE_SIGMA = {
    # a_B is deliberately absent: its chronic-arsenic value is the structural
    # choice zero, and multiplying zero by a log-normal draw never creates
    # uncertainty. Acute-ROS simulations should set a_B explicitly instead.
    "a_W": 0.30, "a_F": 0.30,
    "K_W": 0.25, "K_F": 0.25,
    "tau_W": 0.30, "tau_F": 0.30, "tau_M": 0.30,
    "gam_d2": 0.35, "gam_i2": 0.35, "gam_ia": 0.35,
    # gam_f1r is sampled WIDER than the other reader gains. Its existence is
    # supported by published perturbation/rescue experiments but not calibrated
    # by the deposited-track reanalysis; its magnitude is therefore assumed. The
    # counterfactual Q runs from 1.02x at gam_f1r = 0.1 to 1.42x at 10. Leaving
    # it fixed reported that assumption as certainty -- the METTL3-knockdown Q
    # interval came out at [0.76, 0.82], which is the width of everything
    # EXCEPT the one parameter that sets it.
    "gam_f1r": 0.50,
    "M0_a3b": 0.20, "M0_ned": 0.20, "M0_rep": 0.20, "M0_aox": 0.20,
    "w_aox_fto": 0.30, "nrf2": 0.30,
    # All five lesion-calibration parameters are jointly underidentified from
    # two ratios. Give K_L the same moderate log-scale uncertainty as the four
    # rate/flux parameters rather than silently treating it as known.
    "k_ROS": 0.30, "k_A3B": 0.30, "k_rep": 0.30, "K_L": 0.30,
    "k_fix": 0.30,
    "k_E": 0.25, "k_A": 0.25,
    "K_N": 0.15, "h_N": 0.20, "K_Rd": 0.12,
    "k_prol": 0.20, "k_apo": 0.30, "k_dam": 0.30,
}

# K_B and tau_B are absent for the same reason as a_B: with the chronic-
# arsenic ALKBH5 amplitude fixed at zero, neither can affect any trajectory.
# d0 is also absent because every reported clonal-expansion trajectory is a
# within-draw difference from control, so the shared constant turnover cancels
# exactly. Sampling any of these three would consume random draws and inflate
# the advertised parameter count without adding output uncertainty.

# s_a3b / s_ned / s_aox exist only to normalise control abundance to 1; they
# must be recomputed from the sampled occupancies and reader gains, never
# sampled. Missing s_aox here would make every draw start off its own fixed
# point and turn a normalisation artefact into apparent uncertainty.
def _renormalise(p: Params) -> Params:
    # R_ref is derived, not fixed. The effectors respond to ROS in EXCESS of the
    # resting level, so the threshold has to track each draw's own control fixed
    # point. At E = 0 the antioxidant pool is 1 by normalisation, so that point
    # is r_basal / (k_R + k_A) -- which is 0.30 at the nominal values, exactly
    # where R_ref was pinned.
    #
    # Holding it at 0.30 while sampling k_A pushed 191 of 400 control arms above
    # their own threshold, so half the ensemble began with the writer and eraser
    # already switched on before any exposure. The normalisation still held, but
    # "control" was no longer a resting state and the arsenic effect was being
    # measured from a partly activated baseline.
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
                 audit_checks: bool = False):
    """Return {condition: {observable: array (n_ok, n_t)}} plus the accepted count.

    Each draw gets its own control fixed point and its own normalisation, so a
    band reflects uncertainty in the *effect*, not in where the baseline sat.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer; got {n}")

    rng = np.random.default_rng(seed)
    acc = {c.key: {o: [] for o in observables} for c in conditions}
    audit_counts = None
    if audit_checks:
        from .conditions import CHECKS, CHECK_ROLES
        audit_counts = {cid: 0 for cid in (*[chk.cid for chk in CHECKS], "V13")}
    ok = 0
    for _ in range(n):
        p = sample_params(rng)
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
                    # elementwise against the control trajectory; see figure._ratio
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
            # Evaluate checks on the native endpoint scale and with this draw's
            # own Params. V11's nominal callable intentionally defaults to the
            # nominal Params, so its ensemble form is written explicitly here.
            from .conditions import (AOX_DECAY_RATIO_MIN, CHECKS,
                                     antioxidant_decay_rate, check_repair_edge)
            endpoints = {
                key: {o: float(values[o][-1]) for o in values}
                for key, values in runs.items()
            }
            for chk in CHECKS:
                try:
                    if chk.cid == "V11":
                        passed = (
                            antioxidant_decay_rate(
                                endpoints["as_m3kd"]["M_aox"], p)
                            / antioxidant_decay_rate(endpoints["as"]["M_aox"], p)
                            >= AOX_DECAY_RATIO_MIN)
                    else:
                        passed = bool(chk.test(endpoints))
                except Exception:
                    passed = False
                audit_counts[chk.cid] += int(passed)
            try:
                ctl_m = endpoints["control"]["M_rep"]
                ctl_q = endpoints["control"]["Q"]
                v13, _ = check_repair_edge(
                    endpoints["as"]["M_rep"] / ctl_m,
                    endpoints["as"]["Q"] / ctl_q,
                    endpoints["as_m3kd"]["M_rep"] / ctl_m,
                    endpoints["as_m3kd"]["Q"] / ctl_q)
            except Exception:
                v13 = False
            audit_counts["V13"] += int(v13)

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
                "role": CHECK_ROLES[cid],
                "passes": count,
                "pass_fraction": count / ok if ok else None,
            }
            for cid, count in audit_counts.items()
        },
    }
    return arrays, ok, audit
