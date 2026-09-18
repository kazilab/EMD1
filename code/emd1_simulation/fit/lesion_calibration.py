"""Calibrate the lesion/mutation rates, reproducibly.

    python -m emd1_simulation.fit.lesion_calibration

Five parameters (`k_ROS, k_A3B, k_rep, K_L, k_fix`) are estimated from **two**
digitised mutation-count ratios. That is under-determined, and the point of this
module is to make that explicit rather than to hide it behind a set of numbers
sitting in ``model.py``:

* The two targets are JBC Fig. 1G (control 2 -> arsenic 8, a fourfold rise) and
  Fig. 3Q (arsenic 9 -> FB23-2 4, a 0.44x rescue).
* The A3B-knockdown ratio (Fig. 1G, 8 -> 3) is deliberately **excluded**, so
  that arm stays held out and V5 remains a prediction. An earlier pass fitted to
  it, which quietly turned V5 into a restatement of the fit.
* With two residuals and five unknowns the problem is regularised by a
  log-space prior pulling toward the explicit, fixed ``CALIBRATION_ANCHOR``
  below. The anchor is deliberately independent of the current ``Params``
  values, so rerunning this module is a reproducible calibration rather than a
  moving-prior iteration. Individual parameters are NOT identified; only a
  regularised combination that reproduces the two ratios is selected.
  ``--profile`` shows this directly by scanning each parameter and refitting
  the rest.

So these values should be read as one self-consistent set that reproduces two
measured ratios, not as five separately estimated rate constants.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from ..conditions import BY_KEY
from ..model import Params, control_steady_state, simulate, trace

T_END, N_T = 45.0, 451
NAMES = ["k_ROS", "k_A3B", "k_rep", "K_L", "k_fix"]
KEYS = ("control", "as", "as_ftoi", "as_a3bkd")

# Digitised from J Biol Chem 298:101563 figure panels (see fit/digitised.py).
FIT_TARGETS = {"as/ctl": 8 / 2, "ftoi/as": 4 / 9}
HELD_OUT = {"a3bkd/as": 3 / 8}
PRIOR_WEIGHT = 0.15

# Pre-refit lesion values used as the fixed regularisation anchor. Keep these
# separate from Params: centring the prior on the current fitted output caused
# each invocation to take another small optimisation step, so the calibration
# did not reproduce itself. These constants record that modelling choice and
# make the fitted values in model.py independently checkable.
CALIBRATION_ANCHOR = {
    "k_ROS": 1.0822,
    "k_A3B": 1.3724,
    "k_rep": 2.6977,
    "K_L": 0.5825,
    "k_fix": 0.2155,
}


def _ratios(p: Params, t: np.ndarray) -> dict:
    y0 = control_steady_state(p)
    n = {k: float(trace(simulate(p, BY_KEY[k].iv, BY_KEY[k].E, t, y0=y0),
                        p, BY_KEY[k].iv)["N"][-1]) for k in KEYS}
    return {"as/ctl": n["as"] / n["control"], "ftoi/as": n["as_ftoi"] / n["as"],
            "a3bkd/as": n["as_a3bkd"] / n["as"]}


def fit(base: Params, t: np.ndarray, fixed: dict | None = None) -> tuple[Params, dict]:
    fixed = fixed or {}
    free = [n for n in NAMES if n not in fixed]
    x_anchor = np.log([CALIBRATION_ANCHOR[n] for n in free])

    def build(x):
        d = {n: float(np.exp(v)) for n, v in zip(free, x)}
        d.update(fixed)
        return replace(base, **d)

    def resid(x):
        try:
            g = _ratios(build(x), t)
        except Exception:
            return np.full(len(FIT_TARGETS) + len(free), 10.0)
        r = [np.log(g[k] / v) for k, v in FIT_TARGETS.items()]
        return np.array(r + list(PRIOR_WEIGHT * (x - x_anchor)))

    sol = least_squares(resid, x_anchor, method="trf", xtol=1e-10, ftol=1e-10)
    p = build(sol.x)
    return p, _ratios(p, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true",
                    help="scan each parameter and refit the rest, to show that "
                         "individual values are not identified")
    args = ap.parse_args()

    t = np.linspace(0.0, T_END, N_T)
    base = Params()
    p, got = fit(base, t)

    print("=" * 78)
    print("Lesion / mutation calibration: 5 parameters, 2 measured ratios")
    print("=" * 78)
    print("Fixed log-prior anchor is recorded in CALIBRATION_ANCHOR; it is not")
    print("read from the current fitted values in Params.")
    print(f"\n  {'parameter':10s}{'in model.py':>14s}{'refit here':>13s}")
    for n in NAMES:
        print(f"  {n:10s}{getattr(base, n):14.4f}{getattr(p, n):13.4f}")

    print(f"\n  {'ratio':11s}{'target':>9s}{'model':>9s}   role")
    for k, v in FIT_TARGETS.items():
        print(f"  {k:11s}{v:9.3f}{got[k]:9.3f}   fit target")
    for k, v in HELD_OUT.items():
        print(f"  {k:11s}{v:9.3f}{got[k]:9.3f}   HELD OUT -- prediction (V5)")
    print("\n  The held-out ratio is reproduced in direction but overshoots. That gap is")
    print("  a result, not a residual to be tuned away: JBC 3Q and 1G imply different")
    print("  APOBEC3B potencies, and a single linear A3B -> lesion coupling cannot")
    print("  satisfy both.")

    if args.profile:
        print("\n" + "=" * 78)
        print("Identifiability: fix one parameter, refit the other four")
        print("=" * 78)
        print(f"\n  {'parameter':10s}{'fixed at':>10s}{'as/ctl':>9s}{'ftoi/as':>9s}"
              f"{'residual':>10s}")
        for n in NAMES:
            for mult in (0.5, 1.0, 2.0):
                val = getattr(p, n) * mult
                try:
                    _, g = fit(p, t, fixed={n: val})
                    res = sum((np.log(g[k] / v)) ** 2 for k, v in FIT_TARGETS.items())
                    print(f"  {n:10s}{val:10.4f}{g['as/ctl']:9.3f}"
                          f"{g['ftoi/as']:9.3f}{res:10.2e}")
                except Exception as exc:
                    print(f"  {n:10s}{val:10.4f}   failed: {exc}")
        print("\n  A parameter whose residual stays near zero across a twofold range")
        print("  either way is not identified by these data; the other four absorb it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
