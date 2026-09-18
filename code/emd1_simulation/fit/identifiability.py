"""Local endpoint sensitivity diagnostic, not a likelihood identifiability test.

Positive folds use d(log fold)/d(log parameter). Signed log2 expansion uses
its untransformed derivative per log parameter, which remains finite at zero.
There is no measurement-error weighting: numerical rank is descriptive only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import numpy as np
from ..conditions import BY_KEY
from ..ensemble import FREE_SIGMA
from ..model import Params, normalise_params, control_steady_state, simulate, trace

T_END, N_T = 45.0, 241
OBSERVABLES = ["NEDD4L", "A", "Q", "M_ned", "M_a3b", "A3B", "R", "N", "lnP"]
ARMS = ["control", "as", "as_m3kd", "as_ftoi"]
SIGNED_ROWS = np.array([o == "lnP" for _ in ARMS[1:] for o in OBSERVABLES])


def _observe(p: Params) -> np.ndarray:
    p = normalise_params(p)
    t = np.linspace(0.0, T_END, N_T)
    y0 = control_steady_state(p)
    runs = {k: trace(simulate(p, BY_KEY[k].iv, BY_KEY[k].E, t, y0=y0), p, BY_KEY[k].iv)
            for k in ARMS}
    ctl = runs["control"]
    return np.asarray([
        (runs[arm][o][-1] - ctl[o][-1]) / np.log(2.0) if o == "lnP"
        else runs[arm][o][-1] / ctl[o][-1]
        for arm in ARMS[1:] for o in OBSERVABLES])


def transform_observations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values[~SIGNED_ROWS] <= 0):
        raise ValueError("positive folds and finite signed expansion are required")
    out = values.copy()
    out[~SIGNED_ROWS] = np.log(values[~SIGNED_ROWS])
    return out


def sensitivity(step: float = 0.05) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Symmetric finite differences in log-parameter space; step is log units."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    base = normalise_params(Params())
    names = sorted(FREE_SIGMA)
    f0 = _observe(base)
    S = np.zeros((len(f0), len(names)))
    for j, name in enumerate(names):
        value = getattr(base, name)
        if value <= 0:
            raise ValueError(f"log sensitivity requires a positive parameter: {name}")
        hi = transform_observations(_observe(replace(base, **{name: value * np.exp(step)})))
        lo = transform_observations(_observe(replace(base, **{name: value * np.exp(-step)})))
        S[:, j] = (hi - lo) / (2 * step)
    return S, names, f0


def spectrum(matrix: np.ndarray) -> dict:
    """Report both raw and column-normalized spectra without inferential labels."""
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix[:, norms > 1e-12] / norms[norms > 1e-12]
    raw = np.linalg.svd(matrix, compute_uv=False)
    sv = np.linalg.svd(normalized, compute_uv=False) if normalized.size else np.array([])
    return {"raw_singular_values": raw.tolist(),
            "column_normalized_singular_values": sv.tolist(),
            "ranks_by_relative_threshold": {
                str(threshold): int(np.count_nonzero(sv > sv[0] * threshold)) if sv.size else 0
                for threshold in (0.01, 0.05, 0.1)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    S, names, _ = sensitivity(args.step)
    small, _, _ = sensitivity(args.step / 2)
    print(f"Local sensitivity: {S.shape[0]} modeled endpoints x {len(names)} parameters")
    print("Not independent measured observations; no likelihood or measurement-error weighting.")
    print("Signed expansion uses a linear scale; other outputs use log folds.")
    print("Column-normalized numerical ranks at relative singular-value thresholds:")
    print(spectrum(S)["ranks_by_relative_threshold"])
    print("Half-step ranks:")
    print(spectrum(small)["ranks_by_relative_threshold"])
    print(f"Relative matrix change on halving step: {np.linalg.norm(S-small)/max(np.linalg.norm(small), 1e-12):.4g}")
    print("These ranks do not count data-identified parameters or prove structural identifiability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
