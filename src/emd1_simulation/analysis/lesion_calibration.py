"""Fit aggregate mutation-rate contrasts to a user-supplied assay.

The default lesion parameters are illustrative. Calibration requires matched
aggregate counts and denominators, protocol metadata and a mapping from the
model fixation index to the measured rate. Gao Figures 1G and 3Q provide
mutation-category maxima and cannot supply aggregate rates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..simulation.conditions import BY_KEY
from ..simulation.model import Params, control_steady_state, simulate, trace

T_END, N_T = 45.0, 451
NAMES = ["k_ROS", "k_A3B", "k_rep", "K_L", "k_fix"]
KEYS = ("control", "as", "as_ftoi", "as_a3bkd")
LEGACY_CALIBRATION_ANCHOR = {
    "k_ROS": 1.0822, "k_A3B": 1.3724, "k_rep": 2.6977,
    "K_L": 0.5825, "k_fix": 0.2155,
}
LEGACY_CALIBRATION_STATUS = {
    "active": False,
    "reason": "Mutation-category maxima are not aggregate mutation burden with assay denominators.",
    "historical_targets": {"as/ctl": 4.0, "ftoi/as": 4.0 / 9.0},
    "historical_comparison": {"a3bkd/as": 3.0 / 8.0},
    "parameter_status": "illustrative values without a matched empirical fit",
    "eligible_replacement_observations_bundled": False,
}


class CalibrationDataRequired(ValueError):
    """An empirical calibration was requested without an eligible observation model."""


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationDataRequired(f"{name} must identify the assay or observation mapping")


@dataclass(frozen=True)
class MutationAssayObservation:
    """One aggregate count and its independently documented assay denominator.

    Count all mutations within the stated assay scope. Record the denominator
    unit (e.g., assayed genomes or target bases) consistently within a contrast.
    """
    assay_id: str
    condition_key: str
    aggregate_count: int
    denominator: float
    denominator_unit: str
    protocol_id: str
    source: str
    count_scope: str

    def __post_init__(self):
        for name in ("assay_id", "condition_key", "denominator_unit", "protocol_id", "source"):
            _nonempty(getattr(self, name), name)
        if self.count_scope != "all_assayed_mutations":
            raise CalibrationDataRequired(
                "count_scope must be 'all_assayed_mutations'; category maxima and selected bars "
                "cannot calibrate an aggregate burden endpoint")
        if (isinstance(self.aggregate_count, bool)
                or not isinstance(self.aggregate_count, Integral) or self.aggregate_count < 0):
            raise CalibrationDataRequired("aggregate_count must be a non-negative integer")
        if (isinstance(self.denominator, bool) or not isinstance(self.denominator, Real)
                or not np.isfinite(self.denominator) or self.denominator <= 0):
            raise CalibrationDataRequired("denominator must be finite and positive")

    @property
    def rate(self) -> float:
        return float(self.aggregate_count / self.denominator)


@dataclass(frozen=True)
class AssayContrast:
    """Rate contrast between observations from the same assay and protocol."""
    name: str
    numerator: MutationAssayObservation
    reference: MutationAssayObservation

    def __post_init__(self):
        _nonempty(self.name, "contrast name")
        if not isinstance(self.numerator, MutationAssayObservation) or not isinstance(
                self.reference, MutationAssayObservation):
            raise CalibrationDataRequired("contrasts require explicit MutationAssayObservation records")
        for field in ("assay_id", "protocol_id", "denominator_unit", "count_scope"):
            if getattr(self.numerator, field) != getattr(self.reference, field):
                raise CalibrationDataRequired(f"Contrast {self.name!r} has unmatched {field}")
        if self.numerator.condition_key == self.reference.condition_key:
            raise CalibrationDataRequired("a contrast must identify two different conditions")

    @property
    def rate_ratio(self) -> float:
        if self.numerator.aggregate_count == 0 or self.reference.aggregate_count == 0:
            raise CalibrationDataRequired(
                "The log-ratio objective cannot use zero counts; specify a count likelihood "
                "for such data instead of inserting a pseudocount")
        return self.numerator.rate / self.reference.rate


def _ratios(p: Params, t: np.ndarray) -> dict:
    """Illustrative model fixation-index ratios; no empirical calibration claim."""
    y0 = control_steady_state(p)
    n = {k: float(trace(simulate(p, BY_KEY[k].iv, BY_KEY[k].E, t, y0=y0),
                        p, BY_KEY[k].iv)["N"][-1]) for k in KEYS}
    return {"as/ctl": n["as"] / n["control"], "ftoi/as": n["as_ftoi"] / n["as"],
            "a3bkd/as": n["as_a3bkd"] / n["as"]}


def fit(base: Params, t: np.ndarray, fixed: dict | None = None):
    """Direct callers to the assay-specific calibration interface."""
    raise CalibrationDataRequired(
        "Mutation category-max calibration is disabled. Use fit_assay_contrasts with "
        "aggregate counts, denominators, matched protocols and an observation mapping.")


def fit_assay_contrasts(
    base: Params,
    t: np.ndarray,
    *,
    contrasts: Sequence[AssayContrast],
    observation_model: Callable[[Params, np.ndarray, MutationAssayObservation], float],
    observation_model_description: str,
    regularization_anchor: Mapping[str, float],
    regularization_weight: float = 0.15,
    fixed: Mapping[str, float] | None = None,
) -> tuple[Params, dict]:
    """Fit explicit assay rate ratios with a caller-declared observation mapping.

    The observation_model callable maps parameters, times and an observation
    to the expected aggregate count per unit of its assay denominator. The fit
    minimises squared log-rate-ratio residuals plus a log-parameter penalty.
    Count sampling error and parameter uncertainty are unestimated.
    """
    from ..simulation.settings import validated_parameters
    base = validated_parameters(base)
    contrasts = tuple(contrasts)
    if not contrasts or any(not isinstance(c, AssayContrast) for c in contrasts):
        raise CalibrationDataRequired("at least one eligible AssayContrast is required")
    if len({c.name for c in contrasts}) != len(contrasts):
        raise CalibrationDataRequired("contrast names must be unique")
    if not callable(observation_model):
        raise CalibrationDataRequired("an explicit observation_model callable is required")
    _nonempty(observation_model_description, "observation_model_description")
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise CalibrationDataRequired("model times must start at zero and strictly increase")
    if (isinstance(regularization_weight, bool) or not isinstance(regularization_weight, Real)
            or not np.isfinite(regularization_weight) or regularization_weight < 0):
        raise CalibrationDataRequired("regularization_weight must be finite and non-negative")
    fixed = dict(fixed or {})
    if set(fixed) - set(NAMES):
        raise CalibrationDataRequired("fixed keys must be lesion parameter names")
    free = [name for name in NAMES if name not in fixed]
    anchor = dict(regularization_anchor)
    if set(anchor) != set(free):
        raise CalibrationDataRequired("regularization_anchor must explicitly give every free lesion parameter")
    for name, value in {**fixed, **anchor}.items():
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise CalibrationDataRequired(f"{name} must be a finite positive parameter value")
    # Accessing rate_ratio also rejects zero-count log contrasts without fabricating data.
    observed = {c.name: c.rate_ratio for c in contrasts}
    x_anchor = np.log([anchor[name] for name in free])

    def build(x):
        edits = {name: float(np.exp(value)) for name, value in zip(free, x)}
        edits.update(fixed)
        return replace(base, **edits)

    def predicted(q):
        values = {}
        for contrast in contrasts:
            predictions = [float(observation_model(q, t, observation))
                           for observation in (contrast.numerator, contrast.reference)]
            if any(not np.isfinite(value) or value <= 0 for value in predictions):
                raise CalibrationDataRequired("observation model must return finite positive assay rates")
            values[contrast.name] = predictions[0] / predictions[1]
        return values

    def residual(x):
        pred = predicted(build(x))
        data = [np.log(pred[c.name] / observed[c.name]) for c in contrasts]
        return np.r_[data, regularization_weight * (x - x_anchor)]

    if free:
        solution = least_squares(residual, x_anchor, method="trf", max_nfev=1000,
                                 xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if not solution.success:
            raise RuntimeError(f"assay contrast optimization did not converge: {solution.message}")
        fitted = build(solution.x)
        optimizer = {"success": True, "message": str(solution.message),
                     "evaluations": int(solution.nfev), "penalized_cost": float(solution.cost)}
    else:
        fitted = build(np.array([]))
        optimizer = {"success": True, "message": "All lesion parameters fixed; no optimization",
                     "evaluations": 0, "penalized_cost": float(np.sum(residual(np.array([])) ** 2) / 2)}
    report = {
        "status": "explicit caller-supplied assay calibration; not built-in empirical validation",
        "observed_rate_ratios": observed,
        "predicted_rate_ratios": predicted(fitted),
        "observation_model_description": observation_model_description,
        "assay_contrasts": [asdict(c) for c in contrasts],
        "regularization_anchor": anchor,
        "regularization_weight": float(regularization_weight),
        "free_parameters": free,
        "fixed_parameters": fixed,
        "objective": "unweighted squared log-rate-ratio residuals plus log-parameter penalty",
        "uncertainty": "not estimated; source count variability is not a fitted likelihood",
        "identifiability": "not established by this fit or by optimizer convergence",
        "optimizer": optimizer,
    }
    return fitted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.profile:
        parser.error("A calibration profile requires a matched assay and observation model. "
                     "Use fit_assay_contrasts with assay data.")
    print("The default lesion parameters are illustrative. To fit aggregate mutation rates, "
          "use fit_assay_contrasts with matched counts, denominators and an observation model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
