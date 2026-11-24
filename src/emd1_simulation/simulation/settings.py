"""Validated user settings and independent parameter overrides."""
from dataclasses import asdict, dataclass, replace
from numbers import Real
import numpy as np
from .model import Params, normalise_params

T_END = 45.0
N_T = 451
ENSEMBLE_SEED = 20260822
COUNTERFACTUAL_SEED = 20260825
DERIVED_PARAMETERS = frozenset({"R_ref", "s_a3b", "s_ned", "s_rep", "s_aox"})


@dataclass(frozen=True)
class SimulationSettings:
    """The canonical defaults can be overridden for exploratory runs."""
    t_end_days: float = T_END
    n_time_points: int = N_T
    n_ensemble: int = 400
    ensemble_seed: int = ENSEMBLE_SEED
    counterfactual_seed: int = COUNTERFACTUAL_SEED

    def __post_init__(self):
        for name, minimum in (("n_ensemble", 1), ("n_time_points", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if (isinstance(self.t_end_days, bool) or not isinstance(self.t_end_days, Real)
                or not np.isfinite(self.t_end_days) or self.t_end_days <= 0):
            raise ValueError("t_end_days must be finite and positive")
        for name in ("ensemble_seed", "counterfactual_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, float(self.t_end_days), self.n_time_points)


def prepare_parameters(overrides: dict | None = None) -> Params:
    """Validate independent parameter edits and derive the control baseline."""
    edits = dict(overrides or {})
    unknown = set(edits) - set(asdict(Params()))
    if unknown:
        raise ValueError(f"Unknown model parameters: {', '.join(sorted(unknown))}")
    derived = set(edits) & DERIVED_PARAMETERS
    if derived:
        raise ValueError(f"These parameters are derived automatically: {', '.join(sorted(derived))}")
    p = replace(Params(), **edits)
    _validate_parameters(p)
    return normalise_params(p)

def _validate_parameters(p: Params) -> None:
    for name, value in asdict(p).items():
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        if (name.startswith("M0_") or name == "w_aox_fto") and not 0.05 <= value <= 0.90:
            raise ValueError(f"{name} must be in [0.05, 0.90], the supported ensemble fraction range")
        if name in {"a_B", "w_aox_fto", "w_rep_writer", "w_rep_fto", "repair_proxy_weight", "wnt_min"} and value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
        if (name.startswith(("K_", "tau_", "d_", "n_", "h_"))
                or name in {"r_basal", "k_R", "k_fix"}) and value <= 0:
            raise ValueError(f"{name} must be positive")
    if p.k_ROS + p.k_A3B == 0:
        raise ValueError("At least one lesion-production gain (k_ROS or k_A3B) must be positive")


def validated_parameters(p: Params | None = None) -> Params:
    """Validate and normalise an already constructed parameter object."""
    p = p if p is not None else Params()
    _validate_parameters(p)
    return normalise_params(p)
