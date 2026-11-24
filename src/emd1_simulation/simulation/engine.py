"""Run the nominal simulation, sensitivity analysis and structural scenarios."""
from dataclasses import dataclass, field
import numpy as np
from .model import Params
from .conditions import CONDITIONS
from .ensemble import run_ensemble
from .protocols import run_nominal, run_counterfactual, run_source_protocols
from .scenarios import repair_scenarios
from .settings import SimulationSettings, validated_parameters

OBSERVABLES = ["R", "Fo", "Ab", "Fo_effective", "Ab_effective",
               "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
               "A3B", "NEDD4L", "A", "Q", "WNT", "L", "N", "lnP"]


@dataclass
class SimulationResult:
    """Trajectories and sensitivity summaries for one experiment."""
    params: Params
    time: np.ndarray
    nominal: dict
    ensemble: dict
    counterfactual: dict
    ensemble_draws_ok: int
    check_sensitivity: dict
    repair_structures: dict = field(default_factory=dict)
    delayed: dict = field(default_factory=dict)
    settings: SimulationSettings | None = None
    source_protocols: dict = field(default_factory=dict)


def simulate_experiment(params: Params, settings: SimulationSettings, *,
                        check_evaluator=None, check_roles: dict | None = None) -> SimulationResult:
    """Integrate nominal arms, parameter draws and explicit alternative scenarios.

    Native per-draw trajectories are offered to an optional analysis callback
    before only the normalised ensemble arrays are retained in memory.
    """
    if (check_evaluator is None) != (check_roles is None):
        raise ValueError("Provide check_evaluator and check_roles together")
    params = validated_parameters(params)
    time = settings.time_grid()
    nominal = run_nominal(params, time)
    counterfactual = run_counterfactual(
        params, time, n=settings.n_ensemble, seed=settings.counterfactual_seed)
    audit = check_evaluator is not None
    outputs = run_ensemble(
        CONDITIONS, OBSERVABLES, time, n=settings.n_ensemble,
        seed=settings.ensemble_seed, audit_checks=audit, base=params,
        check_evaluator=check_evaluator, check_roles=check_roles)
    if audit:
        ensemble, accepted, checks = outputs
    else:
        ensemble, accepted = outputs
        checks = {}
    structures = repair_scenarios(params, time)
    delayed = run_nominal(params, time, intervention_start=30.0)
    source_protocols = run_source_protocols(params, time)
    return SimulationResult(
        params=params, time=time, nominal=nominal, ensemble=ensemble,
        counterfactual=counterfactual, ensemble_draws_ok=accepted,
        check_sensitivity=checks, repair_structures=structures, delayed=delayed,
        settings=settings, source_protocols=source_protocols)
