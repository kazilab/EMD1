"""Run the EMD1 simulation and compare its outputs with the bundled measurements."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .simulation.engine import SimulationResult, simulate_experiment
from .simulation.model import Params
from .simulation.settings import (
    SimulationSettings, validated_parameters, T_END, N_T,
    ENSEMBLE_SEED, COUNTERFACTUAL_SEED,
)
from .analysis.checks import CHECKS, CHECK_ROLES, evaluate_draw_checks
from .analysis.results import analyze_simulation
from .reporting.console import print_summary
from .reporting.provenance import reproduction_metadata
from .reporting.export import export_results


@dataclass
class ExperimentResult:
    """Live arrays, analysed values and export paths from one complete workflow."""
    simulation: SimulationResult
    summary: dict
    paths: list[Path]
    exit_code: int


def run_workflow(outdir: str | Path, settings: SimulationSettings | None = None, *,
                 params: Params | None = None, make_figure: bool = True,
                 make_tables: bool = True, strict_evidence: bool = False) -> ExperimentResult:
    """Run the three stages once and retain their outputs for interactive use."""
    settings = settings if settings is not None else SimulationSettings()
    params = validated_parameters(params)

    roles = {cid: CHECK_ROLES[cid] for cid in (*[check.cid for check in CHECKS], "V13")}
    print("Simulation: nominal conditions, sensitivity ensemble and explicit scenarios", flush=True)
    simulated = simulate_experiment(
        params, settings, check_evaluator=evaluate_draw_checks, check_roles=roles)

    print("Analysis: model checks, source comparisons and structural scenarios", flush=True)
    summary = analyze_simulation(simulated, settings)
    summary["reproducibility"] = reproduction_metadata(params, settings)
    summary["output_options"] = {"figure": bool(make_figure), "tables": bool(make_tables)}

    paths = export_results(simulated, summary, outdir,
                           make_figure=make_figure, make_tables=make_tables)
    print_summary(summary)
    print(f"\nResults saved in {Path(outdir)}")

    failed = summary["model_checks_passed"] != len(summary["model_checks"])
    code = 1 if failed else (2 if strict_evidence and not
                            summary["scientific_evidence_audit"]["quantitatively_validated"] else 0)
    return ExperimentResult(simulated, summary, paths, code)


def reproduce(outdir: str | Path, n_ensemble: int = 400, *,
              make_figure: bool = True, make_tables: bool = True,
              strict_evidence: bool = False, params: Params | None = None,
              t_end_days: float = T_END, n_time_points: int = N_T,
              ensemble_seed: int = ENSEMBLE_SEED,
              counterfactual_seed: int = COUNTERFACTUAL_SEED) -> int:
    """Return 0 on success, 1 for failed model checks or 2 for unmet strict evidence criteria."""
    settings = SimulationSettings(
        t_end_days=t_end_days, n_time_points=n_time_points, n_ensemble=n_ensemble,
        ensemble_seed=ensemble_seed, counterfactual_seed=counterfactual_seed)
    return run_workflow(outdir, settings, params=params, make_figure=make_figure,
                        make_tables=make_tables, strict_evidence=strict_evidence).exit_code
