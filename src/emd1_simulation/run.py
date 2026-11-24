"""Simulate arsenic exposure, RNA methylation and reader-dependent transcript responses."""
import argparse
from .paths import OUTPUT_DIR
from .simulation.settings import SimulationSettings
from .workflow import reproduce, run_workflow


def main() -> int:
    defaults = SimulationSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ensemble", type=int, default=defaults.n_ensemble)
    parser.add_argument("--t-end", type=float, default=defaults.t_end_days)
    parser.add_argument("--n-time-points", type=int, default=defaults.n_time_points)
    parser.add_argument("--ensemble-seed", type=int, default=defaults.ensemble_seed)
    parser.add_argument("--counterfactual-seed", type=int, default=defaults.counterfactual_seed)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR / "reproduced"))
    parser.add_argument("--strict-evidence", action="store_true",
                        help="return status 2 while the scientific evidence audit is unresolved")
    args = parser.parse_args()
    try:
        settings = SimulationSettings(
            t_end_days=args.t_end, n_time_points=args.n_time_points, n_ensemble=args.n_ensemble,
            ensemble_seed=args.ensemble_seed, counterfactual_seed=args.counterfactual_seed)
    except ValueError as exc:
        parser.error(str(exc))
    return run_workflow(
        args.outdir, settings, make_figure=not args.no_figure,
        make_tables=not args.no_tables, strict_evidence=args.strict_evidence).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
