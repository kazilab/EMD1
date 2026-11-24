"""Print endpoint results and model checks for a completed run."""
from __future__ import annotations


def print_summary(summary: dict) -> None:
    print("EMD1 arsenic-exposure model")
    print(f"\nEndpoints at nominal model day {summary['t_end_days']:g}")
    print("Values are folds relative to control. lnP gives log₂ fold change.")
    endpoints = summary["endpoints_fold_vs_control"]
    columns = list(next(iter(endpoints.values())))
    widths = {name: max(9, len(name) + 2) for name in columns}
    print(f"{'condition':16s}" + "".join(f"{name:>{widths[name]}s}" for name in columns))
    for condition, values in endpoints.items():
        print(f"{condition:16s}" + "".join(f"{values[name]:>{widths[name]}.2f}" for name in columns))

    checks = summary["model_checks"]
    print(f"\n{summary['model_checks_passed']}/{len(checks)} model consistency checks passed.")
    for check_id, check in checks.items():
        if not check["passed"]:
            print(f"  {check_id} failed. {check['statement']}")

    accepted = summary["ensemble_draws_ok"]
    requested = summary.get("reproducibility", {}).get("ensemble_draws_requested", accepted)
    print(f"\nArsenic endpoints across {accepted}/{requested} accepted parameter draws")
    print(f"{'observable':16s} {'p5':>11s} {'p50':>11s} {'p95':>11s}")
    for observable, percentiles in summary["ensemble"].items():
        print(f"{observable:16s}{percentiles['p5']:>11.3f}"
              f"{percentiles['p50']:>11.3f}{percentiles['p95']:>11.3f}")

    repair = summary["repair_edge"]
    print(f"\nRepair writer-input scenario ({repair['counterfactual_draws_ok']} accepted draws)")
    print(f"  Repair-site occupancy, {repair['counterfactual_M_rep_fold']:.3f}× control")
    print(f"  DDB2 translation proxy, {repair['counterfactual_Q_fold']:.3f}× control")

    if not summary["scientific_evidence_audit"]["quantitatively_validated"]:
        print("\nQuantitative biological validation remains unresolved.")
