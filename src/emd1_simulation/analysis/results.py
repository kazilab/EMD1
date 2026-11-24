"""Summarise solved trajectories, model checks and source comparisons."""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..processing.repair_evidence import (
    REPAIR_DDB2, REPAIR_XPC,
    REPAIR_NULL_BAND, REPAIR_NULL_BY_CONTRAST, REPAIR_N_BACKGROUND,
    REPAIR_N_QUANTIFIED, REPAIR_N_REPAIR_QUANTIFIED,
    REPAIR_FEATURE_ACCESSION, REPAIR_FEATURE_CHROM, REPAIR_FEATURE_STRAND,
    REPAIR_FEATURE_EXONS_0BASED, REPAIR_INPUT_QC_BIN_NT,
    REPAIR_MIN_CONTROL_INPUT_BINS, REPAIR_MIN_CONTROL_INPUT_DENSITY,
)
from ..simulation.conditions import CONDITIONS
from ..simulation.protocols import REPAIR_ARSENIC_WRITER_INPUT
from .checks import Check, CHECK_ROLES, check_repair_edge, validate, v10_ensemble_sensitivity
from .evidence import (
    evidence_audit, population_summary,
    antioxidant_branch_reconciliation,
)

if TYPE_CHECKING:
    from ..simulation.engine import SimulationResult
    from ..simulation.settings import SimulationSettings


ENDPOINT_OBSERVABLES = (
    "R", "Fo", "Fo_effective", "Ab", "Ab_effective",
    "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
    "A3B", "NEDD4L", "A", "Q", "WNT", "N", "lnP",
)


def analyze_simulation(result: SimulationResult, settings: SimulationSettings) -> dict:
    """Return schema-version-3 results, without export or environment metadata."""
    if result.settings is not None and result.settings != settings:
        raise ValueError("Analysis settings differ from the completed simulation; rerun the simulation with those settings")
    if not np.array_equal(result.time, settings.time_grid()):
        raise ValueError("Analysis time grid differs from the completed simulation")
    p, t, runs = result.params, result.time, result.nominal
    ens, cf = result.ensemble, result.counterfactual
    n_ok = result.ensemble_draws_ok
    check_sensitivity = result.check_sensitivity or {
        "interpretation": "Per-draw check retention was not collected for this numerical run.",
        "draws": n_ok, "checks": {}}
    if not result.repair_structures or not result.delayed or not result.source_protocols:
        raise ValueError("Auxiliary simulation outputs are missing; use simulate_experiment to create a complete result")
    t_end_days = settings.t_end_days
    cols = ENDPOINT_OBSERVABLES
    ctl = {name: float(values[-1]) for name, values in runs["control"].items()}

    # V1-V12 use the configured parameters; V13 is the canonical repair-input
    # structural check. Exploratory input changes may intentionally fail it.
    results = validate(runs, p)
    m_rep_as = float(runs["as"]["M_rep"][-1]) / ctl["M_rep"]
    q_as = float(runs["as"]["Q"][-1]) / ctl["Q"]
    m_rep_wkd = float(runs["as_m3kd"]["M_rep"][-1]) / ctl["M_rep"]
    q_wkd = float(runs["as_m3kd"]["Q"][-1]) / ctl["Q"]
    v13, v13_msg = check_repair_edge(m_rep_as, q_as, m_rep_wkd, q_wkd)
    basis = (f"DDB2 enrichment across its first three exons was {REPAIR_DDB2['writer_kd'][0]:.2f}x "
             f"under writer knockdown and {REPAIR_DDB2['arsenic'][0]:.2f}x under arsenic. "
             f"Both values fell within the corresponding background distributions "
             f"({REPAIR_N_QUANTIFIED['writer_kd']} and {REPAIR_N_QUANTIFIED['arsenic']} transcripts). "
             f"XPC was {REPAIR_XPC['writer_kd']:.2f}x and {REPAIR_XPC['arsenic']:.2f}x, respectively. "
             "These enrichments do not establish a functional repair response.")
    results.append((Check(
        "V13", "EMD1 -> KCC3 (repair-site writer response and zero arsenic input)",
        v13_msg, None, basis), v13))
    n_pass = sum(passed for _, passed in results)
    role_counts = {}
    for check, passed in results:
        role = CHECK_ROLES[check.cid]
        record = role_counts.setdefault(role, {"passed": 0, "total": 0})
        record["total"] += 1
        record["passed"] += int(passed)
    v10_sensitivity = v10_ensemble_sensitivity(ens)

    recon = antioxidant_branch_reconciliation(p, runs, float(t_end_days))
    scientific = evidence_audit(p, runs, result.source_protocols, antioxidant_summary=recon)
    population = population_summary(runs, t)
    scenarios = result.repair_structures
    delayed = result.delayed

    summary = {
        "schema_version": 3,
        "time_interpretation": "Nominal model days have no calibrated conversion to experimental time.",
        "scientific_evidence_audit": scientific,
        "population_diagnostics": population,
        "repair_structural_scenarios": scenarios,
        "delayed_intervention_scenario": {
            "intervention_start_nominal_day": 30.0,
            "interpretation": "Illustrative intervention applied only when the simulation extends past nominal day 30.",
            "intervention_within_simulated_horizon": bool(t_end_days > 30),
            "population_diagnostics": population_summary(delayed, t)},
        "t_end_days": float(t_end_days),
        "endpoints_fold_vs_control": {
            c.key: {o: (float(runs[c.key][o][-1] - runs["control"]["lnP"][-1]) / np.log(2)
                        if o == "lnP" else float(runs[c.key][o][-1]) / ctl[o])
                    for o in cols}
            for c in CONDITIONS},
        "endpoint_units": {o: ("log2 fold vs control" if o == "lnP" else "fold vs control")
                           for o in cols},
        "model_checks": {chk.cid: {"role": CHECK_ROLES[chk.cid], "edge": chk.edge,
                                    "statement": chk.statement, "source": chk.source,
                                    "passed": ok}
                         for chk, ok in results},
        "model_checks_passed": n_pass,
        "model_checks_by_role": role_counts,
        "model_check_interpretation": (
            "Counts describe consistency with the selected model structure. V5 tests the "
            "assumed graph. V9 has indirect evidence from a different intervention."),
        "independent_biological_validation_performed": False,
        "ensemble": {o: dict(zip(("p5", "p50", "p95"),
                                 map(float, np.percentile(ens["as"][o][:, -1], [5, 50, 95]))))
                     for o in ens["as"]},
        "ensemble_draws_ok": n_ok,
        "ensemble_interpretation": (
            "The 5th and 95th percentiles summarise sensitivity to independent log-normal "
            "parameter perturbations. These widths are assumed."),
        "ensemble_check_sensitivity": {"V10": v10_sensitivity},
        "ensemble_model_check_sensitivity": check_sensitivity,
        "repair_edge": {
            "model_arsenic": {"M_rep_fold": m_rep_as, "Q_fold": q_as},
            "model_writer_kd": {"M_rep_fold": m_rep_wkd, "Q_fold": q_wkd},
            "counterfactual_params": REPAIR_ARSENIC_WRITER_INPUT,
            "counterfactual_M_rep_fold": float(cf["M_rep"][-1]),
            "counterfactual_Q_fold": float(cf["Q"][-1]),
            "counterfactual_N_fold": float(cf["N"][-1]),
            "counterfactual_draws_ok": int(cf.get("M_rep_n", 0)),
            "counterfactual_Q_p5_p95": (
                [float(cf["Q_lo"][-1]), float(cf["Q_hi"][-1])] if "Q_lo" in cf else None),
            "counterfactual_M_rep_p5_p95": (
                [float(cf["M_rep_lo"][-1]), float(cf["M_rep_hi"][-1])] if "M_rep_lo" in cf else None),
            "measured_ddb2_first_three_exons": {
                k: {"effect": v[0], "background_pct": v[1]}
                for k, v in REPAIR_DDB2.items()},
            "measured_xpc_same_pathway_comparator": REPAIR_XPC,
            "cross_transcript_null_5_95": list(REPAIR_NULL_BAND),
            "cross_transcript_null_by_contrast": {k: list(v) for k, v in REPAIR_NULL_BY_CONTRAST.items()},
            "n_background_sampled": REPAIR_N_BACKGROUND,
            "n_background_quantified": REPAIR_N_QUANTIFIED,
            "n_repair_quantified": REPAIR_N_REPAIR_QUANTIFIED,
            "repair_feature_definition": (
                "annotation-only, strand-aware first three exons of the longest "
                "spliced hg19 RefSeq NM_ isoform; identical across conditions and genes"),
            "ddb2_feature": {
                "accession": REPAIR_FEATURE_ACCESSION,
                "chrom": REPAIR_FEATURE_CHROM,
                "strand": REPAIR_FEATURE_STRAND,
                "exon_intervals_0based_half_open": [list(x) for x in REPAIR_FEATURE_EXONS_0BASED],
                "exonic_nt": sum(end - start for start, end in REPAIR_FEATURE_EXONS_0BASED),
            },
            "control_input_qc": {
                "fixed_bin_nt": REPAIR_INPUT_QC_BIN_NT,
                "minimum_passing_bins_per_control_replicate": REPAIR_MIN_CONTROL_INPUT_BINS,
                "minimum_library_normalised_density": REPAIR_MIN_CONTROL_INPUT_DENSITY,
                "eligibility_uses_treatment_tracks": False,
            },
            "writer_kd_track_reanalysis_supports_decrease": False,
            "edge_live_arsenic_input_assumed_zero": v13,
        },
    }
    return summary
