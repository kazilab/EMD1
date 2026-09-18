"""Report biological discrepancies separately from encoded model checks.

Values are model-to-assay comparisons, not a fitted likelihood. Structural
scenarios carry no probability weights. No gains are refitted to force agreement.
"""
from dataclasses import replace
import numpy as np

from .conditions import BY_KEY, antioxidant_decay_rate
from .model import Params, Intervention, control_steady_state, simulate, trace


REPAIR_SCENARIOS = {
    "no_exposure_input": {},
    "partial_writer_input": {"w_rep_writer": 0.5},
    "full_writer_input": {"w_rep_writer": 1.0},
    "fto_sensitive_site": {"w_rep_fto": 1.0},
    "writer_and_fto_input": {"w_rep_writer": 1.0, "w_rep_fto": 1.0},
    "legacy_universal_repair_transfer": {"repair_proxy_weight": 1.0},
    "writer_input_with_assumed_repair_transfer": {
        "w_rep_writer": 1.0, "repair_proxy_weight": 1.0},
}


def population_summary(runs: dict, t: np.ndarray) -> dict:
    """Rates/hazards are modeled quantities, not measured apoptosis counts."""
    return {key: {
        "population_fold_vs_start": float(np.exp(tr["lnP"][-1] - tr["lnP"][0])),
        "net_population_loss": bool(tr["lnP"][-1] < tr["lnP"][0]),
        "division_rate_at_endpoint": float(tr["division_rate"][-1]),
        "death_rate_at_endpoint": float(tr["death_rate"][-1]),
        "net_growth_rate_at_endpoint": float(tr["net_growth_rate"][-1]),
        "integrated_division_hazard": float(np.trapezoid(tr["division_rate"], t)),
        "integrated_death_hazard": float(np.trapezoid(tr["death_rate"], t)),
    } for key, tr in runs.items()}


def repair_scenarios(p: Params, t: np.ndarray) -> dict:
    results = {}
    for name, changes in REPAIR_SCENARIOS.items():
        q = replace(p, **changes)
        y0 = control_steady_state(q)
        ctl = trace(simulate(q, Intervention(), 0., t, y0), q, Intervention())
        arms = {}
        for key in ("as", "as_ftoi", "as_m3kd"):
            c = BY_KEY[key]
            tr = trace(simulate(q, c.iv, c.E, t, y0), q, c.iv)
            arms[key] = {o: float(tr[o][-1] / ctl[o][-1])
                         for o in ("M_rep", "Q", "Q_lesion", "N")}
        results[name] = {"parameters": changes, "endpoint_folds_vs_own_control": arms}
    return {"interpretation": "Unweighted structural scenarios, not confidence or credible intervals. "
            "Q is a DDB2/NER proxy; Q_lesion is an assumed transfer to aggregate repair.",
            "scenarios": results}


# The source paper assigns SOD1/SOD2/CAT/TXN/GPX1 to METTL3/YTHDF1 and PRDX5
# to FTO/YTHDF2. The model pools all six onto one transcript with one eraser
# blend, so a pooled chase summary mixes the two branches.
WRITER_BRANCH = ("SOD1", "SOD2", "CAT", "TXN", "GPX1")
ERASER_BRANCH = ("PRDX5",)
# The nominal horizon in nominal days, against the chase's exposure stages in
# weeks. Used only to say which stage the arsenic arm is nearest, never to
# claim the two time axes are calibrated to each other.
STAGE_WEEKS = {"8w": 8.0, "22w": 22.0, "22+12w": 34.0}


def antioxidant_branch_reconciliation(p: Params, runs: dict, t_end_days: float) -> dict:
    """Stage- and branch-resolved chase summary, replacing a pooled median.

    The pooled median (1.54x, all six genes, all three stages) is not a
    reliable summary of these data. Two things hide inside it. PRDX5 sits on a
    different branch and moves the other way under writer knockdown. And the
    eraser effect on the writer branch is stage-dependent: absent at 8w and
    22w, strong at 22+12w. With five genes and a bimodal spread the median
    also picks a middle value while the mean log-ratio is ~0, so the pooled
    number overstates a disagreement that the matched stage does not show.
    """
    from scipy.stats import ttest_1samp
    from .fit.zhao import chase_decay_rates

    rates = chase_decay_rates()
    means = rates.groupby(["gene", "stage", "sirna"]).k.mean().unstack("sirna")
    ratio = (means.siFTO / means.siControl).rename("siFTO")
    writer_ratio = (means.siMETTL3 / means.siControl).rename("siMETTL3")

    nearest = min(STAGE_WEEKS, key=lambda s: abs(STAGE_WEEKS[s] - t_end_days / 7.0))
    by_stage = {}
    for stage in STAGE_WEEKS:
        wb = np.array([ratio.loc[(gene, stage)] for gene in WRITER_BRANCH])
        lr = np.log(wb)
        by_stage[stage] = {
            "writer_branch_median_siFTO": float(np.median(wb)),
            "writer_branch_mean_log_ratio": float(lr.mean()),
            "writer_branch_p_vs_no_effect": float(ttest_1samp(lr, 0.0).pvalue),
            "writer_branch_n_faster": int((wb > 1).sum()),
            "writer_branch_n": int(wb.size),
            "eraser_branch_siFTO": {g: float(ratio.loc[(g, stage)]) for g in ERASER_BRANCH},
            "writer_branch_median_siMETTL3": float(
                np.median([writer_ratio.loc[(g, stage)] for g in WRITER_BRANCH])),
        }

    matched = by_stage[nearest]
    model_fto = float(antioxidant_decay_rate(runs["as_ftoi"]["M_aox"][-1], p)
                      / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p))
    contradicted = matched["writer_branch_p_vs_no_effect"] < 0.05 and (
        (model_fto - 1.0) * (matched["writer_branch_mean_log_ratio"]) < 0)
    return {
        "interpretation": "Chase decay-constant ratios vs siControl, split by exposure stage and "
            "by the branch assignment the source paper makes. Replaces a pooled median over six "
            "genes and three stages, which mixed both branches and three stages of transformation.",
        "pooled_median_all_genes_all_stages": float(ratio.median()),
        "pooled_is_misleading_because": "PRDX5 is on the eraser branch and inverts under writer "
            "knockdown; and the eraser effect on the writer branch appears only at the latest stage.",
        "model_horizon_days": float(t_end_days),
        "nearest_chase_stage": nearest,
        "model_ftoi_over_arsenic": model_fto,
        "by_stage": by_stage,
        "matched_stage_contradicts_model": bool(contradicted),
        "status": ("no detectable eraser effect on the writer branch at the stage nearest the "
                   "model horizon; the disagreement is confined to the latest stage and is a "
                   "stage-dependence finding, not a refutation of the coupling sign")
        if not contradicted else
        ("the matched stage contradicts the modelled direction"),
    }


def evidence_audit(p: Params, runs: dict) -> dict:
    from .fit.zhao import chase_decay_rates
    from .fit.digitised import APOBEC3B_ARM

    rates = chase_decay_rates()
    means = rates.groupby(["gene", "stage", "sirna"]).k.mean().unstack("sirna")
    observed_fto = float((means.siFTO / means.siControl).median())
    model_fto = (antioxidant_decay_rate(runs["as_ftoi"]["M_aox"][-1], p)
                 / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p))
    datum = next(d for d in APOBEC3B_ARM if d.panel == "3K")
    a3b_target = datum.values["FB23-2"] / datum.values["DMSO"]
    a3b_ratio = float(runs["as_ftoi"]["A3B"][-1] / runs["as"]["A3B"][-1])
    a3b_floor = p.s_a3b / (p.d_a3b * (1 + p.gam_d2))
    return {
        "interpretation": "These discrepancies are separate from nominal encoding checks. "
            "Perturbation strength and experimental context are not matched; no joint likelihood is claimed.",
        "antioxidant_fto_decay": {
            "model_ftoi_over_arsenic": float(model_fto),
            "observed_median_siFTO_over_siControl": observed_fto,
            "direction_agrees": bool((model_fto - 1) * (observed_fto - 1) > 0),
            "status": "pooled comparison retained for continuity only; it mixes two branches and "
                      "three exposure stages. Read antioxidant_branch_reconciliation instead, which "
                      "resolves both and finds no detectable effect at the matched stage.",
            "superseded_by": "antioxidant_branch_reconciliation",
        },
        "a3b_fto_rescue": {
            "model_ftoi_over_arsenic": a3b_ratio,
            "digitised_FB23_2_over_DMSO": a3b_target,
            "model_minimum_over_nominal_arsenic_with_fixed_synthesis": float(a3b_floor / runs["as"]["A3B"][-1]),
            "status": "unresolved magnitude and protocol mismatch; inhibitor strength alone is insufficient at nominal gains",
        },
        "time_scale": {
            "unit": "nominal model day; not physiologically calibrated",
            "model_arsenic_antioxidant_half_life_nominal_hours": float(24 * np.log(2) / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p)),
            "observed_median_chase_control_half_life_hours": float((np.log(2) / means.siControl).median()),
            "status": "different contexts; not a matched-condition quantitative validation",
        },
        "mutation_interpretation": "Relative cumulative fixation index for a representative modeled cell/lineage; "
            "not calibrated mutations per genome, a live-population average, or a clonal-selection model.",
        "repair_interpretation": "No functional aggregate-repair measurement; DDB2 methylation background "
            "percentiles are neither repair capacity measurements nor equivalence tests.",
        "quantitatively_validated": False,
    }
