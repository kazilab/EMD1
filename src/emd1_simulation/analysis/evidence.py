"""Compare model responses with source measurements in their assay contexts."""
from __future__ import annotations

import numpy as np

from .checks import antioxidant_decay_rate
from ..simulation.model import Params, apobec3b_bounds


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


WRITER_BRANCH = ("SOD1", "SOD2", "CAT", "TXN", "GPX1")
ERASER_BRANCH = ("PRDX5",)
# Experimental labels only. No conversion from nominal model days is defined.
EXPERIMENTAL_STAGES = ("8w", "22w", "22+12w")


def _chase_means():
    from ..processing.chase import chase_decay_rates

    rates = chase_decay_rates()
    if not np.isfinite(rates.k).all() or (rates.k <= 0).any():
        raise ValueError("Chase summaries require finite positive decay estimates")
    return rates.groupby(["gene", "stage", "sirna"]).k.mean().unstack("sirna")


def antioxidant_branch_reconciliation(p: Params, runs: dict, t_end_days: float) -> dict:
    """Descriptive source ratios at every stage, with no model-agreement test.

    Ratios first average fitted rates within each gene/stage/arm. Cross-gene
    summaries describe these selected genes; genes are not biological replicate
    samples. The model has one writer-associated phenomenological pool and does
    not implement the distinct PRDX5/YTHDF2 reader mechanism.
    """
    means = _chase_means()
    ratios = {arm: means[arm] / means.siControl for arm in ("siFTO", "siMETTL3")}
    by_stage = {}
    for stage in EXPERIMENTAL_STAGES:
        writer = {
            arm: {gene: float(values.loc[(gene, stage)]) for gene in WRITER_BRANCH}
            for arm, values in ratios.items()
        }
        wb = np.asarray(list(writer["siFTO"].values()))
        lr = np.log(wb)
        by_stage[stage] = {
            "writer_branch_median_siFTO": float(np.median(wb)),
            "writer_branch_geometric_mean_siFTO": float(np.exp(lr.mean())),
            "writer_branch_mean_log_ratio": float(lr.mean()),
            "writer_branch_min_siFTO": float(wb.min()),
            "writer_branch_max_siFTO": float(wb.max()),
            "writer_branch_n_faster": int((wb > 1).sum()),
            "writer_branch_n": int(wb.size),
            "writer_branch_siFTO": writer["siFTO"],
            "writer_branch_siMETTL3": writer["siMETTL3"],
            "writer_branch_median_siMETTL3": float(np.median(
                list(writer["siMETTL3"].values()))),
            "eraser_branch_siFTO": {
                gene: float(ratios["siFTO"].loc[(gene, stage)]) for gene in ERASER_BRANCH},
            "eraser_branch_siMETTL3": {
                gene: float(ratios["siMETTL3"].loc[(gene, stage)]) for gene in ERASER_BRANCH},
            "control_half_life_hours_by_gene": {
                gene: float(np.log(2) / means.loc[(gene, stage), "siControl"])
                for gene in (*WRITER_BRANCH, *ERASER_BRANCH)},
        }

    model_fto = float(antioxidant_decay_rate(runs["as_ftoi"]["M_aox"][-1], p)
                      / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p))
    return {
        "interpretation": (
            "Descriptive chase decay-rate ratios relative to siControl at each stage "
            "for the writer-associated genes and PRDX5."),
        "model_horizon_nominal_days": float(t_end_days),
        "model_ftoi_over_arsenic": model_fto,
        "model_pool_scope": (
            "One writer-associated phenomenological antioxidant pool; the opposite "
            "PRDX5/YTHDF2 reader mechanism is not represented."),
        "by_stage": by_stage,
        "source_ratio_definition": "mean fitted k(siRNA) / mean fitted k(siControl), per gene and stage",
        "source_replication_note": (
            "Cross-gene summaries describe the five selected writer-associated genes. "
            "They are not estimates of biological-replicate sampling uncertainty."),
        "status": (
            "Quantitative agreement remains unresolved because experimental stages, "
            "intervention strengths and protocols are unmatched."),
    }


def evidence_audit(p: Params, runs: dict, source_protocols: dict, *,
                   antioxidant_summary: dict) -> dict:
    """Compare unexposed control, partial FTO inhibition and complete FTO loss with source assays."""
    from ..processing.digitised import simulation_endpoint_values
    from ..processing.store import source_status

    required = {"control", "ftoi", "fto_complete_loss"}
    missing = required - set(source_protocols)
    if missing:
        raise ValueError(f"Missing unexposed source-background scenarios: {sorted(missing)}")
    ctl = source_protocols["control"]
    ftoi = source_protocols["ftoi"]
    loss = source_protocols["fto_complete_loss"]
    a3b_values = simulation_endpoint_values("3K")
    a3b_target = float(a3b_values["FB23-2"] / a3b_values["DMSO"])
    a3b_ratio = float(ftoi["A3B"][-1] / ctl["A3B"][-1])
    bounds = apobec3b_bounds(p)
    bound = float(bounds["abundance_ratio_lower"])
    model_fto = float(antioxidant_decay_rate(runs["as_ftoi"]["M_aox"][-1], p)
                      / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p))
    control_decay = p.d_a3b * (1 + p.gam_d2 * float(ctl["M_a3b"][-1]))
    complete_loss_decay = p.d_a3b * (1 + p.gam_d2 * float(loss["M_a3b"][-1]))
    audit = {
        "interpretation": (
            "Descriptive comparisons in each assay's exposure context. The model lacks "
            "a joint assay likelihood or a quantitative match to experimental protocols."),
        "source_status": source_status(),
        "antioxidant_fto_decay": {
            "model_ftoi_over_arsenic": model_fto,
            "source_summary_location": "antioxidant_branch_reconciliation.by_stage",
            "status": (
                "The model represents one writer-associated pool. Source observations "
                "are specific to each gene, branch and stage."),
        },
        "a3b_fto_inhibition": {
            "model_ftoi_over_unexposed_control": a3b_ratio,
            "source_FB23_2_over_DMSO": a3b_target,
            "model_complete_fto_loss_over_unexposed_control": float(loss["A3B"][-1] / ctl["A3B"][-1]),
            "abundance_lower_bound_over_unexposed_control": bound,
            "source_context": (
                "Gao Figure 3K: A549, DMSO versus 10 micromolar FB23-2 for 12 experimental "
                "hours; the panel description does not specify arsenic exposure."),
            "model_context": (
                "E=0 in both numerator and denominator; FTO activity multiplier 0.40. "
                "Exposure background is represented explicitly, but potency and timing are uncalibrated."),
            "status": (
                "The fixed-synthesis mechanism cannot reach the digitised abundance ratio "
                "at these gains and baseline occupancy, even with complete FTO inhibition; "
                "the experimental timing and inhibitor strength remain unmatched."
                if bound > a3b_target else
                "The digitised ratio is not excluded by the configured algebraic bound; "
                "this is not fitted agreement. Experimental timing and inhibitor strength remain unmatched."),
        },
        "a3b_kinetic_bounds": {
            **bounds,
            "complete_loss_decay_ratio_at_model_endpoint": float(complete_loss_decay / control_decay),
            "source_context": (
                "Gao Fig 3E describes an A549 FTO-knockdown actinomycin chase with no "
                "specified arsenic exposure. The approximately 3.3-fold value is a "
                "secondary estimate and does not calibrate the model."),
            "status": (
                "Bounds are algebraic model properties, not measured efficiencies or "
                "experimental confidence limits. Reader gain and baseline occupancy are assumed."),
        },
        "time_scale": {
            "unit": "nominal model day; no experimental time conversion",
            "model_arsenic_antioxidant_half_life_nominal_days": float(
                np.log(2) / antioxidant_decay_rate(runs["as"]["M_aox"][-1], p)),
            "source_half_life_location": (
                "antioxidant_branch_reconciliation.by_stage.*.control_half_life_hours_by_gene"),
            "status": (
                "Model half-lives are in nominal days. Source chase half-lives are in "
                "experimental hours, with no established conversion."),
        },
        "mutation_interpretation": (
            "Illustrative cumulative fixation index for a representative modeled lineage. "
            "TP53 substitution-category maxima provide qualitative evidence, without "
            "the aggregate counts and denominators needed for calibration."),
        "repair_interpretation": (
            "No functional aggregate-repair measurement; DDB2 cross-transcript percentiles "
            "are neither repair-capacity measurements nor equivalence tests."),
        "quantitatively_validated": False,
    }
    audit["antioxidant_branch_reconciliation"] = antioxidant_summary
    rows = [{
        "comparison": "APOBEC3B FTO inhibition without arsenic",
        "model_value": a3b_ratio,
        "model_quantity_or_unit": "A3B(E=0,FTO multiplier 0.40) / A3B(E=0,control)",
        "source_value": a3b_target,
        "source_quantity_or_unit": "Gao Fig 3K A3B(FB23-2) / A3B(DMSO), 12 experimental hours",
        "status": audit["a3b_fto_inhibition"]["status"],
    }]
    for stage, values in antioxidant_summary["by_stage"].items():
        rows.append({
            "comparison": f"Writer-associated antioxidant decay at source stage {stage}",
            "model_value": model_fto,
            "model_quantity_or_unit": "model k(arsenic+FTOi) / k(arsenic), uncalibrated nominal endpoint",
            "source_value": values["writer_branch_median_siFTO"],
            "source_quantity_or_unit": "source median across five genes of mean k(siFTO) / mean k(siControl)",
            "status": antioxidant_summary["status"],
        })
        rows.append({
            "comparison": f"PRDX5 decay at source stage {stage}",
            "model_value": None,
            "model_quantity_or_unit": "not represented by the current writer-associated aggregate",
            "source_value": values["eraser_branch_siFTO"]["PRDX5"],
            "source_quantity_or_unit": "source mean PRDX5 k(siFTO) / mean PRDX5 k(siControl)",
            "status": "PRDX5 has a distinct YTHDF2-mediated decay mechanism outside the model.",
        })
    rows.append({
        "comparison": "Antioxidant model half-life under arsenic",
        "model_value": audit["time_scale"]["model_arsenic_antioxidant_half_life_nominal_days"],
        "model_quantity_or_unit": "nominal model days",
        "source_value": None,
        "source_quantity_or_unit": "source half-lives remain gene/stage-specific experimental hours in chase table",
        "status": audit["time_scale"]["status"],
    })
    audit["comparison_rows"] = rows
    return audit
