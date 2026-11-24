"""Export EMD1 trajectories, model checks, source comparisons and parameter settings."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


OBSERVABLES = {
    "R": ("relative ROS units", "Modeled ROS before normalization to control"),
    "W": ("relative writer activity", "METTL3/14 writer activity"),
    "Fo": ("relative underlying FTO state", "FTO state before the intervention multiplier"),
    "Ab": ("relative underlying ALKBH5 state", "ALKBH5 state before the intervention multiplier"),
    "Fo_effective": ("relative effective eraser activity", "FTO state multiplied by the active intervention factor at each time point"),
    "Ab_effective": ("relative effective eraser activity", "ALKBH5 state multiplied by the active intervention factor at each time point"),
    "M_a3b": ("occupancy fraction", "APOBEC3B-site model occupancy"),
    "M_ned": ("occupancy fraction", "NEDD4L-site model occupancy"),
    "M_rep": ("occupancy fraction", "DDB2-associated repair-site model occupancy"),
    "M_aox": ("occupancy fraction", "Occupancy in the writer-associated antioxidant aggregate"),
    "M_mean4": ("mean occupancy fraction", "Unweighted mean occupancy of the four modeled targets"),
    "A3B": ("relative transcript abundance", "APOBEC3B transcript abundance"),
    "NEDD4L": ("relative transcript abundance", "NEDD4L transcript abundance"),
    "A": ("relative activity proxy", "Aggregate transcript abundance in the writer-associated antioxidant branch"),
    "Q": ("relative translation proxy", "DDB2 translation as a proxy for nucleotide excision repair"),
    "Q_lesion": ("relative lesion-removal modifier", "Assumed transfer from Q to aggregate lesion removal"),
    "WNT": ("relative signalling proxy", "WNT signalling readout"),
    "L": ("arbitrary lesion units", "Modeled repairable lesion pool"),
    "N": ("arbitrary cumulative fixation units", "Cumulative fixation index for a representative lineage"),
    "lnP": ("natural log population fold versus start", "Log population expansion from the starting population"),
    "P": ("population fold versus start", "Population size relative to its starting value, computed as exp(clip(lnP, -700, 700))"),
    "division_rate": ("1/nominal model day", "Instantaneous model division rate"),
    "death_rate": ("1/nominal model day", "Instantaneous model death rate"),
    "net_growth_rate": ("1/nominal model day", "Instantaneous division rate minus death rate"),
}

TABLE_DESCRIPTIONS = {
    "endpoint_folds": "Endpoint folds versus control. lnP gives log2 fold change.",
    "arsenic_sensitivity_endpoints": "5th, 50th and 95th percentiles of arsenic endpoints across independent sensitivity draws.",
    "model_checks": "Model consistency checks with their results, roles and evidence sources.",
    "model_check_retention": "Fraction of accepted parameter draws satisfying each model check.",
    "population_diagnostics": "Simulated population expansion, instantaneous rates and integrated hazards.",
    "repair_feature_evidence": "DDB2 and XPC enrichment contrasts over the first three annotated exons. Bounds describe variation across background transcripts.",
    "repair_structural_scenarios": "Endpoint folds for each repair scenario relative to its own control. Scenarios have equal standing without probability weights.",
    "repair_scenario_parameters": "Parameters changed from baseline for each repair scenario.",
    "antioxidant_chase_by_stage": "Experimental transcript-decay ratios by exposure stage and antioxidant branch.",
    "scientific_comparisons": "Model and experimental quantities with their units and assay limitations.",
    "parameters": "Configured parameter values, defaults and evidence for each assumption.",
    "source_status": "Evidence available for each source and its suitability for calibration.",
    "observation_contracts": "Source statistic, denominator, protocol, model observable and requirements for quantitative calibration.",
    "conditions": "Dimensionless exposure and relative intervention activity by condition.",
    "observable_dictionary": "Definitions and units for each model observable.",
    "nominal_trajectories": "Model trajectories in native units over nominal model days.",
    "relative_trajectories": "Folds versus the simultaneous control. lnP gives log2 fold change and net_growth_rate gives the difference per nominal day. N starts at a relative value of one. Other zero denominators give blanks.",
    "source_protocol_trajectories": "Unexposed control, FTO-inhibition and complete-FTO-loss trajectories in native units. Time and dose are illustrative.",
}


def _relative_unit(observable: str) -> str:
    if observable == "lnP":
        return "log2 fold vs control"
    if observable == "net_growth_rate":
        return "difference vs control, 1/nominal model day"
    return "fold vs control"


def _population_table(values: dict, t_end: float) -> pd.DataFrame:
    rows = []
    for condition, diagnostics in values.items():
        for metric, value in diagnostics.items():
            if metric == "net_population_loss":
                unit = "boolean: endpoint below starting population"
            elif metric.endswith("rate_at_endpoint"):
                unit = "1/nominal model day"
            elif metric.startswith("integrated_"):
                unit = "dimensionless model hazard"
            else:
                unit = "population fold versus start"
            rows.append({"condition": condition, "time_nominal_days": t_end,
                         "metric": metric, "value": value, "unit": unit})
    return pd.DataFrame(rows, columns=["condition", "time_nominal_days", "metric", "value", "unit"])


def summary_tables(summary: dict) -> dict[str, pd.DataFrame]:
    """Return tables keyed by filename stem for a completed simulation."""
    t_end = float(summary["t_end_days"])
    units = summary.get("endpoint_units", {})
    endpoint_rows = [
        {"condition": condition, "time_nominal_days": t_end,
         "observable": observable, "value": value,
         "unit": units.get(observable, _relative_unit(observable))}
        for condition, values in summary["endpoints_fold_vs_control"].items()
        for observable, value in values.items()
    ]
    tables = {"endpoint_folds": pd.DataFrame(endpoint_rows)}
    tables["arsenic_sensitivity_endpoints"] = pd.DataFrame([
        {"condition": "as", "time_nominal_days": t_end, "observable": observable,
         **percentiles, "unit": units.get(observable, _relative_unit(observable)),
         "accepted_draws": summary["ensemble_draws_ok"]}
        for observable, percentiles in summary["ensemble"].items()
    ])
    tables["model_checks"] = pd.DataFrame([
        {"check_id": check_id, **record}
        for check_id, record in summary["model_checks"].items()
    ])
    retention = summary.get("ensemble_model_check_sensitivity", {})
    tables["model_check_retention"] = pd.DataFrame([
        {"check_id": check_id, "draws": retention.get("draws"), **record}
        for check_id, record in retention.get("checks", {}).items()
    ], columns=["check_id", "draws", "role", "passes", "pass_fraction"])
    tables["population_diagnostics"] = _population_table(summary["population_diagnostics"], t_end)

    repair = summary["repair_edge"]
    feature_rows = []
    for contrast, ddb2 in repair["measured_ddb2_first_three_exons"].items():
        null_lo, null_hi = repair["cross_transcript_null_by_contrast"][contrast]
        for gene in ("DDB2", "XPC"):
            effect = (ddb2["effect"] if gene == "DDB2" else
                      repair["measured_xpc_same_pathway_comparator"][contrast])
            feature_rows.append({
                "contrast": contrast, "gene": gene, "effect": effect,
                "unit": "background-median-normalised IP/input contrast",
                "background_percentile": ddb2["background_pct"] if gene == "DDB2" else None,
                "background_p5": null_lo, "background_p95": null_hi,
                "background_transcripts_sampled": repair["n_background_sampled"],
                "background_transcripts_quantified": repair["n_background_quantified"][contrast],
                "repair_transcripts_quantified": repair["n_repair_quantified"][contrast],
            })
    tables["repair_feature_evidence"] = pd.DataFrame(feature_rows)

    scenario_rows, parameter_rows = [], []
    for name, scenario in summary["repair_structural_scenarios"]["scenarios"].items():
        for parameter, value in scenario["parameters"].items():
            parameter_rows.append({"scenario": name, "parameter": parameter, "value": value})
        for condition, observables in scenario["endpoint_folds_vs_own_control"].items():
            for observable, value in observables.items():
                scenario_rows.append({"scenario": name, "condition": condition,
                                      "time_nominal_days": t_end, "observable": observable,
                                      "value": value, "unit": "fold vs scenario's own control"})
    tables["repair_structural_scenarios"] = pd.DataFrame(scenario_rows)
    tables["repair_scenario_parameters"] = pd.DataFrame(parameter_rows,
                                                        columns=["scenario", "parameter", "value"])

    audit = summary["scientific_evidence_audit"]
    reconciliation = audit.get("antioxidant_branch_reconciliation", {})
    stage_rows = []
    for stage, values in reconciliation.get("by_stage", {}).items():
        row = {"experimental_stage": stage}
        row.update({key: value for key, value in values.items() if not isinstance(value, dict)})
        row.update({f"{key}_{gene}": value
                    for key, by_gene in values.items() if isinstance(by_gene, dict)
                    for gene, value in by_gene.items()})
        row["ratio_definition"] = "source k(siRNA)/k(siControl); dimensionless"
        stage_rows.append(row)
    tables["antioxidant_chase_by_stage"] = pd.DataFrame(stage_rows)
    comparison_rows = []
    comparison_rows.extend(audit.get("comparison_rows", []))
    tables["scientific_comparisons"] = pd.DataFrame(comparison_rows)

    metadata = summary.get("reproducibility", {})
    tables["parameters"] = pd.DataFrame([
        {"parameter": name, "value": value,
         **metadata.get("parameter_provenance", {}).get(name, {})}
        for name, value in metadata.get("parameters", {}).items()
    ])
    source_status = metadata.get("source_status", {})
    tables["source_status"] = pd.DataFrame([
        {"source": name,
         **{key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
            for key, value in record.items()}}
        for name, record in source_status.get("sources", {}).items()
    ])
    from ..paths import DATA_DIR
    contracts = json.loads((DATA_DIR / "observation_contracts.json").read_text(encoding="utf-8"))
    tables["observation_contracts"] = pd.DataFrame(contracts["records"])
    tables["conditions"] = pd.json_normalize(metadata.get("conditions", []), sep="_")
    tables["observable_dictionary"] = pd.DataFrame([
        {"observable": observable, "native_unit": unit,
         "relative_trajectory_unit": _relative_unit(observable), "meaning": meaning}
        for observable, (unit, meaning) in OBSERVABLES.items()
    ])
    return tables


def trajectory_table(runs: dict, t: np.ndarray, *, relative: bool = False) -> pd.DataFrame:
    """Return one row per condition/time, preserving every trace observable.

    ``relative=False`` keeps native scales, including raw occupancy fractions.
    ``relative=True`` gives fold versus the simultaneous control trajectory,
    except lnP (log2 population fold) and signed net_growth_rate (difference in
    1/nominal model day). Only the shared zero-valued N origin is set to one;
    other zero-denominator folds are undefined and exported as blank values.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or not t.size or not np.isfinite(t).all():
        raise ValueError("Trajectory times must be a non-empty finite one-dimensional array")
    frames = []
    control = runs["control"] if relative else None
    for condition, trace in runs.items():
        values = {"condition": condition, "time_nominal_days": t}
        for observable, array in trace.items():
            array = np.asarray(array, dtype=float)
            if array.shape != t.shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid trajectory shape or values: {condition}/{observable}")
            if relative:
                if observable == "lnP":
                    array = (array - np.asarray(control[observable])) / np.log(2.0)
                elif observable == "net_growth_rate":
                    array = array - np.asarray(control[observable])
                else:
                    denominator = np.asarray(control[observable])
                    ratio = np.full_like(array, np.nan)
                    np.divide(array, denominator, out=ratio, where=denominator != 0)
                    if observable == "N":
                        shared_origin = (t == 0) & (denominator == 0) & (array == 0)
                        ratio[shared_origin] = 1.0
                    array = ratio
            values[observable] = array
        frames.append(pd.DataFrame(values))
    return pd.concat(frames, ignore_index=True)


def export_tables(summary: dict, runs: dict, t: np.ndarray, outdir: Path, *,
                  source_protocols: dict | None = None) -> dict[str, Path]:
    """Write reproducible CSV tables beneath ``outdir/tables`` and return paths."""
    tables = summary_tables(summary)
    tables["nominal_trajectories"] = trajectory_table(runs, t)
    tables["relative_trajectories"] = trajectory_table(runs, t, relative=True)
    if source_protocols:
        tables["source_protocol_trajectories"] = trajectory_table(source_protocols, t)
    directory = Path(outdir) / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    catalog = pd.DataFrame([
        {"table": name, "filename": path.name, "rows": len(tables[name]),
         "description": TABLE_DESCRIPTIONS[name]}
        for name, path in paths.items()
    ])
    catalog_path = directory / "table_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    paths["table_catalog"] = catalog_path
    return paths
