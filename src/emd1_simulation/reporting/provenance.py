"""Record the data snapshot and computational choices used by one run."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import platform

from ..paths import DATABASE_PATH, DATA_DIR, PACKAGE_ROOT
from ..simulation.conditions import CONDITIONS
from ..simulation.ensemble import FREE_SIGMA
from ..simulation.model import Params, normalise_params
from ..simulation.settings import SimulationSettings

RELEASE_VERSION = "2026.11.12-scientific-corrections"


def parameter_provenance(params: Params) -> dict:
    """Record the basis of each configured parameter and its reference value."""
    defaults = normalise_params(Params())
    derived = {"R_ref", "s_a3b", "s_ned", "s_rep", "s_aox"}
    structural = {"a_B", "w_rep_writer", "w_rep_fto", "repair_proxy_weight", "wnt_min"}
    lesion = {"k_ROS", "k_A3B", "k_rep", "K_L", "k_fix"}
    growth = {"K_N", "h_N", "k_prol", "d0", "k_apo", "K_Rd", "n_Rd", "k_dam", "K_Ld", "n_Ld"}
    notes = {
        "a_F": "Illustrative FTO response amplitude. The NEDD4L locus lacks verified enrichment data for calibration.",
        "a_W": "Assumed writer response amplitude.",
        "gam_d2": "YTHDF2 promotes decay. At the reference occupancy of 0.35 and assumed gain of 10, the largest possible decay ratio is 2.444.",
        "gam_i2": "Stabilization follows the reported reader effect. Its magnitude depends on assumed occupancy and intervention strength.",
        "gam_ia": "Assumed stabilizing gain for the writer-associated antioxidant aggregate. PRDX5 decay is outside this branch.",
        "gam_f1r": "Functional evidence supports the DDB2 translation mechanism. Its strength and transfer across protocols are assumed.",
        "w_aox_fto": "Assumed FTO contribution to the writer-associated antioxidant aggregate.",
        "w_rep_writer": "Assumed exposure-specific writer input. The default is zero.",
        "w_rep_fto": "Assumed FTO contribution to repair-site erasure.",
        "repair_proxy_weight": "Assumed transfer from the DDB2/NER proxy to aggregate lesion removal. The default is zero.",
        "a_B": "Arsenic-induced ALKBH5 inhibition is assumed to be zero by default.",
        "nrf2": "Assumed amplitude of direct antioxidant induction.",
    }
    records = {}
    for name, value in asdict(params).items():
        if name in derived:
            category = "derived_normalization"
            basis = "Computed from independent parameters so resting biochemical abundances are normalized."
        elif name in structural:
            category = "structural_assumption"
            basis = "Specified by the assumed model structure."
        elif name in lesion:
            category = "illustrative_legacy_value"
            basis = "Illustrative value originally tuned to maxima of substitution categories. These maxima cannot calibrate the aggregate fixation index."
        elif name in growth:
            category = "illustrative_legacy_value"
            basis = "Illustrative growth value. The fit objective and matched observations are unavailable."
        elif name.startswith("M0_"):
            category = "assumed_occupancy"
            basis = "Occupancy is assumed. Regional MeRIP enrichment measures relative enrichment."
        else:
            category = "assumed_effective_parameter"
            basis = "Assumed rate, timescale or gain in nominal model units."
        records[name] = {
            "value": value,
            "reference_value": getattr(defaults, name),
            "classification": category,
            "directly_measured": False,
            "changed_from_reference": bool(value != getattr(defaults, name)),
            "basis": notes.get(name, basis),
        }
    return records


def source_code_sha256() -> str:
    """Stable digest of package Python sources, excluding compiled caches."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reproduction_metadata(params: Params, settings: SimulationSettings) -> dict:
    """Record settings, parameter sources and checksums for this run."""
    from ..processing.store import source_status
    defaults = SimulationSettings()
    canonical_model = params == normalise_params(Params())
    canonical_grid = (settings.t_end_days == defaults.t_end_days
                      and settings.n_time_points == defaults.n_time_points)
    return {
        "release_version": RELEASE_VERSION,
        "source_code_sha256": source_code_sha256(),
        "scientific_scope": "Illustrative mechanistic model without independent quantitative validation.",
        "default_calibration_status": "The default run uses illustrative parameters without fitting them to the source observations.",
        "source_status": source_status(),
        "scientific_metadata_sha256": {
            name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("source_status.json", "observation_contracts.json")
        },
        "input_database": "data/emd1.sqlite",
        "input_database_sha256": hashlib.sha256(DATABASE_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "ensemble_draws_requested": settings.n_ensemble,
        "ensemble_seed": settings.ensemble_seed,
        "counterfactual_seed": settings.counterfactual_seed,
        "time_points": settings.n_time_points,
        "time_horizon_nominal_days": settings.t_end_days,
        "canonical_model_and_grid": bool(canonical_model and canonical_grid),
        "canonical_ensemble": bool(canonical_model and canonical_grid
                                  and settings.n_ensemble == defaults.n_ensemble
                                  and settings.ensemble_seed == defaults.ensemble_seed
                                  and settings.counterfactual_seed == defaults.counterfactual_seed),
        "parameters": asdict(params),
        "parameter_provenance": parameter_provenance(params),
        "parameter_log_sigmas": FREE_SIGMA,
        "conditions": [{"key": condition.key, "exposure": condition.E,
                        "intervention": asdict(condition.iv), "plotted": condition.plotted}
                       for condition in CONDITIONS],
    }
