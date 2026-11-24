"""Write simulation summaries, tables and figures."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from ..simulation.engine import SimulationResult
from ..simulation.model import Params
from ..simulation.settings import DERIVED_PARAMETERS


def export_results(result: SimulationResult, summary: dict, outdir: str | Path, *,
                   make_figure: bool = True, make_tables: bool = True) -> list[Path]:
    """Reject non-finite values before writing files and remove disabled exports on reruns."""
    try:
        payload = json.dumps(summary, indent=2, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Simulation analysis contains non-finite or unsupported values; "
            "no results were exported. Review the configured parameters and horizon."
        ) from exc
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "emd1_simulation_summary.json"
    path.write_text(payload, encoding="utf-8")
    paths = [path]

    if not make_figure:
        # Keep reporting imports optional for numerical-only workflows.
        stems = ("emd1_simulation", "emd1_figure1_exposure_adaptation",
                 "emd1_figure2_reader_specificity", "emd1_figure3_conditional_repair",
                 "emd1_supplementary_figure1_downstream_outputs",
                 "emd1_supplementary_figure2_conditional_repair",
                 "emd1_figureS3_downstream_outputs", "emd1_figureS4_conditional_repair")
        for stem in stems:
            for suffix in ("pdf", "svg", "png", "eps", "tif"):
                (directory / f"{stem}.{suffix}").unlink(missing_ok=True)
            (directory / f"{stem}_legend.md").unlink(missing_ok=True)
            (directory / f"{stem}_with_key.png").unlink(missing_ok=True)
        (directory / "emd1_figure_panel_map.csv").unlink(missing_ok=True)
    if make_tables:
        from .tables import export_tables
        paths.extend(export_tables(summary, result.nominal, result.time, directory,
                                   source_protocols=result.source_protocols).values())
    else:
        from .tables import TABLE_DESCRIPTIONS
        table_dir = directory / "tables"
        for name in (*TABLE_DESCRIPTIONS, "table_catalog"):
            (table_dir / f"{name}.csv").unlink(missing_ok=True)
        if table_dir.is_dir() and not any(table_dir.iterdir()):
            table_dir.rmdir()

    if make_figure:
        from .figure import build_figures
        changes = {name: value for name, value in asdict(result.params).items()
                   if name not in DERIVED_PARAMETERS and value != getattr(Params(), name)}
        paths.extend(build_figures(
            result.nominal, result.ensemble, result.time, directory,
            cf=result.counterfactual, n_ensemble=result.ensemble_draws_ok,
            parameter_overrides=changes))
    return paths
