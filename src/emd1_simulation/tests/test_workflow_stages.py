"""Integration checks for the boundaries between the scientific workflow stages."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_numerical_engine_runs_without_database_or_optional_samplers():
    """Importing and using the numerical model must not acquire evidence data."""
    program = r'''
import importlib.abc
import sqlite3
import sys

class BlockOptionalLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'numpyro'}:
            raise ModuleNotFoundError('blocked optional sampler')

sys.meta_path.insert(0, BlockOptionalLibraries())
def blocked_database(*args, **kwargs):
    raise AssertionError('Numerical simulation unexpectedly opened SQLite')
sqlite3.connect = blocked_database

from emd1_simulation.simulation.engine import simulate_experiment
from emd1_simulation.simulation.settings import SimulationSettings, prepare_parameters
settings = SimulationSettings(t_end_days=1.0, n_time_points=3, n_ensemble=1)
result = simulate_experiment(prepare_parameters(), settings)
assert result.nominal['as']['R'].shape == (3,)
assert result.ensemble['as']['R'].shape == (1, 3)
assert result.ensemble_draws_ok == 1
assert 'emd1_simulation.processing.store' not in sys.modules
assert 'emd1_simulation.analysis.checks' not in sys.modules
assert 'emd1_simulation.reporting.figure' not in sys.modules
assert 'jax' not in sys.modules and 'numpyro' not in sys.modules
'''
    completed = subprocess.run([sys.executable, '-c', program], cwd=ROOT / 'src',
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tables_can_be_regenerated_without_running_an_ode(tmp_path, monkeypatch):
    from emd1_simulation.simulation import model, protocols
    from emd1_simulation.reporting.tables import export_tables

    from emd1_simulation.analysis.results import analyze_simulation
    from emd1_simulation.simulation.engine import simulate_experiment
    from emd1_simulation.simulation.settings import SimulationSettings, prepare_parameters

    settings = SimulationSettings(n_ensemble=1, n_time_points=9)
    summary = analyze_simulation(simulate_experiment(prepare_parameters(), settings), settings)

    def forbidden(*args, **kwargs):
        pytest.fail('Reporting must not repeat numerical integration')
    monkeypatch.setattr(model, 'simulate', forbidden)
    monkeypatch.setattr(protocols, 'run_nominal', forbidden)
    times = np.array([0.0, 45.0])
    runs = {
        'control': {'R': np.array([1.0, 1.0]), 'N': np.array([0.0, 2.0]),
                    'lnP': np.array([0.0, 0.2]), 'net_growth_rate': np.array([0.1, 0.1])},
        'as': {'R': np.array([1.0, 2.0]), 'N': np.array([0.0, 8.0]),
               'lnP': np.array([0.0, 0.4]), 'net_growth_rate': np.array([0.2, 0.2])},
    }
    paths = export_tables(summary, runs, times, tmp_path)
    assert paths['nominal_trajectories'].is_file()
    assert paths['relative_trajectories'].is_file()
    assert paths['table_catalog'].is_file()
    import pandas as pd
    relative = pd.read_csv(paths['relative_trajectories'])
    arsenic = relative[relative.condition == 'as']
    np.testing.assert_allclose(arsenic.N, [1.0, 4.0])
    np.testing.assert_allclose(arsenic.net_growth_rate, [0.1, 0.1])


def test_workflow_exposes_live_results_as_well_as_saved_outputs(tmp_path):
    from emd1_simulation.simulation.settings import SimulationSettings, prepare_parameters
    from emd1_simulation.workflow import run_workflow

    settings = SimulationSettings(t_end_days=7.0, n_time_points=21, n_ensemble=2,
                                  ensemble_seed=17, counterfactual_seed=29)
    experiment = run_workflow(tmp_path, settings, params=prepare_parameters({'k_E': 0}),
                              make_figure=False)
    assert experiment.exit_code == 1  # Intentional exposure-null exploratory run.
    assert experiment.simulation.params.k_E == 0
    assert experiment.simulation.time[-1] == 7.0
    np.testing.assert_allclose(experiment.simulation.ensemble['as']['R'], 1.0)
    assert experiment.summary['endpoints_fold_vs_control']['as']['R'] == pytest.approx(1.0)
    assert (tmp_path / 'emd1_simulation_summary.json') in experiment.paths
    assert all(path.is_file() for path in experiment.paths)


def test_nonfinite_summary_is_rejected_before_overwriting_outputs(tmp_path):
    from emd1_simulation.reporting.export import export_results
    from emd1_simulation.simulation.engine import SimulationResult
    from emd1_simulation.simulation.settings import prepare_parameters

    previous = tmp_path / 'emd1_simulation_summary.json'
    previous.write_text('{"previous": true}')
    result = SimulationResult(prepare_parameters(), np.array([0., 1.]), {}, {}, {}, 0, {})
    with pytest.raises(ValueError, match='(?i)(finite|range|json)'):
        export_results(result, {'population': float('inf')}, tmp_path,
                       make_figure=False, make_tables=False)
    assert previous.read_text() == '{"previous": true}'
    assert list(tmp_path.iterdir()) == [previous]


def test_analysis_and_reporting_use_completed_simulation_without_another_ode(tmp_path, monkeypatch):
    from emd1_simulation.simulation import model
    from emd1_simulation.simulation.engine import simulate_experiment
    from emd1_simulation.simulation.settings import SimulationSettings, prepare_parameters
    from emd1_simulation.analysis.results import analyze_simulation
    from emd1_simulation.reporting.export import export_results

    settings = SimulationSettings(t_end_days=7.0, n_time_points=9, n_ensemble=1)
    simulation = simulate_experiment(prepare_parameters(), settings)

    def forbidden(*args, **kwargs):
        pytest.fail('Analysis and reporting must use the completed simulation')
    monkeypatch.setattr(model, 'solve_ivp', forbidden)
    summary = analyze_simulation(simulation, settings)
    paths = export_results(simulation, summary, tmp_path, make_figure=False)
    assert summary['t_end_days'] == 7.0
    assert summary['ensemble_draws_ok'] == 1
    assert paths and all(path.is_file() for path in paths)


def test_analysis_rejects_settings_from_a_different_simulation():
    from dataclasses import replace
    from emd1_simulation.simulation.engine import simulate_experiment
    from emd1_simulation.simulation.model import Params
    from emd1_simulation.simulation.settings import SimulationSettings
    from emd1_simulation.analysis.results import analyze_simulation

    settings = SimulationSettings(n_ensemble=1, t_end_days=7.0, n_time_points=11)
    result = simulate_experiment(Params(), settings)
    for changed in (replace(settings, t_end_days=45.0),
                    replace(settings, ensemble_seed=17)):
        with pytest.raises(ValueError, match='settings differ'):
            analyze_simulation(result, changed)
    result.time[-1] = 8.0
    with pytest.raises(ValueError, match='time grid differs'):
        analyze_simulation(result, settings)


def test_custom_check_callback_requires_matching_role_definition():
    from emd1_simulation.simulation.engine import simulate_experiment
    from emd1_simulation.simulation.model import Params
    from emd1_simulation.simulation.settings import SimulationSettings

    with pytest.raises(ValueError, match='together'):
        simulate_experiment(Params(), SimulationSettings(n_ensemble=1),
                            check_evaluator=lambda p, runs: {'example': True})
