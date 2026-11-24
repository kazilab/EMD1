"""Regression checks for configurable simulation runs."""

from dataclasses import replace

import numpy as np

from emd1_simulation.simulation.conditions import BY_KEY
from emd1_simulation.simulation.ensemble import run_ensemble
from emd1_simulation.simulation.model import Params


def test_ensemble_respects_disabled_exposure_gain_in_custom_base():
    conditions = [BY_KEY["control"], BY_KEY["as"]]
    times = np.linspace(0.0, 45.0, 31)
    observables = ["R", "A3B", "N", "lnP"]

    custom, accepted = run_ensemble(
        conditions, observables, times, n=2, seed=20260822,
        base=replace(Params(), k_E=0.0))
    assert accepted == 2
    for observable in observables:
        null = 0.0 if observable == "lnP" else 1.0
        np.testing.assert_allclose(custom["as"][observable], null, atol=1e-9)

    default, accepted = run_ensemble(
        conditions, ["R"], times, n=2, seed=20260822)
    assert accepted == 2
    assert (default["as"]["R"][:, -1] > 1.0).all()


def test_explicit_default_base_preserves_seeded_ensemble():
    conditions = [BY_KEY["control"], BY_KEY["as"]]
    times = np.linspace(0.0, 45.0, 31)
    observables = ["R", "N", "lnP"]
    implicit, implicit_count = run_ensemble(
        conditions, observables, times, n=2, seed=20260822)
    explicit, explicit_count = run_ensemble(
        conditions, observables, times, n=2, seed=20260822, base=Params())
    assert implicit_count == explicit_count == 2
    for condition in ("control", "as"):
        for observable in observables:
            np.testing.assert_array_equal(
                implicit[condition][observable], explicit[condition][observable])


def test_tool_records_custom_settings_and_exports_reusable_tables(tmp_path):
    import json
    import pandas as pd
    import pytest
    from emd1_simulation.simulation.settings import prepare_parameters
    from emd1_simulation.workflow import reproduce

    status = reproduce(
        tmp_path, n_ensemble=2, make_figure=False, params=prepare_parameters({'k_E': 0.0}),
        t_end_days=7.0, n_time_points=21, ensemble_seed=17, counterfactual_seed=29)
    assert status == 1  # Zero exposure gain intentionally fails arsenic sign checks.
    summary = json.loads((tmp_path / 'emd1_simulation_summary.json').read_text())
    metadata = summary['reproducibility']
    assert metadata['parameters']['k_E'] == 0.0
    assert metadata['ensemble_seed'] == 17 and metadata['counterfactual_seed'] == 29
    assert metadata['time_points'] == 21 and summary['t_end_days'] == 7.0
    assert not metadata['canonical_model_and_grid'] and not metadata['canonical_ensemble']
    assert summary['endpoints_fold_vs_control']['as']['R'] == pytest.approx(1.0)
    assert summary['ensemble']['R']['p50'] == pytest.approx(1.0)
    assert not summary['delayed_intervention_scenario']['intervention_within_simulated_horizon']

    relative = pd.read_csv(tmp_path / 'tables/relative_trajectories.csv')
    assert len(relative) == 8 * 21
    assert relative.time_nominal_days.max() == 7.0
    np.testing.assert_allclose(relative.loc[relative.condition == 'as', 'R'], 1.0)
    endpoints = pd.read_csv(tmp_path / 'tables/endpoint_folds.csv')
    assert set(endpoints.loc[endpoints.observable == 'lnP', 'unit']) == {'log2 fold vs control'}
    catalog = pd.read_csv(tmp_path / 'tables/table_catalog.csv')
    assert {'nominal_trajectories', 'relative_trajectories', 'scientific_comparisons'} <= set(catalog.table)
    assert not list(tmp_path.glob('*.png'))


def test_tool_can_skip_table_and_figure_exports(tmp_path):
    from emd1_simulation.workflow import reproduce

    (tmp_path / 'emd1_simulation.png').write_bytes(b'old figure')
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'tables/endpoint_folds.csv').write_text('old generated table')
    (tmp_path / 'tables/custom_selection.csv').write_text('user selection')
    reproduce(tmp_path, n_ensemble=1, make_figure=False, make_tables=False)
    assert not (tmp_path / 'emd1_simulation.png').exists()
    assert not (tmp_path / 'tables/endpoint_folds.csv').exists()
    assert (tmp_path / 'tables/custom_selection.csv').read_text() == 'user selection'
    assert (tmp_path / 'emd1_simulation_summary.json').is_file()


def test_tool_rejects_invalid_or_derived_parameter_edits():
    import pytest
    from emd1_simulation.simulation.settings import prepare_parameters

    for changes in ({'not_a_parameter': 2}, {'R_ref': 0.2}, {'s_aox': 1},
                    {'tau_F': 0}, {'K_L': -1}, {'M0_a3b': 1.2},
                    {'k_E': float('nan')}, {'k_E': '2'}, {'k_E': True},
                    {'w_aox_fto': 2}, {'w_aox_fto': 0}, {'a_B': 1.1}, {'k_ROS': 0, 'k_A3B': 0}):
        with pytest.raises((ValueError, TypeError)):
            prepare_parameters(changes)


def test_nominal_decay_check_uses_configured_reader_gain():
    from emd1_simulation.simulation.settings import prepare_parameters
    from emd1_simulation.simulation.protocols import run_nominal
    from emd1_simulation.analysis.checks import validate

    p = prepare_parameters({'gam_ia': 0.0})
    runs = run_nominal(p, np.linspace(0, 45, 31))
    checks = {check.cid: passed for check, passed in validate(runs, p)}
    # With no occupancy-dependent stabilization, writer KD cannot change the
    # effective decay constant. The canonical V11 ratio threshold must fail.
    assert checks['V11'] is False
