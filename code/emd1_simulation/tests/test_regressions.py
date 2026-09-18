"""Regression tests for the corrected EMD1 simulation invariants.

These tests intentionally use small deterministic samples and never invoke the
CLI entry point, so running them does not regenerate figures or summaries.
"""

import csv
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from emd1_simulation.conditions import (
    CHECKS,
    CHECK_ROLES,
    BY_KEY,
    OUT_OF_CALIBRATION_CHECK_KEYS,
    REPAIR_ARSENIC_WRITER_INPUT,
    REPAIR_DDB2,
    REPAIR_FEATURE_ACCESSION,
    REPAIR_FEATURE_CHROM,
    REPAIR_FEATURE_EXONS_0BASED,
    REPAIR_FEATURE_STRAND,
    REPAIR_INPUT_QC_BIN_NT,
    REPAIR_MIN_CONTROL_INPUT_BINS,
    REPAIR_MIN_CONTROL_INPUT_DENSITY,
    REPAIR_MODULE,
    REPAIR_N_QUANTIFIED,
    REPAIR_N_REPAIR_QUANTIFIED,
    REPAIR_NULL_BY_CONTRAST,
    REPAIR_XPC,
    check_repair_edge,
)
from emd1_simulation.ensemble import FREE_SIGMA, run_ensemble, sample_params
from emd1_simulation.fit.lesion_calibration import (
    CALIBRATION_ANCHOR,
    NAMES as LESION_PARAMETER_NAMES,
    fit as fit_lesion_parameters,
)
from emd1_simulation.fit.peaks import PUBLISHED_FEATURES
from emd1_simulation.model import (
    IDX,
    Params,
    control_steady_state,
    simulate,
    trace,
)
from emd1_simulation.run import (
    run_counterfactual,
    run_nominal,
    v10_ensemble_sensitivity,
    validate,
)


T_EVAL = np.linspace(0.0, 45.0, 91)


def _repair_folds(p: Params) -> tuple[float, float, float, float]:
    """Return the four model quantities consumed by the V13 check."""
    y0 = control_steady_state(p)
    runs = {}
    for key in ("control", "as", "as_m3kd"):
        condition = BY_KEY[key]
        runs[key] = trace(
            simulate(p, condition.iv, condition.E, T_EVAL, y0=y0),
            p,
            condition.iv,
        )

    ctl_m_rep = float(runs["control"]["M_rep"][-1])
    ctl_q = float(runs["control"]["Q"][-1])
    return (
        float(runs["as"]["M_rep"][-1]) / ctl_m_rep,
        float(runs["as"]["Q"][-1]) / ctl_q,
        float(runs["as_m3kd"]["M_rep"][-1]) / ctl_m_rep,
        float(runs["as_m3kd"]["Q"][-1]) / ctl_q,
    )


def _require_source_workbook() -> None:
    """Skip when the closed-access chase workbook is not present.

    The public archive cannot redistribute it (J Hazard Mater, PMID 38142659;
    see data/THIRD_PARTY.md). The values derived from it ship as CSV instead,
    so the analysis stays checkable without the source file.
    """
    from emd1_simulation.fit.zhao import XLSX
    if not XLSX.exists():
        pytest.skip(
            f"{XLSX.name} not present: closed-access source, not redistributed. "
            "Derived values are in derived/chase_decay_constants.csv."
        )


def test_free_sigma_contains_only_effective_uncertainties() -> None:
    assert {"a_B", "K_B", "tau_B", "d0"}.isdisjoint(FREE_SIGMA)
    assert "K_L" in FREE_SIGMA

    rng = np.random.default_rng(8675309)
    draws = [sample_params(rng) for _ in range(8)]
    invariant = [
        name
        for name in FREE_SIGMA
        if np.ptp([getattr(draw, name) for draw in draws]) == 0.0
    ]
    assert invariant == [], f"declared free parameters did not vary: {invariant}"


def test_sampled_controls_use_their_own_resting_redox_reference() -> None:
    rng = np.random.default_rng(20260825)
    for _ in range(6):
        p = sample_params(rng)
        y0 = control_steady_state(p)

        assert p.R_ref == pytest.approx(p.r_basal / (p.k_R + p.k_A))
        assert y0[IDX["R"]] == pytest.approx(p.R_ref, abs=1e-9)
        for observable in ("W", "Fo", "Ab"):
            assert y0[IDX[observable]] == pytest.approx(1.0, abs=1e-9)


def test_nominal_endpoint_checks_and_v13_pass() -> None:
    runs = run_nominal(Params(), T_EVAL)
    results = validate(runs)

    assert len(CHECKS) == 12
    assert [check.cid for check, _ in results] == [f"V{i}" for i in range(1, 13)]
    assert all(passed for _, passed in results), [
        check.cid for check, passed in results if not passed
    ]

    v13_passed, _ = check_repair_edge(*_repair_folds(Params()))
    assert v13_passed


def test_only_data_covered_arms_count_as_out_of_calibration_checks() -> None:
    assert OUT_OF_CALIBRATION_CHECK_KEYS == frozenset({"as_a3bkd", "as_nedresc"})
    assert {cid for cid, role in CHECK_ROLES.items()
            if role == "out-of-calibration qualitative check"} == {
        "V5", "V9"
    }
    v10 = next(check for check in CHECKS if check.cid == "V10")
    assert "antioxidant reader" not in v10.statement.lower()


def test_nedd4l_primary_feature_is_prespecified_from_published_coordinate() -> None:
    chrom, start, end, source = PUBLISHED_FEATURES["NEDD4L"]
    assert (chrom, start, end) == ("chr18", 55712458, 55712608)
    assert end - start == 150
    assert "Fig. 3f" in source


def test_counterfactual_ensemble_samples_around_supplied_base(monkeypatch) -> None:
    import emd1_simulation.ensemble as ensemble_module

    base = replace(Params(), gam_f1r=4.0)
    seen = []

    def fake_sample(_rng, base=None):
        seen.append(base)
        return base

    monkeypatch.setattr(ensemble_module, "sample_params", fake_sample)
    out = run_counterfactual(base, T_EVAL, n=2)

    assert seen == [base, base]
    assert out["M_rep_n"] == 2
    assert out["Q_n"] == 2


def test_counterfactual_rejects_negative_draw_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        run_counterfactual(Params(), T_EVAL, n=-1)


def test_v10_ensemble_sensitivity_reports_nonrobust_writer_expansion() -> None:
    ens, n_ok = run_ensemble(
        [BY_KEY[key] for key in ("control", "as", "as_m3kd")],
        ["A", "lnP"], T_EVAL, n=40, seed=20260822)
    assert n_ok == 40
    audit = v10_ensemble_sensitivity(ens)
    assert audit["draws"] == 40
    assert 0 <= audit["passes"] <= 40
    assert len(audit["writer_kd_minus_arsenic_log2_expansion_p5_p50_p95"]) == 3


def test_ensemble_check_audit_uses_native_draw_specific_parameters() -> None:
    ens, n_ok, audit = run_ensemble(
        list(BY_KEY.values()),
        ["R", "Fo", "Ab", "M_a3b", "M_ned", "M_rep", "M_aox", "M_mean4",
         "A3B", "NEDD4L", "A", "Q", "WNT", "L", "N", "lnP"],
        T_EVAL, n=4, seed=20260822, audit_checks=True)
    assert n_ok == 4
    assert set(audit["checks"]) == {f"V{i}" for i in range(1, 14)}
    assert all(item["passes"] <= n_ok for item in audit["checks"].values())
    assert ens["as"]["A"].shape[0] == n_ok


def test_repair_edge_constants_match_reproduced_csv() -> None:
    name = "repair_edge_first_three_exons.csv"
    # working repo, then the public archive layout (derived/ beside code/)
    candidates = [Path(__file__).parents[2] / "figures/output" / name,
                  Path(__file__).parents[3] / "derived" / name]
    csv_path = next((c for c in candidates if c.exists()), None)
    assert csv_path is not None, f"Required reference CSV missing: {name}"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    contrast_key = {
        ("GSE145923", "As/Ctl"): "arsenic",
        ("GSE145923", "AsT/Ctl"): "transformed",
        ("GSE145924", "shMETTL14/shNC"): "writer_kd",
        ("GSE145924", "UVB/shNC"): "uvb",
    }
    assert len(rows) == 8
    for row in rows:
        key = contrast_key[(row["series"], row["contrast"])]
        effect = float(row["effect"])
        assert effect == pytest.approx(
            float(row["pooled_raw_ratio"]) / float(row["background_raw_median"]))
        assert row["effect_definition"] == "pooled_raw_ratio / background_raw_median"
        if row["gene"] == "DDB2":
            assert effect == pytest.approx(REPAIR_DDB2[key][0])
            assert float(row["pct"]) == pytest.approx(REPAIR_DDB2[key][1])
        else:
            assert row["gene"] == "XPC"
            assert effect == pytest.approx(REPAIR_XPC[key])
        assert float(row["null_lo"]) == pytest.approx(REPAIR_NULL_BY_CONTRAST[key][0])
        assert float(row["null_hi"]) == pytest.approx(REPAIR_NULL_BY_CONTRAST[key][1])
        assert float(row["module"]) == pytest.approx(REPAIR_MODULE[key])
        assert int(row["n_background_used"]) == REPAIR_N_QUANTIFIED[key]
        assert int(row["n_repair"]) == REPAIR_N_REPAIR_QUANTIFIED[key]
        if row["gene"] == "DDB2":
            assert row["accession"] == REPAIR_FEATURE_ACCESSION
            assert row["chrom"] == REPAIR_FEATURE_CHROM
            assert row["strand"] == REPAIR_FEATURE_STRAND
            assert row["exon_intervals_0based"] == ";".join(
                f"{start}-{end}" for start, end in REPAIR_FEATURE_EXONS_0BASED)
            assert int(row["region_exon_nt"]) == sum(
                end - start for start, end in REPAIR_FEATURE_EXONS_0BASED)
            assert int(row["input_qc_bin_nt"]) == REPAIR_INPUT_QC_BIN_NT
            assert int(row["min_control_input_bins"]) == REPAIR_MIN_CONTROL_INPUT_BINS
            assert float(row["min_control_input_density"]) == REPAIR_MIN_CONTROL_INPUT_DENSITY
        assert "paired_rep1" not in row
        assert "paired_rep2" not in row
        for rep in (1, 2):
            assert float(row[f"index_ratio_rep{rep}"]) == pytest.approx(
                float(row[f"index_background_normalized_ratio_rep{rep}"]))


@pytest.mark.parametrize(
    "broken_params",
    [
        pytest.param(replace(Params(), gam_f1r=0.0), id="reader-edge-deleted"),
        pytest.param(replace(Params(), w_rep_writer=1.0), id="arsenic-site-input-on"),
    ],
)
def test_v13_rejects_broken_structural_claims(broken_params: Params) -> None:
    passed, _ = check_repair_edge(*_repair_folds(broken_params))
    assert not passed


def test_lesion_calibration_reproduces_current_params_from_fixed_anchor() -> None:
    base = Params()
    assert any(
        not np.isclose(CALIBRATION_ANCHOR[name], getattr(base, name))
        for name in LESION_PARAMETER_NAMES
    ), "the fixed calibration anchor must remain independent of fitted Params"

    fitted, _ = fit_lesion_parameters(base, np.array([0.0, 45.0]))
    for name in LESION_PARAMETER_NAMES:
        assert getattr(fitted, name) == pytest.approx(
            getattr(base, name), rel=5e-7, abs=1e-9
        )


@pytest.mark.parametrize("n", [0, -1])
def test_run_ensemble_rejects_nonpositive_draw_counts(n: int) -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        run_ensemble([BY_KEY["control"]], ["R"], np.array([0.0, 1.0]), n=n)


def test_counterfactual_changes_only_writer_input_and_moves_repair_outputs() -> None:
    assert REPAIR_ARSENIC_WRITER_INPUT == {"w_rep_writer": 1.0}

    base = Params()
    counterfactual_params = replace(base, **REPAIR_ARSENIC_WRITER_INPUT)
    changed = {
        field.name
        for field in fields(Params)
        if getattr(base, field.name) != getattr(counterfactual_params, field.name)
    }
    assert changed == {"w_rep_writer"}

    counterfactual = run_counterfactual(base, T_EVAL)
    assert float(counterfactual["M_rep"][-1]) > 1.05
    assert float(counterfactual["Q"][-1]) > 1.05


def test_sensitivity_preserves_baseline_and_signed_zero(monkeypatch) -> None:
    from emd1_simulation.fit import identifiability as ident
    from emd1_simulation.model import normalise_params
    from emd1_simulation.ensemble import _renormalise
    p = replace(Params(), gam_ia=9.0, M0_ned=0.6, k_A=3.5)
    assert normalise_params(p) == _renormalise(p)
    nominal = Params()
    assert nominal.s_ned == pytest.approx(nominal.d_ned / (1.0 + nominal.gam_i2 * nominal.M0_ned))
    assert nominal.s_aox == pytest.approx(nominal.d_aox / (1.0 + nominal.gam_ia * nominal.M0_aox))
    assert nominal.s_a3b == pytest.approx(nominal.d_a3b * (1.0 + nominal.gam_d2 * nominal.M0_a3b))
    y_nominal = control_steady_state(nominal)
    assert y_nominal[IDX["T_ned"]] == pytest.approx(1, abs=1e-8)
    assert y_nominal[IDX["T_aox"]] == pytest.approx(1, abs=1e-8)
    q = normalise_params(p)
    state = control_steady_state(q)
    assert state[IDX['T_aox']] == pytest.approx(1, abs=1e-8)
    assert state[IDX['T_ned']] == pytest.approx(1, abs=1e-8)
    assert state[IDX['W']] == pytest.approx(1, abs=1e-8)
    values = np.ones(len(ident.SIGNED_ROWS))
    values[ident.SIGNED_ROWS] = [-0.01, 0., 0.01]
    transformed = ident.transform_observations(values)
    np.testing.assert_allclose(transformed[ident.SIGNED_ROWS], [-0.01, 0., 0.01])
    np.testing.assert_allclose(transformed[~ident.SIGNED_ROWS], 0.)


def test_sensitivity_step_convergence() -> None:
    from emd1_simulation.fit.identifiability import sensitivity
    coarse, names, _ = sensitivity(0.02)
    fine, fine_names, _ = sensitivity(0.01)
    assert names == fine_names
    assert np.isfinite(fine).all()
    assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 0.01


def test_ddb2_proxy_does_not_repair_aggregate_lesions_by_default() -> None:
    p = Params()
    base = run_nominal(p, T_EVAL)
    driven = run_nominal(replace(p, w_rep_writer=1), T_EVAL)
    assert driven['as']['Q'][-1] > base['as']['Q'][-1]
    np.testing.assert_allclose(driven['as']['Q_lesion'], 1.)
    np.testing.assert_allclose(driven['as']['N'], base['as']['N'], rtol=2e-6, atol=1e-8)
    coupled = run_nominal(replace(p, w_rep_writer=1, repair_proxy_weight=1), T_EVAL)
    assert coupled['as']['N'][-1] < driven['as']['N'][-1]


def test_fto_sensitive_repair_scenario_is_runnable() -> None:
    runs = run_nominal(replace(Params(), w_rep_fto=1), T_EVAL)
    assert runs['as']['M_rep'][-1] < runs['control']['M_rep'][-1]
    assert runs['as_ftoi']['M_rep'][-1] > runs['as']['M_rep'][-1]


@pytest.mark.parametrize('start', [0., 10., 10.25, 45., 50.])
def test_intervention_protocol_switch(start) -> None:
    from emd1_simulation.model import Intervention, simulate_protocol
    p = Params()
    c = BY_KEY['as_m3kd']
    y0 = control_steady_state(p)
    t = np.linspace(0, 45, 46)
    actual = simulate_protocol(p, c.iv, c.E, t, start, y0)
    if start == 0:
        np.testing.assert_allclose(actual, simulate(p, c.iv, c.E, t, y0))
    elif start >= t[-1]:
        np.testing.assert_allclose(actual, simulate(p, Intervention(), c.E, t, y0))
    else:
        first_t = np.unique(np.append(t[t <= start], start))
        first = simulate(p, Intervention(), c.E, first_t, y0)
        second_t = np.insert(t[t > start], 0, start)
        second = simulate(p, c.iv, c.E, second_t, first[:, -1])
        np.testing.assert_allclose(actual[:, t > start], second[:, 1:])
    assert actual.shape == (len(IDX), len(t))


def test_delayed_readouts_use_preintervention_reader_activity() -> None:
    runs = run_nominal(Params(), T_EVAL, intervention_start=30.)
    before = T_EVAL < 30
    np.testing.assert_allclose(runs['as_aoxkd']['death_rate'][before],
                               runs['as']['death_rate'][before], rtol=1e-6)
    np.testing.assert_allclose(runs['as_m3kd']['N'][before],
                               runs['as']['N'][before], rtol=1e-6)


def test_population_accounting_and_relative_suppression() -> None:
    from emd1_simulation.scientific_audit import population_summary
    t = np.linspace(0, 45, 901)
    runs = run_nominal(Params(), t)
    pop = population_summary(runs, t)
    for key, tr in runs.items():
        np.testing.assert_allclose(tr['net_growth_rate'],
                                   tr['division_rate'] - tr['death_rate'])
        assert pop[key]['integrated_division_hazard'] - pop[key]['integrated_death_hazard'] == pytest.approx(tr['lnP'][-1], abs=1e-4)
    assert pop['as_m3kd']['population_fold_vs_start'] > 1
    assert not pop['as_m3kd']['net_population_loss']
    assert runs['as_m3kd']['lnP'][-1] < runs['as']['lnP'][-1]


def test_lesion_scale_symmetry_preserves_relative_outputs() -> None:
    p = Params()
    factor = 10
    scaled = replace(p, **{name: factor * getattr(p, name)
                           for name in ('k_ROS', 'k_A3B', 'k_rep', 'K_L', 'K_Ld')})
    runs, alt = run_nominal(p, T_EVAL), run_nominal(scaled, T_EVAL)
    for key in runs:
        np.testing.assert_allclose(alt[key]['N'], factor * runs[key]['N'], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(alt[key]['lnP'], runs[key]['lnP'], rtol=2e-6, atol=1e-7)


def test_scientific_conflicts_are_not_reported_as_validation() -> None:
    from emd1_simulation.scientific_audit import evidence_audit
    p = Params(); runs = run_nominal(p, T_EVAL)
    audit = evidence_audit(p, runs)
    assert not audit['quantitatively_validated']
    assert not audit['antioxidant_fto_decay']['direction_agrees']
    rescue = audit['a3b_fto_rescue']
    assert rescue['model_ftoi_over_arsenic'] == pytest.approx(
        runs['as_ftoi']['A3B'][-1] / runs['as']['A3B'][-1])
    assert rescue['model_minimum_over_nominal_arsenic_with_fixed_synthesis'] > rescue['digitised_FB23_2_over_DMSO']


def test_chase_parser_verifies_arm_labels_rather_than_assuming_order() -> None:
    """A silent swap of siMETTL3 and siFTO would invert the antioxidant result.

    The parser used to locate 'siControl' and then read the next two triplets
    as siMETTL3 and siFTO without checking their headers, and mapped column
    blocks to exposure stages purely left-to-right. Both happen to be right in
    this workbook, so the bug was invisible; it must fail loudly instead.
    """
    _require_source_workbook()
    from emd1_simulation.fit import zhao

    df = zhao.parse_fig5b()
    assert set(df.sirna) == set(zhao.ARM_ORDER)
    assert set(df.stage) == set(zhao.STAGES)
    assert sorted(df.hour.unique().tolist()) == [0.0, 2.0, 4.0, 6.0]
    # every series is normalised to 1.0 at t = 0
    np.testing.assert_allclose(df[df.hour == 0].frac.values, 1.0)
    # 6 genes x 3 stages x 3 arms x 3 reps x 4 timepoints
    assert len(df) == 6 * 3 * 3 * 3 * 4

    grid = zhao._grid("Fig. 5 and S5")
    stage_by_col = zhao._stage_header(grid)
    assert set(stage_by_col.values()) == set(zhao.STAGES)

    swapped = [list(r) for r in grid]
    for row in swapped:
        for j, cell in enumerate(row):
            if cell == "siMETTL3":
                row[j] = "siFTO"
            elif cell == "siFTO":
                row[j] = "siMETTL3"
    zhao._grid.cache_clear() if hasattr(zhao._grid, "cache_clear") else None
    original = zhao._grid
    zhao._grid = lambda _sheet: swapped
    try:
        with pytest.raises(ValueError, match="Arm order in the workbook has changed"):
            zhao.parse_fig5b()
    finally:
        zhao._grid = original


def test_antioxidant_reconciliation_resolves_stage_and_branch() -> None:
    """The pooled chase median mixes two branches and three exposure stages.

    At the stage nearest the model horizon there is no detectable eraser effect
    on the writer branch, so the pooled 1.54x must not be reported as a
    direction disagreement with the model.
    """
    from emd1_simulation.scientific_audit import antioxidant_branch_reconciliation

    p = Params(); runs = run_nominal(p, T_EVAL)
    rec = antioxidant_branch_reconciliation(p, runs, 45.0)

    assert rec['nearest_chase_stage'] == '8w'
    assert not rec['matched_stage_contradicts_model']
    # pooled summary is retained, but flagged
    assert rec['pooled_median_all_genes_all_stages'] == pytest.approx(1.543, abs=0.01)

    early = rec['by_stage']['8w']
    late = rec['by_stage']['22+12w']
    # no detectable effect early, strong and consistent late
    assert early['writer_branch_p_vs_no_effect'] > 0.5
    assert late['writer_branch_p_vs_no_effect'] < 0.01
    assert late['writer_branch_n_faster'] == late['writer_branch_n']
    assert late['writer_branch_mean_log_ratio'] > early['writer_branch_mean_log_ratio']

    # the writer branch responds to writer knockdown at every stage: that is
    # the coupling the model encodes, and it is not in question
    for stage in rec['by_stage'].values():
        assert stage['writer_branch_median_siMETTL3'] > 1.5
