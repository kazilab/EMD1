"""Independent numerical checks and end-to-end tests of the EMD1 workflow."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from emd1_simulation.paths import PROJECT_ROOT, DATA_DIR
from emd1_simulation.simulation.model import Params, IDX, rhs, control_steady_state, simulate, normalise_params
from emd1_simulation.simulation.conditions import CONDITIONS
from emd1_simulation.simulation.settings import SimulationSettings
from emd1_simulation.reporting.provenance import parameter_provenance, RELEASE_VERSION

NUMERICAL_DIAGNOSTICS = {}


class IndependentNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = normalise_params(Params())
        cls.t = np.linspace(0, 45, 91)
        cls.y0 = control_steady_state(cls.p)
        cls.solutions = {}
        for condition in CONDITIONS:
            cls.solutions[condition.key] = simulate(
                cls.p, condition.iv, condition.E, cls.t, y0=cls.y0)

    def test_control_biochemical_fixed_point(self):
        control = CONDITIONS[0]
        derivative = rhs(0, self.y0, self.p, control.iv, control.E)
        # Lesion/fixation and population bookkeeping are not biochemical fixed points.
        maximum = float(np.max(np.abs(derivative[:12])))
        NUMERICAL_DIAGNOSTICS["maximum_control_biochemical_derivative"] = maximum
        self.assertLess(maximum, 1e-10)

    def test_all_nominal_trajectories_are_finite_and_physical(self):
        for key, solution in self.solutions.items():
            with self.subTest(condition=key):
                self.assertTrue(np.isfinite(solution).all())
                self.assertGreaterEqual(float(solution[:IDX["lp"]].min()), -1e-8)
                for name in ("M_a3b", "M_ned", "M_rep", "M_aox"):
                    occupancy = solution[IDX[name]]
                    self.assertGreaterEqual(float(occupancy.min()), -1e-8)
                    self.assertLessEqual(float(occupancy.max()), 1 + 1e-8)
                self.assertGreaterEqual(float(np.diff(solution[IDX["N"]]).min()), -1e-9)

    def test_independent_high_accuracy_solver_matches_all_arms(self):
        errors = {}
        for condition in CONDITIONS:
            with self.subTest(condition=condition.key):
                independent = solve_ivp(
                    rhs, (self.t[0], self.t[-1]), self.y0, t_eval=self.t,
                    args=(self.p, condition.iv, condition.E), method="DOP853",
                    rtol=1e-10, atol=1e-12)
                self.assertTrue(independent.success)
                default = self.solutions[condition.key]
                error = float(np.max(np.abs(default - independent.y) / (1 + np.abs(independent.y))))
                errors[condition.key] = error
                self.assertLess(error, 2e-6)
        NUMERICAL_DIAGNOSTICS["DOP853_vs_LSODA_scaled_errors"] = errors

    def test_lesion_fixation_flux_has_one_sink_and_one_source(self):
        p = self.p
        maximum = 0.0
        for condition in CONDITIONS:
            solution = self.solutions[condition.key]
            for index in (0, len(self.t) // 2, len(self.t) - 1):
                y = solution[:, index]
                dy = rhs(self.t[index], y, p, condition.iv, condition.E)
                # The default repair transfer weight gives a lesion-removal multiplier of one.
                generation = p.k_ROS * y[IDX["R"]] + p.k_A3B * y[IDX["T_a3b"]]
                removal = p.k_rep * y[IDX["L"]] / (p.K_L + y[IDX["L"]])
                error = abs(float(dy[IDX["L"]] + dy[IDX["N"]] - (generation - removal)))
                maximum = max(maximum, error)
        NUMERICAL_DIAGNOSTICS["maximum_lesion_plus_fixation_flux_error"] = maximum
        self.assertLess(maximum, 1e-10)


class CorrectedReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from emd1_simulation.workflow import run_workflow
        cls.directory = tempfile.TemporaryDirectory(prefix="emd1-release-check-")
        cls.output = Path(cls.directory.name)
        cls.log = io.StringIO()
        with redirect_stdout(cls.log):
            cls.experiment = run_workflow(
                cls.output, SimulationSettings(n_ensemble=2, n_time_points=91),
                make_figure=False, make_tables=True, strict_evidence=True)
        cls.summary = cls.experiment.summary

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_strict_evidence_rejects_unvalidated_run_even_when_checks_pass(self):
        self.assertEqual(self.experiment.exit_code, 2)
        self.assertEqual(self.summary["model_checks_passed"], len(self.summary["model_checks"]))
        self.assertFalse(self.summary["independent_biological_validation_performed"])
        self.assertFalse(self.summary["scientific_evidence_audit"]["quantitatively_validated"])
        self.assertNotIn("out_of_calibration_checks_passed", self.summary)

    def test_all_parameters_are_classified_without_claimed_measurements(self):
        metadata = self.summary["reproducibility"]
        records = metadata["parameter_provenance"]
        self.assertEqual(set(records), set(asdict(Params())))
        self.assertTrue(all(not record["directly_measured"] for record in records.values()))
        self.assertEqual(records["R_ref"]["classification"], "derived_normalization")
        self.assertEqual(records["gam_d2"]["classification"], "assumed_effective_parameter")
        table = pd.read_csv(self.output / "tables/parameters.csv")
        self.assertEqual(len(table), len(records))
        self.assertIn("classification", table.columns)
        self.assertFalse(table.directly_measured.any())

    def test_source_contracts_and_quarantine_are_exported(self):
        status = self.summary["reproducibility"]["source_status"]["sources"]["nedd4l_exon1"]
        self.assertIsNone(status["trusted_default_interval"])
        self.assertIsNone(status["quantitative_target_fold"])
        contracts = pd.read_csv(self.output / "tables/observation_contracts.csv")
        self.assertFalse(contracts.quantitative_fit_eligible.any())
        self.assertGreaterEqual(len(contracts), 8)
        status_table = pd.read_csv(self.output / "tables/source_status.csv")
        self.assertIn("nedd4l_exon1", set(status_table.source))

    def test_effective_activities_have_units_in_native_and_relative_exports(self):
        native = pd.read_csv(self.output / "tables/nominal_trajectories.csv")
        ftoi = native[native.condition == "as_ftoi"]
        np.testing.assert_allclose(ftoi.Fo_effective, 0.4 * ftoi.Fo)
        a5 = native[native.condition == "as_a5kd"]
        np.testing.assert_allclose(a5.Ab_effective, 0.3 * a5.Ab)
        dictionary = pd.read_csv(self.output / "tables/observable_dictionary.csv")
        trace_names = set(native) - {"condition", "time_nominal_days"}
        self.assertEqual(trace_names, set(dictionary.observable))
        self.assertTrue((self.output / "tables/source_protocol_trajectories.csv").is_file())

    def test_comparison_exports_do_not_imply_prdx5_prediction_or_time_match(self):
        table = pd.read_csv(self.output / "tables/scientific_comparisons.csv")
        prdx = table[table.comparison.str.startswith("PRDX5")]
        self.assertEqual(len(prdx), 3)
        self.assertTrue(prdx.model_value.isna().all())
        self.assertIn("without arsenic", table.iloc[0].comparison)
        stage = pd.read_csv(self.output / "tables/antioxidant_chase_by_stage.csv")
        self.assertEqual(set(stage.experimental_stage), {"8w", "22w", "22+12w"})
        self.assertIn("eraser_branch_siMETTL3_PRDX5", stage)
        self.assertIn("writer_branch_siFTO_SOD1", stage)
        self.assertFalse(any("p_vs_no_effect" in key or "nearest" in key for key in stage.columns))

    def test_json_roundtrip_preserves_roles_and_metadata(self):
        saved = json.loads((self.output / "emd1_simulation_summary.json").read_text())
        self.assertEqual(saved["schema_version"], 3)
        self.assertEqual(saved["model_checks"]["V5"]["role"], "qualitative assumed-graph consistency")
        self.assertEqual(saved["model_checks"]["V9"]["role"], "indirect pathway support")
        self.assertEqual(saved["reproducibility"]["release_version"], RELEASE_VERSION)
        self.assertEqual(saved["scientific_evidence_audit"]["comparison_rows"],
                         self.summary["scientific_evidence_audit"]["comparison_rows"])

        expected_hashes = {
            name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
            for name in ("source_status.json", "observation_contracts.json")
        }
        self.assertEqual(saved["reproducibility"]["scientific_metadata_sha256"], expected_hashes)

    def test_notebook_is_clean_and_compiles(self):
        notebook = json.loads((PROJECT_ROOT / "EMD1_reproduce_figure.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                text = "".join(cell["source"])
                compile(text, "<notebook-cell>", "exec")


if __name__ == "__main__":
    unittest.main()
