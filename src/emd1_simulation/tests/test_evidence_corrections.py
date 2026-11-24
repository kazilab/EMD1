"""Guard scientific comparison contexts, scope and descriptive reporting."""
import copy
import unittest

import numpy as np

from emd1_simulation.analysis.checks import CHECKS, CHECK_ROLES, check_repair_edge
from emd1_simulation.analysis.evidence import (
    antioxidant_branch_reconciliation, evidence_audit,
)
from emd1_simulation.simulation.model import Params
from emd1_simulation.simulation.protocols import run_nominal, run_source_protocols


class EvidenceCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = Params()
        cls.time = np.linspace(0., 45., 91)
        cls.runs = run_nominal(cls.params, cls.time)
        cls.protocols = run_source_protocols(cls.params, cls.time)
        cls.chase = antioxidant_branch_reconciliation(cls.params, cls.runs, 45.)
        cls.audit = evidence_audit(cls.params, cls.runs, cls.protocols,
                                   antioxidant_summary=cls.chase)

    def test_nominal_horizon_does_not_select_source_stage_or_change_inference(self):
        short = copy.deepcopy(self.chase)
        long = antioxidant_branch_reconciliation(self.params, self.runs, 238.)
        self.assertEqual(set(short["by_stage"]), {"8w", "22w", "22+12w"})
        self.assertEqual(short.pop("model_horizon_nominal_days"), 45.)
        self.assertEqual(long.pop("model_horizon_nominal_days"), 238.)
        self.assertEqual(short, long)
        forbidden = {"nearest_chase_stage", "matched_stage_contradicts_model",
                     "writer_branch_p_vs_no_effect", "direction_agrees"}
        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(self.audit)

    def test_opposite_branch_observations_remain_visible(self):
        stage = self.chase["by_stage"]["8w"]
        self.assertLess(stage["eraser_branch_siMETTL3"]["PRDX5"], 1.)
        self.assertTrue(all(v > 1. for v in stage["writer_branch_siMETTL3"].values()))
        self.assertEqual(len(stage["control_half_life_hours_by_gene"]), 6)
        self.assertNotIn("pooled_median_all_genes_all_stages", self.chase)

    def test_inhibitor_comparison_uses_unexposed_numerator_and_denominator(self):
        record = self.audit["a3b_fto_inhibition"]
        expected = (self.protocols["ftoi"]["A3B"][-1]
                    / self.protocols["control"]["A3B"][-1])
        arsenic_ratio = self.runs["as_ftoi"]["A3B"][-1] / self.runs["as"]["A3B"][-1]
        self.assertAlmostEqual(record["model_ftoi_over_unexposed_control"], expected)
        self.assertGreater(abs(expected - arsenic_ratio), .01)
        self.assertAlmostEqual(record["source_FB23_2_over_DMSO"], .12)
        self.assertAlmostEqual(record["abundance_lower_bound_over_unexposed_control"], 4.5 / 11.)
        self.assertNotIn("a3b_fto_rescue", self.audit)

    def test_missing_unexposed_scenario_never_falls_back_to_arsenic(self):
        with self.assertRaisesRegex(ValueError, "Missing unexposed"):
            evidence_audit(self.params, self.runs, {}, antioxidant_summary=self.chase)

    def test_unmodeled_prdx5_has_no_invented_model_comparison_value(self):
        rows = self.audit["comparison_rows"]
        prdx5 = [r for r in rows if r["comparison"].startswith("PRDX5")]
        self.assertEqual(len(prdx5), 3)
        self.assertTrue(all(r["model_value"] is None for r in prdx5))
        self.assertTrue(all(r["source_value"] > 0 for r in prdx5))
        for row in rows:
            self.assertEqual(set(row), {"comparison", "model_value", "model_quantity_or_unit",
                                       "source_value", "source_quantity_or_unit", "status"})

    def test_no_nominal_to_experimental_time_conversion_is_exported(self):
        record = self.audit["time_scale"]
        self.assertIn("model_arsenic_antioxidant_half_life_nominal_days", record)
        self.assertNotIn("model_arsenic_antioxidant_half_life_nominal_hours", record)
        self.assertFalse(self.audit["quantitatively_validated"])

    def test_causal_checks_carry_correct_evidence_roles(self):
        checks = {c.cid: c for c in CHECKS}
        self.assertEqual(CHECK_ROLES["V5"], "qualitative assumed-graph consistency")
        self.assertEqual(CHECK_ROLES["V9"], "indirect pathway support")
        self.assertIn("does not discriminate", checks["V5"].statement)
        self.assertIn("FTO-deleted", checks["V9"].source)
        self.assertNotIn("0.61", checks["V2"].source)
        self.assertIn("quarantined", checks["V2"].source)
        self.assertFalse(any("out-of-calibration" in role for role in CHECK_ROLES.values()))

    def test_v5_requires_ros_invariance_not_merely_no_decrease(self):
        check = next(c for c in CHECKS if c.cid == "V5")
        endpoints = {key: {name: float(values[-1]) for name, values in trace.items()}
                     for key, trace in self.runs.items()}
        self.assertTrue(check.test(endpoints))
        endpoints["as_a3bkd"]["R"] = 2. * endpoints["as"]["R"]
        self.assertFalse(check.test(endpoints))

    def test_v13_requires_model_flatness_not_membership_in_a_source_background(self):
        passed, message = check_repair_edge(1.2, 1.02, .43, .79)
        self.assertFalse(passed)
        self.assertIn("canonical zero arsenic response=False", message)
        self.assertNotIn("inside", message)
        self.assertFalse(check_repair_edge(1.1691491, 1.0327385, .5407952, .9111217)[0])
        self.assertTrue(check_repair_edge(1.0, 1.0, .43, .79)[0])
        self.assertTrue(check_repair_edge(1.0 + 1e-7, 1.0 - 1e-7, .43, .79)[0])
        self.assertFalse(check_repair_edge(1.0, 1.0, 1.0, 1.0)[0])


if __name__ == "__main__":
    unittest.main()
