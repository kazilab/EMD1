"""Scientific regression checks for the EMD1 model."""
from dataclasses import replace
import unittest
import numpy as np

from emd1_simulation.analysis.lesion_calibration import (
    AssayContrast, CalibrationDataRequired, MutationAssayObservation,
    NAMES, fit, fit_assay_contrasts,
)
from emd1_simulation.simulation.model import (
    ANTIOXIDANT_SCOPE, Params, apobec3b_bounds,
)
from emd1_simulation.simulation.protocols import run_nominal, run_source_protocols


class ModelScientificCorrections(unittest.TestCase):
    def test_unexposed_decay_and_abundance_bounds_match_complete_fto_loss(self):
        p = Params()
        bounds = apobec3b_bounds(p)
        self.assertAlmostEqual(bounds["decay_ratio_upper"], 11.0 / 4.5)
        self.assertAlmostEqual(bounds["abundance_ratio_lower"], 4.5 / 11.0)
        self.assertLess(bounds["decay_ratio_upper"], 3.3)
        runs = run_source_protocols(p, np.array([0.0, 45.0]))
        ctl, loss = runs["control"], runs["fto_complete_loss"]
        rate_ratio = ((1 + p.gam_d2 * loss["M_a3b"][-1])
                      / (1 + p.gam_d2 * ctl["M_a3b"][-1]))
        self.assertAlmostEqual(rate_ratio, bounds["decay_ratio_upper"], places=7)
        self.assertAlmostEqual(loss["A3B"][-1] / ctl["A3B"][-1],
                               bounds["abundance_ratio_lower"], places=6)
        self.assertEqual(loss["Fo_effective"][-1], 0.0)
        self.assertGreater(loss["Fo"][-1], 0.99)
        self.assertAlmostEqual(runs["ftoi"]["Fo_effective"][-1],
                               0.4 * runs["ftoi"]["Fo"][-1])

    def test_bounds_state_reference_and_synthesis_assumptions(self):
        p = Params()
        reference = 0.2
        result = apobec3b_bounds(p, reference_occupancy=reference, synthesis_multiplier=0.5)
        self.assertAlmostEqual(result["decay_ratio_upper"], 11.0 / 3.0)
        self.assertAlmostEqual(result["abundance_ratio_lower"], 0.5 * 3.0 / 11.0)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                apobec3b_bounds(p, reference_occupancy=invalid)

    def test_effective_erasers_switch_only_when_delayed_intervention_applies(self):
        t = np.array([0.0, 15.0, 30.0, 45.0])
        runs = run_nominal(Params(), t, intervention_start=30.0)
        fto = runs["as_ftoi"]
        alkbh5 = runs["as_a5kd"]
        np.testing.assert_allclose(fto["Fo_effective"][:2], fto["Fo"][:2])
        np.testing.assert_allclose(fto["Fo_effective"][2:], 0.4 * fto["Fo"][2:])
        np.testing.assert_allclose(alkbh5["Ab_effective"][:2], alkbh5["Ab"][:2])
        np.testing.assert_allclose(alkbh5["Ab_effective"][2:], 0.3 * alkbh5["Ab"][2:])
        # The endpoint is not a post-treatment observation when the switch is
        # declared outside the integration horizon.
        endpoint = run_nominal(Params(), t[:-1], intervention_start=30.0)
        np.testing.assert_allclose(endpoint["as_ftoi"]["Fo_effective"], endpoint["as_ftoi"]["Fo"])
        np.testing.assert_allclose(endpoint["as_a5kd"]["Ab_effective"], endpoint["as_a5kd"]["Ab"])

    def test_antioxidant_scope_does_not_claim_a_prdx5_reader_branch(self):
        self.assertEqual(ANTIOXIDANT_SCOPE["representation"], "writer_associated_stabilising_aggregate")
        self.assertEqual(set(ANTIOXIDANT_SCOPE["source_branch_genes"]),
                         {"SOD1", "SOD2", "CAT", "TXN", "GPX1"})
        self.assertEqual(ANTIOXIDANT_SCOPE["unrepresented_branch_genes"], ["PRDX5"])
        self.assertFalse(ANTIOXIDANT_SCOPE["branch_specific_predictions_supported"])


class MutationCalibrationContract(unittest.TestCase):
    @staticmethod
    def observation(condition, count=100, denominator=1000, **overrides):
        values = dict(assay_id="synthetic_assay", condition_key=condition,
                      aggregate_count=count, denominator=denominator,
                      denominator_unit="assayed_reporter_molecules", protocol_id="synthetic_matched_protocol",
                      source="Synthetic test fixture, not experimental data",
                      count_scope="all_assayed_mutations")
        values.update(overrides)
        return MutationAssayObservation(**values)

    def test_legacy_implicit_calibration_is_explicitly_rejected(self):
        with self.assertRaisesRegex(CalibrationDataRequired, "category-max calibration is disabled"):
            fit(Params(), np.array([0.0, 45.0]))

    def test_category_max_and_missing_denominators_cannot_enter_calibration(self):
        for overrides in ({"count_scope": "mutation_category_maximum"},
                          {"denominator": 0}, {"denominator": float("nan")},
                          {"denominator_unit": ""}, {"source": ""}):
            with self.assertRaises(CalibrationDataRequired):
                self.observation("treated", **overrides)

    def test_assay_contrast_rejects_unmatched_protocol_or_denominator_units(self):
        reference = self.observation("reference")
        for overrides in ({"protocol_id": "different_protocol"},
                          {"assay_id": "different_assay"}, {"denominator_unit": "cells"}):
            with self.assertRaises(CalibrationDataRequired):
                AssayContrast("unmatched", self.observation("treated", **overrides), reference)

    def test_log_ratio_fit_rejects_zero_count_instead_of_inventing_pseudocount(self):
        contrast = AssayContrast("zero", self.observation("treated", count=0),
                                 self.observation("reference"))
        with self.assertRaisesRegex(CalibrationDataRequired, "pseudocount"):
            _ = contrast.rate_ratio

    def test_explicit_fit_uses_aggregate_rates_with_denominators(self):
        # Raw counts give 240/90, but the correct rate ratio is
        # (240/1200)/(90/900) = 2. The synthetic observation map makes the
        # free k_ROS parameter equal that rate ratio, verifying the contract.
        contrast = AssayContrast("synthetic_contrast",
                                 self.observation("treated", 240, 1200),
                                 self.observation("reference", 90, 900))
        base = Params()
        fitted, report = fit_assay_contrasts(
            base, np.array([0.0, 1.0]), contrasts=[contrast],
            observation_model=lambda p, t, obs: (0.1 * p.k_ROS
                if obs.condition_key == "treated" else 0.1),
            observation_model_description="Synthetic algebraic rate map for contract verification only",
            regularization_anchor={"k_ROS": 1.0}, regularization_weight=0.0,
            fixed={name: getattr(base, name) for name in NAMES if name != "k_ROS"})
        self.assertAlmostEqual(fitted.k_ROS, 2.0, places=6)
        self.assertAlmostEqual(report["observed_rate_ratios"]["synthetic_contrast"], 2.0)
        self.assertIn("not established", report["identifiability"])
        self.assertIn("not estimated", report["uncertainty"])

    def test_fit_requires_explicit_observation_mapping_and_anchor(self):
        contrast = AssayContrast("test", self.observation("treated"), self.observation("reference"))
        with self.assertRaisesRegex(CalibrationDataRequired, "callable"):
            fit_assay_contrasts(Params(), np.array([0.0, 1.0]), contrasts=[contrast],
                observation_model=None, observation_model_description="test", regularization_anchor={})
        with self.assertRaisesRegex(CalibrationDataRequired, "every free"):
            fit_assay_contrasts(Params(), np.array([0.0, 1.0]), contrasts=[contrast],
                observation_model=lambda p, t, obs: 1.0, observation_model_description="test",
                regularization_anchor={})

    def test_fixed_fit_normalizes_base_before_observation_mapping(self):
        contrast = AssayContrast("test", self.observation("treated"), self.observation("reference"))
        base = replace(Params(), gam_d2=20.0)
        seen_synthesis = []

        def observe(p, t, observation):
            seen_synthesis.append(p.s_a3b)
            return 1.0

        fitted, _ = fit_assay_contrasts(
            base, np.array([0.0, 1.0]), contrasts=[contrast],
            observation_model=observe, observation_model_description="Synthetic normalization check",
            regularization_anchor={}, fixed={name: getattr(base, name) for name in NAMES})
        self.assertEqual(fitted.s_a3b, 8.0)
        self.assertTrue(seen_synthesis)
        self.assertTrue(all(value == 8.0 for value in seen_synthesis))

    def test_fit_rejects_invalid_base_parameters(self):
        contrast = AssayContrast("test", self.observation("treated"), self.observation("reference"))
        base = replace(Params(), tau_M=0.0)
        with self.assertRaisesRegex(ValueError, "tau_M must be positive"):
            fit_assay_contrasts(
                base, np.array([0.0, 1.0]), contrasts=[contrast],
                observation_model=lambda p, t, obs: 1.0,
                observation_model_description="Synthetic invalid-base check",
                regularization_anchor={}, fixed={name: getattr(base, name) for name in NAMES})


if __name__ == "__main__":
    unittest.main()
