"""Scientific posterior reporting can be verified without an optional sampler."""
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from emd1_simulation.analysis.bayesian_diagnostics import format_diagnostics
from emd1_simulation.analysis.bayes import factorial_contrasts


class BayesianDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.statistics = {
            "mu": {"r_hat": 1.002, "n_eff": 1200},
            "delta": {"r_hat": np.array([1.08, 1.01]), "n_eff": np.array([54, 270])},
            "sigma": {"r_hat": 1.001, "n_eff": 900},
            "tau_arm": {"r_hat": 1.5, "n_eff": 5},
            "b_panel": {"r_hat": 1.03, "n_eff": 200},
        }

    def test_default_diagnostics_include_hierarchy_and_panel_offsets(self):
        result = format_diagnostics(self.statistics, divergences=3)
        self.assertIn("coverage = all supplied sites (5/5;", result)
        self.assertIn("tau_arm", result)
        self.assertIn("b_panel", result)
        self.assertIn("max r_hat = 1.5000", result)
        self.assertIn("min ESS = 5", result)
        self.assertIn("divergences = 3", result)

    def test_explicit_subset_is_labelled_and_does_not_claim_global_coverage(self):
        result = format_diagnostics(self.statistics, names=("mu", "delta", "sigma"))
        self.assertIn("coverage = selected subset (3/5;", result)
        self.assertIn("max r_hat = 1.0800", result)
        self.assertIn("min ESS = 54", result)
        self.assertNotIn("all supplied sites", result)

    def test_nonfinite_hierarchical_diagnostic_is_not_silently_omitted(self):
        self.statistics["tau_arm"]["r_hat"] = np.nan
        result = format_diagnostics(self.statistics)
        self.assertIn("max r_hat = unavailable (undefined diagnostic)", result)
        self.assertIn("min ESS = 5", result)

    def test_missing_or_undefined_diagnostics_do_not_imply_convergence(self):
        statistics = {"mu": {"r_hat": np.nan, "n_eff": 400},
                      "delta": {"r_hat": [1.01, np.nan], "n_eff": 200},
                      "sigma": {"r_hat": 1.001}}
        result = format_diagnostics(statistics)
        self.assertIn("max r_hat = unavailable (undefined diagnostic)", result)
        self.assertIn("min ESS = unavailable (missing diagnostic)", result)
        self.assertIn("unavailable", format_diagnostics({}))
        selected = format_diagnostics({"mu": statistics["mu"]}, names=("mu", "sigma"))
        self.assertIn("missing selected sites", selected)
        self.assertIn("missing requested prefixes = sigma", selected)

    def test_empty_explicit_subset_is_rejected(self):
        with self.assertRaises(ValueError):
            format_diagnostics(self.statistics, names=())

    def test_positive_interaction_does_not_establish_abolition(self):
        result = factorial_contrasts([-2.] * 4, [.5] * 4, [-1.] * 4)
        self.assertEqual(result["probability_interaction_positive"], 1.)
        self.assertEqual(result["interaction_log_p5_p50_p95"], [.5, .5, .5])
        self.assertEqual(result["conditional_fto_log_p5_p50_p95"], [-1.5, -1.5, -1.5])
        self.assertEqual(result["probability_conditional_fto_negative"], 1.)
        self.assertEqual(set(result["equivalence"]), {"status"})
        self.assertIn("not assessed", result["equivalence"]["status"])

    def test_equivalence_requires_explicit_justified_margin(self):
        for margin, justification in [(0., "reason"), (np.nan, "reason"), (.1, None),
                                       (.1, " "), (None, "reason")]:
            with self.subTest(margin=margin, justification=justification):
                with self.assertRaises(ValueError):
                    factorial_contrasts([0.], [0.], [0.],
                                        equivalence_log_margin=margin,
                                        equivalence_justification=justification)
        result = factorial_contrasts([-.5, -.5], [.5, .5], [.45, 1.],
                                     equivalence_log_margin=.1,
                                     equivalence_justification="Prespecified example for this unit test")
        self.assertEqual(result["equivalence"]["conditional_fto_probability_within_margin"], .5)
        self.assertIn("no automatic decision", result["equivalence"]["status"])

    def test_noncorresponding_or_nonfinite_draws_are_rejected(self):
        for fto, writer, both in [([1.], [1., 2.], [1.]), ([], [], []),
                                  ([np.nan], [1.], [1.])]:
            with self.assertRaises(ValueError):
                factorial_contrasts(fto, writer, both)

    def test_import_and_contrasts_do_not_load_optional_backend(self):
        source = Path(__file__).resolve().parents[2]
        program = r'''
import importlib.abc
import sys
class BlockOptionalLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'numpyro'}:
            raise ModuleNotFoundError('blocked optional sampler')
sys.meta_path.insert(0, BlockOptionalLibraries())
from emd1_simulation.analysis import bayes
bayes.factorial_contrasts([-2.], [.5], [-1.])
assert 'jax' not in sys.modules and 'numpyro' not in sys.modules
try:
    bayes._load_backend()
except ImportError as error:
    assert 'requirements-bayes.txt' in str(error)
    assert 'offline simulation does not need' in str(error)
else:
    raise AssertionError('Missing optional dependencies were not reported')
'''
        completed = subprocess.run([sys.executable, '-c', program], cwd=source,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
