#!/usr/bin/env python3
"""Run the offline scientific regression tests for EMD1."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
MODULES = (
    "emd1_simulation.tests.test_model_scientific_corrections",
    "emd1_simulation.tests.test_evidence_corrections",
    "emd1_simulation.tests.test_bayesian_diagnostics",
    "emd1_simulation.tests.test_source_corrections",
    "emd1_simulation.tests.test_release_scientific_corrections",
)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Write the results to this JSON file")
    args = parser.parse_args()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    from emd1_simulation.reporting.provenance import RELEASE_VERSION, source_code_sha256
    from emd1_simulation.tests.test_release_scientific_corrections import NUMERICAL_DIAGNOSTICS
    record = {
        "release_version": RELEASE_VERSION,
        "scope": "Model, source-data and workflow regression tests",
        "runner": "Python standard-library unittest",
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "source_code_sha256": source_code_sha256(),
        "tests_run": result.testsRun,
        "passed": result.wasSuccessful(),
        "passed_test_ids": result.passed_ids,
        "failures": [{"test": test.id(), "traceback": error} for test, error in result.failures],
        "errors": [{"test": test.id(), "traceback": error} for test, error in result.errors],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "elapsed_seconds": time.perf_counter() - start,
        "independent_numerical_diagnostics": NUMERICAL_DIAGNOSTICS,
        "pytest_suite_executed_by_this_command": False,
        "canonical_400_draw_run_executed_by_this_command": False,
        "optional_bayesian_sampling_executed": False,
        "original_raw_source_reanalysis_executed": False,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Verification record: {args.output.resolve()}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
