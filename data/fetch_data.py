#!/usr/bin/env python3
"""Download and verify optional original source files for EMD1."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emd1_simulation.acquisition.download import (
    checked_copy, fetch, load_sources, main, sha256, verify,
)

__all__ = ["checked_copy", "fetch", "load_sources", "main", "sha256", "verify"]

if __name__ == "__main__":
    raise SystemExit(main())
