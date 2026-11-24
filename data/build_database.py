#!/usr/bin/env python3
"""Build the EMD1 SQLite database from the curated input tables."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emd1_simulation.processing.build_database import build, build_snapshot, main

__all__ = ["build", "build_snapshot", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
