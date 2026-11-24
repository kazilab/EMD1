"""Read curated source measurements and their provenance from SQLite."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

import pandas as pd


from ..paths import DATABASE_PATH, DATA_DIR

SCHEMA_VERSION = 1
TABLES = frozenset({
    "metadata", "references", "provenance", "chase_decay_constants",
    "digitised_endpoints", "repair_edge_summary",
})


@contextmanager
def connect_readonly(database: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open an existing database without ever creating or modifying a file."""
    path = Path(database) if database is not None else DATABASE_PATH
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required EMD1 input database is missing: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported EMD1 database schema {version}; expected {SCHEMA_VERSION}: {path}"
            )
        yield connection
    finally:
        connection.close()


def read_table(name: str, database: str | Path | None = None) -> pd.DataFrame:
    """Return a table in its stable insertion order; table names are allowlisted."""
    if name not in TABLES:
        raise ValueError(f"Unknown EMD1 input table: {name!r}")
    with connect_readonly(database) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{name}" ORDER BY rowid', connection)


def chase_decay_constants(database: str | Path | None = None) -> pd.DataFrame:
    """162 per-series decay estimates; k is h^-1 and half_life is hours."""
    return read_table("chase_decay_constants", database)


def digitised_endpoints(database: str | Path | None = None) -> pd.DataFrame:
    """Seven approximate assay summaries; not calibrated mutation burdens."""
    return read_table("digitised_endpoints", database)


def repair_edge_summary(database: str | Path | None = None) -> pd.DataFrame:
    """Eight annotation-fixed DDB2/XPC contrasts and their source provenance."""
    return read_table("repair_edge_summary", database)


def references(database: str | Path | None = None) -> pd.DataFrame:
    return read_table("references", database)


def provenance(database: str | Path | None = None) -> pd.DataFrame:
    return read_table("provenance", database)


def source_status() -> dict:
    """Read the documented scientific limits of the source measurements."""
    path = DATA_DIR / "source_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("schema_version") != 1 or not isinstance(status.get("sources"), dict):
        raise ValueError(f"Invalid scientific source-status schema: {path}")
    return status
