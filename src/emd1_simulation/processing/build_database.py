#!/usr/bin/env python3
"""Build the EMD1 SQLite database from the bundled CSV exports, or use
--source-root and --output to import an original derived-data directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile


from ..paths import DATABASE_PATH, DATA_DIR

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = DATABASE_PATH
DEFAULT_SNAPSHOT = DATA_DIR / "curated_snapshot"

REFERENCES = [
    ("gao2022", "Gao et al., J Biol Chem 298(2):101563 (2022)",
     "m6A demethylation of cytidine deaminase APOBEC3B mRNA orchestrates arsenic-induced mutagenesis",
     "10.1016/j.jbc.2022.101563", "34998823", "https://pubmed.ncbi.nlm.nih.gov/34998823/",
     "Approximate readings of panels 1G, 3K and 3Q; mutation values are maxima across substitution-category bars, not total mutation burden."),
    ("cui2021", "Cui et al., Nat Commun 12:2183 (2021)",
     "Autophagy of the m6A mRNA demethylase FTO is impaired by low-level arsenic exposure to promote tumorigenesis",
     "10.1038/s41467-021-22469-6", "33846348", "https://pubmed.ncbi.nlm.nih.gov/33846348/",
     "Publication associated with GSE145923; stored repair contrasts are a local reanalysis of deposited tracks."),
    ("yang2021", "Yang et al., Proc Natl Acad Sci USA 118(35):e2025948118 (2021)",
     "METTL14 facilitates global genome repair and suppresses skin tumorigenesis",
     "10.1073/pnas.2025948118", "34452996", "https://pubmed.ncbi.nlm.nih.gov/34452996/",
     "Publication associated with GSE145924; published UVB mechanism and local track reanalysis are distinct evidence."),
    ("zhao2024", "Zhao et al., J Hazard Mater 465:133329 (2024)",
     "N6-methyladenosine promotes aberrant redox homeostasis required for arsenic carcinogenesis by controlling the adaptation of key antioxidant enzymes",
     "10.1016/j.jhazmat.2023.133329", "38142659", "https://pubmed.ncbi.nlm.nih.gov/38142659/",
     "Chase estimates derived from Fig. 5B in the source workbook, sheet 'Fig. 5 and S5'; original observations are not redistributed."),
    ("ucsc_hg19", "UCSC Genome Browser, hg19 refGene annotation snapshot", "hg19 refGene table",
     None, None, "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
     "Source snapshot identified by SHA-256 in provenance; coordinates use hg19 and 0-based half-open exon intervals."),
    ("ncbi_annotation", "NCBI, Human.GRCh38.p13 gene annotation snapshot", "Human.GRCh38.p13 annotation",
     None, None, "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz",
     "GOProcess DNA repair matching provided the repair-gene set; this GRCh38 annotation did not supply the hg19 exon coordinates."),
]

CHASE_COLUMNS = [
    ("gene", "TEXT"), ("stage", "TEXT"), ("sirna", "TEXT"), ("rep", "INTEGER"),
    ("k", "REAL"), ("half_life", "REAL"),
]
DIGITISED_COLUMNS = [
    ("arm", "TEXT"), ("paper", "TEXT"), ("panel", "TEXT"), ("quantity", "TEXT"),
    ("condition", "TEXT"), ("value", "REAL"), ("unit", "TEXT"),
    ("read_method", "TEXT"), ("p_value", "REAL"),
]
REPAIR_COLUMNS = [
    ("series", "TEXT"), ("contrast", "TEXT"), ("gene", "TEXT"),
    ("pooled_raw_ratio", "REAL"), ("background_raw_median", "REAL"), ("effect", "REAL"),
    ("effect_definition", "TEXT"), ("pct", "REAL"), ("null_lo", "REAL"), ("null_hi", "REAL"),
    ("module", "REAL"), ("n_background_drawn", "INTEGER"), ("n_background_used", "INTEGER"),
    ("n_repair", "INTEGER"), ("accession", "TEXT"), ("chrom", "TEXT"), ("strand", "TEXT"),
    ("region_bounding_start_0based", "INTEGER"), ("region_bounding_end_0based", "INTEGER"),
    ("exon_intervals_0based", "TEXT"), ("region_exon_nt", "INTEGER"),
    ("input_qc_bin_nt", "INTEGER"), ("min_control_input_bins", "INTEGER"),
    ("min_control_input_density", "REAL"), ("control_input_passing_bins_min", "INTEGER"),
    ("control_input_mean_density_min", "REAL"),
    ("index_raw_ratio_rep1", "REAL"), ("index_background_normalized_ratio_rep1", "REAL"),
    ("index_ratio_rep1", "REAL"), ("index_raw_ratio_rep2", "REAL"),
    ("index_background_normalized_ratio_rep2", "REAL"), ("index_ratio_rep2", "REAL"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, columns: list[tuple[str, str]]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = [name for name, _ in columns]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    for row in rows:
        for name, kind in columns:
            value = row[name]
            if value == "" and name == "p_value":
                row[name] = None
            elif kind == "INTEGER":
                row[name] = int(value)
            elif kind == "REAL":
                row[name] = float(value)
                if not math.isfinite(row[name]):
                    raise ValueError(f"Non-finite {name} in {path}")
            elif value == "":
                raise ValueError(f"Empty {name} in {path}")
    return rows


def create_input_table(connection, name, columns, unique):
    definition = ", ".join(f'"{column}" {kind}' for column, kind in columns)
    definition += ', reference_id TEXT NOT NULL REFERENCES "references"(reference_id)'
    definition += ', provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id)'
    definition += ", UNIQUE(" + ", ".join(f'"{column}"' for column in unique) + ")"
    connection.execute(f'CREATE TABLE "{name}" ({definition})')


def insert_input_rows(connection, name, columns, rows, reference, provenance_id):
    placeholders = ",".join("?" for _ in range(len(columns) + 2))
    values = [tuple(row[column] for column, _ in columns) + (
        reference(row) if callable(reference) else reference, provenance_id) for row in rows]
    connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', values)


def build(source_root: Path, output: Path) -> dict[str, int]:
    """Import source-summary CSVs and metadata from a derived-data directory."""
    source_root = source_root.resolve()
    output = output.resolve()
    paths = {
        "chase": source_root / "derived/chase_decay_constants.csv",
        "digitised": source_root / "derived/digitised_endpoints.csv",
        "repair": source_root / "derived/repair_edge_first_three_exons.csv",
    }
    chase = read_csv(paths["chase"], CHASE_COLUMNS)
    digitised = [row for row in read_csv(paths["digitised"], DIGITISED_COLUMNS)
                 if "34998823" in row["paper"] and row["panel"] in {"1G", "3K", "3Q"}]
    repair = read_csv(paths["repair"], REPAIR_COLUMNS)
    if (len(chase), len(digitised), len(repair)) != (162, 7, 8):
        raise ValueError("Expected exactly 162 chase estimates, 7 selected endpoints and 8 repair contrasts")
    if {(row["panel"], row["condition"]) for row in digitised} != {
        ("1G", "Ctrl shRNA"), ("1G", "Ctrl shRNA +As"), ("1G", "A3B shRNA +As"),
        ("3K", "DMSO"), ("3K", "FB23-2"), ("3Q", "DMSO +As"), ("3Q", "FB23-2 +As"),
    }:
        raise ValueError("Unexpected panel or condition in the selected digitised endpoints")
    for row in digitised:
        row["paper"] = "Gao et al., J Biol Chem 298:101563 (2022)  PMID 34998823"

    with (source_root / "data/SOURCES.tsv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="emd1-database-", dir=output.parent) as directory:
        temporary = Path(directory) / "emd1.sqlite"
        with sqlite3.connect(temporary) as connection:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.executescript('''
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE "references" (
                    reference_id TEXT PRIMARY KEY, citation TEXT NOT NULL, title TEXT NOT NULL,
                    doi TEXT, pmid TEXT, url TEXT NOT NULL, notes TEXT NOT NULL);
                CREATE TABLE provenance (
                    artifact_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, sha256 TEXT NOT NULL,
                    url TEXT, archive_member TEXT, automatic INTEGER,
                    reference_id TEXT REFERENCES "references"(reference_id),
                    role TEXT NOT NULL, transformation_notes TEXT NOT NULL);
            ''')
            connection.executemany('INSERT INTO "references" VALUES (?, ?, ?, ?, ?, ?, ?)', REFERENCES)
            metadata = [
                ("schema_version", str(SCHEMA_VERSION)),
                ("scope", "Import of the original derived tables and source metadata for EMD1."),
                ("chase_units", "k in h^-1, half_life in hours, exposure stage in experimental weeks"),
                ("repair_coordinates", "hg19, 0-based half-open exon intervals in transcriptional order"),
                ("repair_null", "5th–95th percentiles across background transcripts. Replicate uncertainty is unquantified."),
                ("digitisation", "Approximate published figure readings. Gao panels 1G and 3Q report maximum counts across substitution categories in cloned TP53 3D-PCR products."),
                ("model_time", "Nominal model days have no calibrated correspondence to experimental chase hours."),
                ("source_manifest_sha256", sha256(source_root / "data/SOURCES.tsv")),
                ("builder", "Import of derived CSVs by processing.build_database.build."),
            ]
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata)
            artifacts = [
                ("derived_chase", "derived/chase_decay_constants.csv", sha256(paths["chase"]), None, None, None, "zhao2024", "derived input",
                 "Zhao Fig. 5B, sheet 'Fig. 5 and S5'. For each gene, stage, siRNA and replicate, regress ln(positive remaining fraction) on time (0, 2, 4 and 6 h). k = −slope and half_life = ln(2)/k."),
                ("derived_digitised", "derived/digitised_endpoints.csv", sha256(paths["digitised"]), None, None, None, "gao2022", "selected derived input",
                 "Seven of 37 source CSV rows, selected by PMID 34998823 and panels 1G, 3K and 3Q. The paper field was standardised to Gao et al. Panel 3K reports APOBEC3B abundance. Panels 1G and 3Q report TP53 substitution-category maxima."),
                ("derived_repair", "derived/repair_edge_first_three_exons.csv", sha256(paths["repair"]), None, None, None, None, "derived input",
                 "Eight DDB2/XPC contrasts from fit/repair_edge.py. Use the longest spliced hg19 NM_ isoform and its first three transcriptional exons. Require at least three 100-nt bins at density ≥1 in every control input replicate. Pool library-normalised IP and input by condition. Divide each IP/input contrast by its background median. Seed 20260825 selects 600 background transcripts before coverage filtering."),
            ]
            wanted = {"hg19_refGene.txt.gz", "GSE144620/Human.GRCh38.p13.annot.tsv.gz",
                      "papers/Zhao-PMID-38142659.xlsx", "papers/JBC2022_APOBEC3B_supp.zip"}
            selected = [row for row in manifest if row["path"] in wanted or
                        row["path"].startswith(("GSE145923/bw/", "GSE145924/bw/"))]
            if len(selected) != 28:
                raise ValueError(f"Expected 28 upstream source records, found {len(selected)}")
            for index, row in enumerate(sorted(selected, key=lambda value: value["path"]), 1):
                path = row["path"]
                if path.startswith("GSE145923/"):
                    reference_id = "cui2021"
                elif path.startswith("GSE145924/"):
                    reference_id = "yang2021"
                elif "Zhao-" in path:
                    reference_id = "zhao2024"
                elif "APOBEC3B" in path:
                    reference_id = "gao2022"
                elif path.startswith("hg19_"):
                    reference_id = "ucsc_hg19"
                else:
                    reference_id = "ncbi_annotation"
                artifacts.append((f"upstream_{index:02d}", path, row["sha256"], row["url"],
                                  row["archive_member"] or None, int(row["automatic"]),
                                  reference_id, "upstream source metadata only",
                                  "Source location and checksum from data/SOURCES.tsv."))
            connection.executemany("INSERT INTO provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", artifacts)
            create_input_table(connection, "chase_decay_constants", CHASE_COLUMNS,
                               ["gene", "stage", "sirna", "rep"])
            create_input_table(connection, "digitised_endpoints", DIGITISED_COLUMNS,
                               ["panel", "condition"])
            create_input_table(connection, "repair_edge_summary", REPAIR_COLUMNS,
                               ["series", "contrast", "gene"])
            insert_input_rows(connection, "chase_decay_constants", CHASE_COLUMNS, chase,
                              "zhao2024", "derived_chase")
            insert_input_rows(connection, "digitised_endpoints", DIGITISED_COLUMNS, digitised,
                              "gao2022", "derived_digitised")
            insert_input_rows(connection, "repair_edge_summary", REPAIR_COLUMNS, repair,
                              lambda row: {"GSE145923": "cui2021", "GSE145924": "yang2021"}[row["series"]],
                              "derived_repair")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Database foreign-key verification failed")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Database integrity verification failed")
            connection.commit()
            connection.execute("VACUUM")
        temporary.replace(output)
    return {"chase_decay_constants": len(chase), "digitised_endpoints": len(digitised),
            "repair_edge_summary": len(repair), "references": len(REFERENCES),
            "provenance": len(artifacts)}


def build_snapshot(output: Path = DEFAULT_OUTPUT,
                   snapshot_dir: Path = DEFAULT_SNAPSHOT) -> dict[str, int]:
    """Restore the curated tables and verify checksums, types, row counts and foreign keys."""
    snapshot_dir, output = Path(snapshot_dir).resolve(), Path(output).resolve()
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("origin") != "export_of_supplied_curated_sqlite"
            or manifest.get("independent_raw_reextraction") is not False):
        raise ValueError("Unexpected curated-snapshot provenance/schema")
    tables = manifest.get("tables", [])
    if {t.get("name") for t in tables} != {
            "metadata", "references", "provenance", "chase_decay_constants",
            "digitised_endpoints", "repair_edge_summary"}:
        raise ValueError("Snapshot does not contain the expected complete table set")
    files = [manifest["schema"], *tables]
    for item in files:
        path = snapshot_dir / item["path"]
        if path.resolve().parent != snapshot_dir:
            raise ValueError("Snapshot file must be directly inside its declared directory")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Curated snapshot checksum mismatch: {item['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tempfile.TemporaryDirectory(prefix="emd1-snapshot-", dir=output.parent) as directory:
        temporary = Path(directory) / "emd1.sqlite"
        with sqlite3.connect(temporary) as connection:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.executescript((snapshot_dir / manifest["schema"]["path"]).read_text())
            for table in tables:
                name = table["name"]
                columns = table["columns"]
                expected = [c["name"] for c in columns]
                actual = [r[1] for r in connection.execute(f'PRAGMA table_info("{name}")')]
                if actual != expected:
                    raise ValueError(f"Snapshot schema/column mismatch for {name}")
                rows = []
                with (snapshot_dir / table["path"]).open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    if next(reader, None) != expected:
                        raise ValueError(f"Unexpected CSV header for {name}")
                    for row in reader:
                        if len(row) != len(columns):
                            raise ValueError(f"Malformed CSV row for {name}")
                        parsed = []
                        for value, column in zip(row, columns):
                            if value == manifest["null_token"]:
                                parsed.append(None)
                            elif column["type"] == "INTEGER":
                                parsed.append(int(value))
                            elif column["type"] == "REAL":
                                number = float(value)
                                if not math.isfinite(number):
                                    raise ValueError(f"Nonfinite snapshot value in {name}")
                                parsed.append(number)
                            elif column["type"] == "TEXT":
                                parsed.append(value)
                            else:
                                raise ValueError(f"Unexpected snapshot type {column['type']}")
                        rows.append(tuple(parsed))
                if len(rows) != table["rows"]:
                    raise ValueError(f"Curated snapshot row-count mismatch for {name}")
                marks = ",".join("?" for _ in columns)
                connection.executemany(f'INSERT INTO "{name}" VALUES ({marks})', rows)
                counts[name] = len(rows)
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Rebuilt snapshot has invalid foreign keys")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Rebuilt snapshot failed SQLite integrity checking")
            connection.commit()
            connection.execute("VACUUM")
        temporary.replace(output)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-root", type=Path,
                        help="Directory containing the original derived CSVs and data/SOURCES.tsv (requires --output)")
    source.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT,
                        help="Directory containing the checksummed table export (default: data/curated_snapshot)")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.source_root is not None and arguments.output is None:
        parser.error("--source-root requires an explicit --output path")
    output = arguments.output if arguments.output is not None else DEFAULT_OUTPUT
    counts = (build(arguments.source_root, output) if arguments.source_root is not None
              else build_snapshot(output, arguments.snapshot_dir))
    print(f"Built {output.resolve()} ({output.stat().st_size:,} bytes)")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
