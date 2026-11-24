"""Source verification and offline database tests using synthetic mapping fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from emd1_simulation.paths import DATA_DIR, DATABASE_PATH
from emd1_simulation.processing import peaks, repair_features
from emd1_simulation.processing.build_database import build_snapshot, DEFAULT_SNAPSHOT
from emd1_simulation.processing.store import source_status
from emd1_simulation.processing.verified_features import chain_mapping, load_verified_feature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceFeatureTests(unittest.TestCase):
    def test_unverified_default_cannot_read_tracks(self):
        self.assertNotIn("NEDD4L", peaks.PUBLISHED_FEATURES)
        with mock.patch.object(peaks, "tracks", side_effect=AssertionError("must not read tracks")):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                peaks.published_feature_enrichment("NEDD4L")
        status = source_status()["sources"]
        self.assertIsNone(status["nedd4l_exon1"]["trusted_default_interval"])
        self.assertIsNone(status["nedd4l_exon1"]["quantitative_target_fold"])
        self.assertFalse(status["gao_mutation_category_bars"]["eligible_for_aggregate_mutation_index_calibration"])
        self.assertFalse(status["gao_a3b_3k"]["matched_to_as_ftoi_over_as"])

    def _fixture(self, directory):
        root = Path(directory)
        chain = root / "synthetic.chain"
        chain.write_text("chain 1 chr18 100000000 + 58221600 58221800 chr18 100000000 + 1000 1200 1\n200\n")
        annotation = root / "synthetic_refgene.txt"
        annotation.write_text("0\tNM_001144966.3\tchr18\t+\t900\t2000\t900\t2000\t2\t900,1800,\t1300,2000,\t0\tNEDD4L\n")
        record = {
            "schema_version": 1, "gene": "NEDD4L", "transcript": "NM_001144966.3",
            "publication_doi": "10.1038/s41467-021-22469-6",
            "published_interval": "NC_000018.10:58221610-58221760",
            "source_assembly": "hg38", "target_assembly": "hg19",
            "coordinate_convention": "0-based-half-open",
            "verification_method": "ucsc_chain_and_refseq_first_exon",
            "published_coordinate_interpretation": "printed_interval_is_0_based_half_open",
            "coordinate_interpretation_evidence": "Synthetic fixture only; not biological mapping evidence.",
            "source_interval": {"chrom": "chr18", "start": 58221610, "end": 58221760},
            "target_interval": {"chrom": "chr18", "start": 1010, "end": 1160},
            "assembly_chain": {"path": chain.name, "sha256": digest(chain),
                               "source_url": "https://example.invalid/synthetic-chain-not-biological"},
            "target_refgene": {"path": annotation.name, "sha256": digest(annotation),
                               "source_url": "https://example.invalid/synthetic-annotation-not-biological"},
        }
        path = root / "synthetic_feature.json"
        path.write_text(json.dumps(record))
        return path, record

    def test_complete_mapping_and_first_exon_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._fixture(tmp)
            feature = load_verified_feature(path)
            self.assertEqual((feature["chrom"], feature["start_0based"], feature["end_0based"]),
                             ("chr18", 1010, 1160))
            self.assertEqual(feature["status"], "mapping_verified_against_supplied_evidence")

    def test_boolean_flag_does_not_count_as_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unverified.json"
            path.write_text(json.dumps({"gene": "NEDD4L", "verified": True}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_verified_feature(path)

    def test_changed_chain_and_wrong_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, r = self._fixture(tmp)
            r["target_interval"]["start"] += 1
            path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, "does not match.*chain"):
                load_verified_feature(path)
            path, r = self._fixture(tmp)
            (Path(tmp) / r["assembly_chain"]["path"]).write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_verified_feature(path)

    def test_annotation_must_contain_cited_first_exon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, r = self._fixture(tmp)
            annotation = Path(tmp) / r["target_refgene"]["path"]
            annotation.write_text(annotation.read_text().replace("900,1800,", "1200,1800,"))
            r["target_refgene"]["sha256"] = digest(annotation)
            path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, "outside.*first exon"):
                load_verified_feature(path)

    def test_split_and_ambiguous_chain_maps_are_not_silently_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = Path(tmp) / "split.chain"
            chain.write_text("chain 1 chr18 100000000 + 58221600 58221800 chr18 100000000 + 1000 1200 1\n50 0 0\n150\n")
            with self.assertRaisesRegex(ValueError, "unique single-block"):
                chain_mapping(chain, ("chr18", 58221610, 58221760))
            one = "chain 1 chr18 100000000 + 58221600 58221800 chr18 100000000 + 1000 1200 1\n200\n"
            two = "chain 1 chr18 100000000 + 58221600 58221800 chr18 100000000 + 2000 2200 2\n200\n"
            chain.write_text(one + two)
            with self.assertRaisesRegex(ValueError, "found 2"):
                chain_mapping(chain, ("chr18", 58221610, 58221760))

    def test_malformed_or_unfinished_chains_cannot_verify_an_early_match(self):
        header = "chain 1 chr18 100000000 + 58221600 58222000 chr18 100000000 + 1000 1400 1\n"
        cases = {
            "block_exceeds_bounds": "chain 1 chr18 100000000 + 58221600 58221620 chr18 100000000 + 1000 1020 1\n200\n",
            "missing_terminal_at_eof": header + "200 0 0\n",
            "missing_terminal_before_blank": header + "200 0 0\n\n",
            "missing_terminal_before_header": header + "200 0 0\n" + header,
            "short_terminal": header + "200\n",
            "negative_gap": header + "200 -1 0\n200\n",
            "invalid_header_bounds": header.replace("58221600 58222000", "58222000 58221600") + "400\n",
            "invalid_source_strand": header.replace("100000000 + 58221600", "100000000 - 58221600") + "400\n",
            "nonterminal_at_end": header + "400 0 0\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            chain = Path(tmp) / "malformed.chain"
            for name, contents in cases.items():
                with self.subTest(name=name):
                    chain.write_text(contents)
                    with self.assertRaises(ValueError):
                        chain_mapping(chain, ("chr18", 58221610, 58221760))


class TrackExtractionTests(unittest.TestCase):
    def test_exploratory_window_keeps_exact_location_with_small_bins(self):
        signal = np.ones(20)
        signal[1:13] = 10.0
        cols = {}
        for cond in ("Ctl", "As", "AsT"):
            for rep in (1, 2):
                cols[f"{cond}_IP{rep}"] = signal.copy()
                cols[f"{cond}_input{rep}"] = np.full(20, 2.0)
        profile = pd.DataFrame(cols)
        profile.attrs["bin_widths"] = np.full(20, 25)
        positions = 1000 + np.arange(20) * 25
        with mock.patch.object(peaks, "profiles", side_effect=lambda *args: (profile.copy(), positions.copy())):
            best = peaks.peak_enrichment("APOBEC3B", binsize=25)
            self.assertEqual(best["peak_start_bin"], 1)
            self.assertEqual(best["peak_start_0based"], 1025)
            self.assertEqual(best["peak_kb"], "0.0")
            self.assertAlmostEqual(best["Ctl_enr"], 5.0)
            table = peaks.region_table("APOBEC3B", binsize=25)
            row = table.loc[table.region.str.startswith("best window") & table.cond.eq("Ctl")].iloc[0]
            self.assertAlmostEqual(row.enr, 5.0)

    def test_sparse_ip_and_dense_input_with_equal_integrals_have_equal_bin_signal(self):
        arrays = {"input.bw": [1.0] * 100, "ip.bw": [10.0] * 10 + [None] * 90,
                  "partial.bw": [1.0] * 100 + [2.0] * 50}

        class FakeBW:
            def __init__(self, path): self.values = arrays[path]
            def stats(self, chrom, start, end, *, type, exact):
                if type != "sum" or exact is not True:
                    raise AssertionError("Scientific bin extraction must request exact sums")
                return [sum(v for v in self.values[start:end] if v is not None)]
            def close(self): pass

        fake = types.SimpleNamespace(open=lambda p: FakeBW(p))
        with mock.patch.dict("sys.modules", {"pyBigWig": fake}):
            inp = peaks.binned(peaks.Track(Path("input.bw"), "Ctl", "input", 1, 1e9), "chr1", 0, 100, 100)
            ip = peaks.binned(peaks.Track(Path("ip.bw"), "Ctl", "IP", 1, 1e9), "chr1", 0, 100, 100)
            np.testing.assert_allclose(inp, [1.0])
            np.testing.assert_allclose(ip / inp, [1.0])
            partial = peaks.binned(peaks.Track(Path("partial.bw"), "Ctl", "input", 1, 1e9), "chr1", 0, 150, 100)
            np.testing.assert_allclose(partial, [1.0, 2.0])

    def _source_tracks(self, root):
        folder = Path(root) / "GSE145923" / "bw"
        folder.mkdir(parents=True)
        for tag in ("noAs", "chrAs", "mouseAs"):
            for kind in ("ip", "input"):
                for rep in (1, 2):
                    (folder / f"GSM_fake_As-{kind}-{tag}{rep}.bw").touch()
        return folder

    def test_missing_both_sides_of_one_replicate_is_rejected(self):
        class FakeBW:
            def header(self): return {"sumData": 1000.0}
            def close(self): pass
        fake = types.SimpleNamespace(open=lambda p: FakeBW())
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._source_tracks(tmp)
            with mock.patch.dict("sys.modules", {"pyBigWig": fake}), mock.patch.object(peaks, "BW", folder):
                self.assertEqual(len(peaks.tracks()), 12)
                for kind in ("ip", "input"):
                    (folder / f"GSM_fake_As-{kind}-chrAs2.bw").unlink()
                with self.assertRaisesRegex(ValueError, "replicates 1 and 2"):
                    peaks.tracks()
            with mock.patch.dict("sys.modules", {"pyBigWig": fake}), mock.patch.object(repair_features, "DATA", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "incomplete.*replicates"):
                    repair_features._open_tracks(repair_features.SERIES[0])


class CuratedSnapshotTests(unittest.TestCase):
    def test_offline_rebuild_preserves_all_tables_and_unresolved_status(self):
        before = digest(DATA_DIR / "source_status.json")
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = Path(tmp) / "rebuilt.sqlite"
            counts = build_snapshot(rebuilt)
            self.assertEqual(counts["chase_decay_constants"], 162)
            self.assertEqual(counts["digitised_endpoints"], 7)
            self.assertEqual(counts["repair_edge_summary"], 8)
            with sqlite3.connect(rebuilt) as actual, sqlite3.connect(DATABASE_PATH) as expected:
                self.assertEqual(actual.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(actual.execute("PRAGMA foreign_key_check").fetchall(), [])
                for name in counts:
                    self.assertEqual(actual.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall(),
                                     expected.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall())
            # Determinism is checked in the current SQLite runtime, not asserted across versions.
            other = Path(tmp) / "rebuilt_again.sqlite"
            build_snapshot(other)
            self.assertEqual(digest(rebuilt), digest(other))
        self.assertEqual(before, digest(DATA_DIR / "source_status.json"))
        self.assertIsNone(source_status()["sources"]["nedd4l_exon1"]["quantitative_target_fold"])

    def test_tampered_snapshot_is_rejected_without_replacing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            shutil.copytree(DEFAULT_SNAPSHOT, snapshot)
            file = snapshot / "digitised_endpoints.csv"
            file.write_text(file.read_text() + "tampered\n")
            destination = Path(tmp) / "prior.sqlite"
            destination.write_bytes(b"prior file must survive failed verification")
            before = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_snapshot(destination, snapshot)
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
