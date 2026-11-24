"""Checks for the public archive, requiring neither network nor raw data."""
import hashlib
import io
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def fetch_module():
    from emd1_simulation.acquisition import download
    return download


def test_bundled_chase_data_work_without_workbook(monkeypatch):
    from emd1_simulation.processing import chase as zhao
    monkeypatch.setattr(zhao, 'XLSX', ROOT / 'absent_source_workbook.xlsx')
    monkeypatch.setattr(zhao, 'parse_fig5b',
                        lambda: pytest.fail('Offline reproduction must use SQLite.'))
    rates = zhao.chase_decay_rates()
    assert len(rates) == 162
    assert not rates.duplicated(['gene', 'stage', 'sirna', 'rep']).any()
    assert np.isfinite(rates.k).all()
    assert set(rates.sirna) == {'siControl', 'siMETTL3', 'siFTO'}


def test_sqlite_inputs_have_integrity_and_traceable_foreign_keys():
    from emd1_simulation.processing.store import connect_readonly, read_table

    with connect_readonly() as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
        for table, count in [('chase_decay_constants', 162),
                             ('digitised_endpoints', 7), ('repair_edge_summary', 8)]:
            assert connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == count
            # A derived measurement must be linked to both a source and its
            # extraction history; no unreferenced numbers enter the analysis.
            assert read_table(table)[['reference_id', 'provenance_id']].notna().all().all()


def test_chase_constants_preserve_units_and_series_identity():
    from emd1_simulation.processing.store import read_table

    chase = read_table('chase_decay_constants')
    assert not chase.duplicated(['gene', 'stage', 'sirna', 'rep']).any()
    assert set(chase.gene) == {'CAT', 'GPX1', 'PRDX5', 'SOD1', 'SOD2', 'TXN'}
    assert set(chase.stage) == {'8w', '22w', '22+12w'}
    assert set(chase.rep) == {1, 2, 3}
    assert np.isfinite(chase[['k', 'half_life']]).all().all()
    assert (chase.k > 0).all()
    np.testing.assert_allclose(chase.half_life, np.log(2) / chase.k, rtol=1e-12)


def test_only_required_digitised_endpoints_are_bundled():
    from emd1_simulation.processing.store import read_table

    endpoints = read_table('digitised_endpoints')
    assert set(zip(endpoints.panel, endpoints.condition)) == {
        ('3K', 'DMSO'), ('3K', 'FB23-2'),
        ('1G', 'Ctrl shRNA'), ('1G', 'Ctrl shRNA +As'), ('1G', 'A3B shRNA +As'),
        ('3Q', 'DMSO +As'), ('3Q', 'FB23-2 +As'),
    }
    rescue = endpoints[endpoints.panel == '3K'].set_index('condition').value
    assert rescue['FB23-2'] / rescue['DMSO'] == pytest.approx(0.12)
    assert set(endpoints.loc[endpoints.panel != '3K', 'unit']) == {
        'max count across substitution types'}


def test_database_is_read_only_and_missing_input_is_not_created(tmp_path):
    from emd1_simulation.processing.store import connect_readonly, read_table

    with connect_readonly() as connection:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute('CREATE TABLE unexpected_write (value INTEGER)')
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(FileNotFoundError):
        read_table('chase_decay_constants', database=missing)
    assert not missing.exists()


def test_nominal_arsenic_results_match_reported_rounded_fold_changes():
    from emd1_simulation.simulation.model import Params
    from emd1_simulation.simulation.protocols import run_nominal

    runs = run_nominal(Params(), np.linspace(0, 45, 91))
    # Values are reported in the manuscript to two decimal places.
    for key, expected in {"R": 3.36, "M_aox": 1.24, "A3B": 1.47, "NEDD4L": 0.74}.items():
        assert runs["as"][key][-1] / runs["control"][key][-1] == pytest.approx(expected, abs=0.005)

def test_download_checksum_failure_preserves_existing_file(tmp_path):
    module = fetch_module()
    destination = tmp_path / 'source.dat'
    destination.write_bytes(b'original')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        module.checked_copy(io.BytesIO(b'wrong'), destination, '0' * 64)
    assert destination.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [destination]


def test_source_verification_reports_missing_and_mismatched_files(tmp_path):
    module = fetch_module()
    expected = hashlib.sha256(b'correct').hexdigest()
    rows = [dict(path='source', sha256=expected, automatic='1')]
    assert module.verify(rows, tmp_path) == 1
    (tmp_path / 'source').write_bytes(b'wrong')
    assert module.verify(rows, tmp_path) == 1
    (tmp_path / 'source').write_bytes(b'correct')
    assert module.verify(rows, tmp_path) == 0


def test_tar_fetch_only_installs_manifest_members(tmp_path):
    module = fetch_module()
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as stream:
        for name, content in [('sample.bw', b'test source'), ('../escape', b'ignored')]:
            entry = tarfile.TarInfo(name); entry.size = len(content)
            stream.addfile(entry, io.BytesIO(content))
    rows = [dict(path='bw/sample.bw', sha256=hashlib.sha256(b'test source').hexdigest(),
                 url=archive.as_uri(), archive_member='sample.bw', automatic='1')]
    destination = tmp_path / 'download'
    module.fetch(rows, destination)
    assert (destination / 'bw/sample.bw').read_bytes() == b'test source'
    assert not (tmp_path / 'escape').exists()


def test_repair_gene_list_does_not_require_full_annotation(monkeypatch):
    from emd1_simulation.processing import datasets
    monkeypatch.setattr(datasets, 'ANNOT', ROOT / 'absent_annotation.tsv.gz')
    genes = datasets.repair_gene_symbols()
    assert len(genes) == 320
    assert {'DDB2', 'XPC'}.issubset(genes)
