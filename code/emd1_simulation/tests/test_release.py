"""Checks for the public archive, requiring neither network nor raw data."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def fetch_module():
    spec = importlib.util.spec_from_file_location('fetch_data', ROOT / 'data/fetch_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_chase_data_work_without_workbook(monkeypatch):
    from emd1_simulation.fit import zhao
    monkeypatch.setattr(zhao, 'XLSX', ROOT / 'absent_source_workbook.xlsx')
    monkeypatch.setattr(zhao, '_derived_decay_candidates',
                        lambda: [ROOT / 'derived/chase_decay_constants.csv'])
    rates = zhao.chase_decay_rates()
    assert len(rates) == 162
    assert not rates.duplicated(['gene', 'stage', 'sirna', 'rep']).any()
    assert np.isfinite(rates.k).all()
    assert set(rates.sirna) == {'siControl', 'siMETTL3', 'siFTO'}


def test_reference_nominal_endpoints_match_code():
    from emd1_simulation.model import Params
    from emd1_simulation.run import run_nominal
    reference = json.loads((ROOT / 'outputs/emd1_simulation_summary.json').read_text())
    runs = run_nominal(Params(), np.linspace(0, 45, 91))
    for arm, expected in reference['endpoints_fold_vs_control'].items():
        for observable, value in expected.items():
            if observable == 'lnP':
                actual = (runs[arm][observable][-1] - runs['control'][observable][-1]) / np.log(2)
            else:
                actual = runs[arm][observable][-1] / runs['control'][observable][-1]
            assert actual == pytest.approx(value, rel=1e-6, abs=1e-8)


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
    from emd1_simulation.fit import datasets
    monkeypatch.setattr(datasets, 'ANNOT', ROOT / 'absent_annotation.tsv.gz')
    genes = datasets.repair_gene_symbols()
    assert len(genes) == 320
    assert {'DDB2', 'XPC'}.issubset(genes)
