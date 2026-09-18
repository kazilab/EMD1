# EMD1 simulation

Code, derived inputs and numerical reference output for the accompanying
manuscript, which contains the methods, results and interpretation.
EMD1 is assigned to KCC4, upstream KCC5 and downstream KCC2/KCC10;
KCC3 is conditional on demonstrated repair engagement. The simulation is a
hypothesis illustration, not independent experimental validation.

## Reproduce

Use Python 3.11 or later, starting in this archive's root directory:

```bash
python -m pip install -r code/emd1_simulation/requirements.txt
cd code
python -m emd1_simulation.run --no-figure --outdir ../reproduced
```

This runs the seeded 400-draw ensemble offline using the bundled derived
inputs. Compare `reproduced/emd1_simulation_summary.json` with the reference in
`outputs/`; allow floating-point tolerance across environments. To regenerate
the figure as well, omit `--no-figure`.

From the same `code/` directory:

```bash
python -m pip install pytest
python -m pytest emd1_simulation/tests -q -p no:cacheprovider
python -m emd1_simulation.fit.lesion_calibration
python -m emd1_simulation.fit.identifiability
```

Without the original chase workbook, its parser test is skipped. The optional
Bayesian analysis requires `requirements-bayes.txt` and original workbooks;
the derived chase constants do not replace raw observations for that fit.

The offline run and tests were checked with Python 3.13.9, NumPy 2.5.1,
SciPy 1.18.0, pandas 3.0.5, Matplotlib 3.11.1, openpyxl 3.1.5 and pytest 8.4.2.

## Files

| Path | Contents |
|---|---|
| `code/emd1_simulation/` | Simulation, calibration, analysis, figure generation and tests |
| `derived/` | Four small input tables; definitions and provenance in `data/THIRD_PARTY.md` |
| `outputs/emd1_simulation_summary.json` | Numerical reference output |
| `data/` | External source manifest, optional downloader and provenance |
| `MANIFEST.sha256` | Checksums of the deposit files |

Optional source download, from the archive root:

```bash
python data/fetch_data.py
python data/fetch_data.py --verify
```

Downloads go under `code/emd1_simulation/data/`. Inputs marked `automatic=0`
in `data/SOURCES.tsv` require manual retrieval; see `data/THIRD_PARTY.md`.
The main run requires no downloads.

## Licence and citation

Software: MIT (`LICENSE`). Derived tables and outputs: CC-BY-4.0
(`LICENSE-DATA`). Third-party inputs retain their original terms.
Cite using `CITATION.cff`.
