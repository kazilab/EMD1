# EMD1: illustrative mechanistic simulation

An illustrative mechanistic simulation of EMD1 (transcript-specific m⁶A
regulation under oxidative stress), with chronic arsenic as the exemplar
exposure. The model implements the proposed typed links KCC5 → EMD1 (home KCC4)
→ KCC2, KCC10, with KCC3 carried as an assumed zero arsenic input. It couples
ROS, four transcript-specific m⁶A occupancies with their own reader responses, a
DDB2/NER translation proxy, aggregate lesion removal, a cumulative
mutation-fixation index and relative population expansion, around a seeded
400-draw sensitivity ensemble over a 45 nominal-day horizon.

The mechanistic model's parameters are illustrative, not fitted. They are
relative effect sizes, not physiological rate constants: time is in nominal model
days and exposure is dimensionless. A separate data-confrontation pass compares
the model with public MeRIP-seq and digitised measurements and **reports the
disagreements it found rather than resolving them**; that pass is an evidence
audit, not a calibration, and the summary JSON records it as
`quantitatively_validated: false`.

Its 13 internal consistency checks were revised during development; they are
neither preregistered nor independent experimental validation. Internal checks
and regression tests establish that the code computes what it claims, not that
the model is biologically correct. Model output is not an independent
key-characteristic or EMD positive, and the typed EMD1–KCC links remain
conditional on the functional evidence cited in the manuscript.

## Reproduce offline

Use Python 3.11 or later, starting in this archive's root directory:

```bash
python -m pip install -r code/emd1_simulation/requirements.txt
cd code
python -m emd1_simulation.run --no-figure --outdir ../reproduced
```

This runs the seeded 400-draw ensemble offline using the bundled derived inputs
and needs no downloads. Compare `reproduced/emd1_simulation_summary.json` with
the reference in `outputs/`; allow floating-point tolerance across environments.
To regenerate the figure as well, omit `--no-figure`.

From the same `code/` directory:

```bash
python -m pip install pytest
python -m pytest emd1_simulation/tests -q -p no:cacheprovider
python -m emd1_simulation.fit.lesion_calibration
python -m emd1_simulation.fit.identifiability
```

Without the original chase workbook, its parser test is skipped. The optional
Bayesian analysis requires `requirements-bayes.txt` and original workbooks; the
derived chase constants do not replace raw observations for that fit.

The deposited run passed **13/13** internal checks, including 2/2 checks outside
the calibration set, with 400/400 ensemble draws integrating. The test suite
passed **36 of 37** tests with 1 skipped, the skip being the chase-workbook
parser noted above. Reviewed software: Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0,
pandas 3.0.5, Matplotlib 3.11.1, openpyxl 3.1.5 and pytest 8.4.2. Floating-point
results may vary slightly on other platforms.

## What the analysis shows

**Reader-dependent responses can produce opposite transcript effects from a
shared methylation driver.** Methylation is modelled separately at APOBEC3B,
NEDD4L, a DDB2-associated repair proxy and a pooled antioxidant module, and
opposite-sign responses follow from the reader functions. **Opposite signs alone
do not establish that multiple site variables are needed**; separating a
site-resolved representation from a shared-driver one requires independent
measurements that this deposit does not contain.

**`M_mean4` is not transcriptome-wide m⁶A.** It is an unweighted mean of the
four modelled occupancies. MeRIP-seq cannot measure global methylation
stoichiometry, because the IP library is sequenced to its own depth; an earlier
global-m⁶A inference drawn from median log2(IP/input) was withdrawn for that
reason, and the corresponding model claim cannot be tested against these
libraries at all.

**The EMD1 → KCC3 edge rests on an assumption the data cannot settle.** Under
the nominal zero-input assumption the DDB2/NER proxy stays flat. The edge is not
absent — it responds to a writer perturbation — and the measurement cannot
exclude a moderate effect. On the predeclared first-three-exon feature, DDB2 sits
at 0.91× at background percentile 38 of 336 quantified transcripts, with the
same-pathway XPC control also inside the null, and the repair-set median has no
power: it stays within 0.97–1.03 even where transformed-line DDB2 is 0.39×.

**The default antioxidant stability law remains a challenged hypothesis.** It
cannot reproduce both the writer-loss and the eraser-loss chase responses. Seven
unweighted repair-structure scenarios are reported instead of a single preferred
structure, and the summary exposes the unresolved conflicts in every run.

**Two source comparisons disagree with the model and are reported as
disagreements.** The modelled antioxidant half-life is 109.2 nominal hours
against an observed median chase-control half-life of 6.72 hours, in a different
experimental context. The A3B/FTO rescue is 0.621 in the model against a
digitised FB23-2/DMSO value of 0.12, an unresolved magnitude and protocol
mismatch that inhibitor strength alone does not close at nominal gains.

## Contents and provenance

| Path | Contents |
|---|---|
| `code/emd1_simulation/` | Simulation, calibration, analysis, figure generation and tests |
| `code/emd1_simulation/conditions.py` | Checks V1–V13, each with its evidentiary basis |
| `code/emd1_simulation/fit/` | Data-confrontation pass, identifiability, lesion calibration and the repair-edge reanalysis |
| `derived/` | Four small input tables; definitions and provenance in `data/THIRD_PARTY.md` |
| `outputs/emd1_simulation_summary.json` | Numerical reference output |
| `data/` | External source manifest, optional downloader and provenance |

The deposit excludes publication images, Word files and their generators, and
caches. The figure regenerates from code. The Word toolchain remains in the
working manuscript project and is unnecessary to reproduce the numerical
analysis.

Optional source download, from the archive root:

```bash
python data/fetch_data.py
python data/fetch_data.py --verify
```

Downloads go under `code/emd1_simulation/data/`. Inputs marked `automatic=0` in
`data/SOURCES.tsv` require manual retrieval; see `data/THIRD_PARTY.md`. The main
run requires no downloads.

## Evidentiary limits

- **Nothing in the mechanistic model is fitted.** Parameters are relative effect
  sizes, not physiological rate constants. Time is in nominal model days and
  exposure is dimensionless, so the 45-day horizon and the nominal day-30
  delayed intervention are simulation choices, not an experimental protocol.
- **The ensemble is a stress test, not a posterior.** The 5th–95th bands are
  heuristic sensitivity ranges from independent log-normal draws — not
  confidence intervals, credible intervals, or a calibration-conditioned
  posterior. The repair-structure scenarios are unweighted and carry no
  probability.
- **The mutation index is relative.** It is a cumulative fixation index for a
  representative modelled cell or lineage, not calibrated mutations per genome, a
  live-population average, or a clonal-selection model.
- **Repair capacity is not measured.** `Q` is a DDB2/NER translation proxy, not
  total genome-maintenance capacity, and aggregate lesion removal is a separate
  quantity. No functional aggregate-repair measurement enters the deposit, and
  DDB2 methylation background percentiles are neither repair-capacity
  measurements nor equivalence tests.
- **Population quantities are model diagnostics.** Absolute population factors,
  instantaneous division and death rates and integrated hazards are not measured
  apoptosis or death counts, and expansion suppression is distinguished from net
  cell loss.
- **Passing the checks is not validation.** A successful default run does not
  mean quantitative biological validation; `--strict-evidence` returns exit
  status 2 while the scientific evidence audit remains unresolved.

## Archive identity and citation

This directory is the complete EMD1 reproducibility archive. For peer review,
provide the whole directory as the submission's code/data archive or through a
reviewer-accessible repository. When a DOI-minting repository record is created,
cite that DOI in the manuscript without changing the archived version.

Software: MIT (`LICENSE`). Derived tables and outputs are released as new derived
results under CC BY 4.0; third-party inputs retain their original terms, recorded
in `data/THIRD_PARTY.md`. Cite the accompanying manuscript, which contains the
methods, results and interpretation.
