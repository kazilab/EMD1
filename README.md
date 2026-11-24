# EMD1 RNA methylation and reader effects

This simulation examines how arsenic exposure and RNA methylation affect transcript abundance through reader-dependent decay and stabilization. The default notebook reproduces Figures 1, 2, S3 and S4 offline.

## Run the notebook

From the EMD1 directory, use Python 3.13.9 for the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c requirements-tested.txt
python -m jupyter lab EMD1_reproduce_figure.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. In Jupyter, select the environment’s Python kernel and choose **Restart Kernel and Run All Cells**. For Python 3.11 or newer with unpinned package versions, install `requirements.txt` without the constraints file.

The defaults evaluate eight conditions over 45 nominal model days. The sensitivity analysis and the repair counterfactual each use 400 parameter draws with fixed seeds. To explore other settings, change the run name and parameters in the notebook. [Open the notebook](EMD1_reproduce_figure.ipynb).

## Results

The notebook writes figures, captions, a numerical summary and a run log to `outputs/reproduced/`. The `tables/` subdirectory contains trajectories, endpoint contrasts, model checks and source comparisons. See `table_catalog.csv` for the contents and `observable_dictionary.csv` for units.

| Figure | Quantity | Output filename stem |
| --- | --- | --- |
| 1 | Arsenic exposure and antioxidant adaptation | `emd1_figure1_exposure_adaptation` |
| 2 | Aggregate methylation and reader-specific transcript responses | `emd1_figure2_reader_specificity` |
| S3 | Conditional fixation and relative population expansion | `emd1_figureS3_downstream_outputs` |
| S4 | Conditional engagement of the DNA-repair pathway | `emd1_figureS4_conditional_repair` |

Exposure and time are nominal. The shaded sensitivity ranges describe variation across assigned parameter values. Published measurements provide descriptive comparisons. Units, assay limitations, source attribution and instructions for rebuilding the database are in the [data notes](data/THIRD_PARTY.md).

## Command line and tests

For command-line use alone, install `src/emd1_simulation/requirements.txt` from the EMD1 directory. Run the analysis as follows. Use `--help` for the available settings.

```bash
cd src
python -m emd1_simulation --outdir ../outputs/reproduced
```

The notebook and command line both use `run_workflow()` in [workflow.py](src/emd1_simulation/workflow.py). The [model equations](src/emd1_simulation/simulation/model.py) and [intervention settings](src/emd1_simulation/simulation/conditions.py) define the simulated responses.

From EMD1, run `python verify_release.py` for the offline scientific checks. To run the full suite, install `requirements-dev.txt`, change to `src` and run `python -m pytest emd1_simulation/tests -q`. Optional original-source processing and Bayesian fitting use `src/emd1_simulation/requirements-source.txt` and `requirements-bayes.txt` in the same directory.

Software is distributed under the [MIT license](LICENSE).
