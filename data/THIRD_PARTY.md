# Input provenance

No third-party raw data or publisher figure images are bundled. `SOURCES.tsv`
records source URLs, local paths, reference SHA-256 checksums, TAR members and
automatic-download status. Original sources retain their own terms; the data
licence in this deposit does not relicense them.

| Source | Use |
|---|---|
| GEO GSE145923; PMID 33846348 | Arsenic/NEDD4L and repair-transcript coverage tracks |
| GEO GSE145924; PMID 34452996 | METTL14-knockdown/UVB DDB2 and XPC coverage tracks |
| UCSC hg19 refGene | Strand-aware exon coordinates for coverage analysis |
| NCBI Human.GRCh38.p13 annotation | Transcript identifiers and repair-gene selection |
| PMID 38142659 | Antioxidant actinomycin-chase observations |
| PMID 34998823 (PMC8814665) | APOBEC3B figure-panel endpoints |
| PMID 33846348 source workbook | NEDD4L observations |

## Bundled derived inputs

- `chase_decay_constants.csv`: 162 decay series; `k` is per hour and
  `half_life` is in hours. Gene, stage, intervention and replicate identify
  each series. These derived estimates support the offline audit, not a
  refit of the original chase observations.
- `digitised_endpoints.csv`: 37 panel-derived values with source, panel,
  extraction method and units. JBC 1G/3Q mutation values are the tallest
  substitution-category bar, not total mutation burden.
- `repair_edge_first_three_exons.csv`: eight DDB2/XPC rows with coordinates,
  enrichment, replicate quality checks and a cross-transcript null. The null
  is not a biological-replicate confidence interval.
- `repair_gene_symbols.csv`: 320 unique symbols selected by case-insensitive
  `DNA repair` matching in the annotation's `GOProcess` column. This is a
  fallback for background-gene selection when the full annotation is absent.

## Optional source reanalysis

Run `python data/fetch_data.py` from the archive root. It downloads entries
marked `automatic=1` and verifies each checksum. GEO tracks are extracted
from their series RAW TAR archives. A checksum mismatch fails rather than
silently accepting a changed source. `--verify` checks files without downloading.

The full NCBI annotation and two other sources are marked `automatic=0`.
Retrieve them manually from their manifest URLs and place them at the listed
paths under `code/emd1_simulation/data/`. NCBI may require an interactive
browser. Absence of a manual input is reported as optional by the downloader;
an analysis that needs it still requires that file.

From `code/`, `python -m emd1_simulation.fit.peaks` and
`python -m emd1_simulation.fit.repair_edge` use the downloaded coverage tracks
and UCSC coordinates. Repair-background selection can use the bundled gene
list. `python -m emd1_simulation.fit.zhao` requires the original chase workbook
and full NCBI annotation. `python -m emd1_simulation.fit.bayes` requires both
source workbooks and the optional dependencies in `requirements-bayes.txt`.
