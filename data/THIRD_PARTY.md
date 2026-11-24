# EMD1 source data

The default analysis reads 177 source measurements and estimates from `emd1.sqlite`. Each row links to a citation in the `references` table and an extraction record in `provenance`. Consult the source studies for the original workbooks, coverage tracks and figures.

| Table | Rows | Measurements and interpretation |
|---|---:|---|
| `chase_decay_constants` | 162 | Zhao et al., Fig. 5B. For each gene, stage, siRNA and replicate, the negative slope of log remaining fraction at 0, 2, 4 and 6 hours gives the decay constant `k` (h⁻¹). Half-life is ln(2)/`k` in hours. Exposure stage is in experimental weeks. |
| `digitised_endpoints` | 7 | Gao et al., Figs. 1G and 3Q report the largest substitution-category count among cloned TP53 3D-PCR products. The selected counts lack a denominator for estimating mutation burden or rate. Fig. 3K gives approximate APOBEC3B mRNA abundance for FB23-2 relative to DMSO (0.12). Arsenic exposure is unspecified in that panel. |
| `repair_edge_summary` | 8 | DDB2 and XPC enrichment in chronic arsenic, arsenic-transformed, METTL14-knockdown and UVB conditions relative to their respective controls. Each effect divides the pooled IP/input contrast by its background median. The 5th–95th percentiles describe variation across background transcripts. |

The ODE uses nominal model days. Direct quantitative comparison with an assay requires a mapping of its time, intervention and readout to the model. In particular, the GEO repair experiment perturbs METTL14 and the model perturbs METTL3. The assay-specific comparisons are recorded in [observation_contracts.json](observation_contracts.json).

## Rebuild the database

From the EMD1 directory, run the following command to rebuild the bundled tables in a new SQLite file.

```bash
python data/build_database.py --output emd1-rebuilt.sqlite
```

Omitting `--output` replaces `data/emd1.sqlite`. The builder verifies file checksums, row counts, column types, database integrity and foreign keys. The manifest describes the CSV exports and schema (SQLite user version 1). CSV nulls are encoded as `\N`. Rebuilding the database uses the exported tables. Refitting decay estimates or repeating track extraction requires the original observations. [Snapshot manifest](curated_snapshot/manifest.json)

## Optional source reanalysis

From the EMD1 directory, run `python data/fetch_data.py` to download and checksum the automatic inputs into `data/raw/`. Use `python data/fetch_data.py --verify` to check local inputs. The manifest lists URLs, SHA-256 checksums, storage paths and any archive members. Obtain entries marked `automatic=0` directly from their sources. The gene-level count matrices used by `processing.datasets` require a separate download from the respective GEO series. [Source manifest](SOURCES.tsv)

- **Decay estimates.** Zhao et al. provide the chase observations in the workbook sheet `Fig. 5 and S5`. With the workbook present, `decay_rates(parse_fig5b())` in `emd1_simulation.processing.chase` refits the estimates.
- **Repair features.** From `src/`, run `python -m emd1_simulation.processing.repair_features` with the GSE145923 and GSE145924 tracks and hg19 refGene annotation. The analysis pools two IP and two input replicates per condition over the first three exons of the longest spliced RefSeq NM_ isoform. Coordinates are hg19, zero-based and half-open, in transcriptional order. GRCh38 gene annotations supply the repair-gene labels. The output is `outputs/source_reanalysis/repair_edge_first_three_exons.csv`.
- **NEDD4L exon 1.** Cui et al. give the hg38 interval `NC_000018.10:58221610-58221760` for `NM_001144966.3`. Its location in the hg19 coverage tracks is unresolved, so there is no default enrichment estimate. To re-extract it, copy the feature template, set the coordinate interpretation to `printed_interval_is_0_based_half_open` or `printed_interval_is_1_based_closed`, cite the evidence for that choice and supply the expected target interval, hg38-to-hg19 chain and hg19 RefSeq annotation with their source URLs and SHA-256 hashes. The extractor verifies unique mapping of the entire interval within one chain block and containment in the cited first exon. From `src/`, run `python -m emd1_simulation.processing.peaks --verified-nedd4l-feature ../data/nedd4l_feature_verified.json` with the completed record and source tracks. [Feature template](nedd4l_feature_template.json) · [Source status](source_status.json)

## Sources and rights

- Gao et al. (2022), *Journal of Biological Chemistry*, 298, 101563. APOBEC3B abundance and TP53 substitution categories. [Article](https://doi.org/10.1016/j.jbc.2022.101563)
- Cui et al. (2021), *Nature Communications*, 12, 2183. Arsenic and NEDD4L data, GSE145923. [Article](https://doi.org/10.1038/s41467-021-22469-6)
- Yang et al. (2021), *Proceedings of the National Academy of Sciences*, 118, e2025948118. METTL14, DDB2 and UVB data, GSE145924. [Article](https://doi.org/10.1073/pnas.2025948118)
- Zhao et al. (2024), *Journal of Hazardous Materials*, 465, 133329. Antioxidant transcript chases. [Article](https://doi.org/10.1016/j.jhazmat.2023.133329)
- Exon coordinates come from UCSC hg19 refGene. Repair-gene labels come from a case-insensitive `DNA repair` match in the NCBI annotation field `GOProcess`. [UCSC hg19 refGene](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz) · [NCBI Human.GRCh38.p13 annotation](https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz)

Original inputs retain their source terms. Derived tables and outputs are released under CC BY 4.0 with attribution to the accompanying manuscript and source studies. The software is licensed under MIT. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [MIT license](../LICENSE)
