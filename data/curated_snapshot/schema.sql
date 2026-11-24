CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE "references" (
                    reference_id TEXT PRIMARY KEY, citation TEXT NOT NULL, title TEXT NOT NULL,
                    doi TEXT, pmid TEXT, url TEXT NOT NULL, notes TEXT NOT NULL);

CREATE TABLE provenance (
                    artifact_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, sha256 TEXT NOT NULL,
                    url TEXT, archive_member TEXT, automatic INTEGER,
                    reference_id TEXT REFERENCES "references"(reference_id),
                    role TEXT NOT NULL, transformation_notes TEXT NOT NULL);

CREATE TABLE "chase_decay_constants" ("gene" TEXT, "stage" TEXT, "sirna" TEXT, "rep" INTEGER, "k" REAL, "half_life" REAL, reference_id TEXT NOT NULL REFERENCES "references"(reference_id), provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id), UNIQUE("gene", "stage", "sirna", "rep"));

CREATE TABLE "digitised_endpoints" ("arm" TEXT, "paper" TEXT, "panel" TEXT, "quantity" TEXT, "condition" TEXT, "value" REAL, "unit" TEXT, "read_method" TEXT, "p_value" REAL, reference_id TEXT NOT NULL REFERENCES "references"(reference_id), provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id), UNIQUE("panel", "condition"));

CREATE TABLE "repair_edge_summary" ("series" TEXT, "contrast" TEXT, "gene" TEXT, "pooled_raw_ratio" REAL, "background_raw_median" REAL, "effect" REAL, "effect_definition" TEXT, "pct" REAL, "null_lo" REAL, "null_hi" REAL, "module" REAL, "n_background_drawn" INTEGER, "n_background_used" INTEGER, "n_repair" INTEGER, "accession" TEXT, "chrom" TEXT, "strand" TEXT, "region_bounding_start_0based" INTEGER, "region_bounding_end_0based" INTEGER, "exon_intervals_0based" TEXT, "region_exon_nt" INTEGER, "input_qc_bin_nt" INTEGER, "min_control_input_bins" INTEGER, "min_control_input_density" REAL, "control_input_passing_bins_min" INTEGER, "control_input_mean_density_min" REAL, "index_raw_ratio_rep1" REAL, "index_background_normalized_ratio_rep1" REAL, "index_ratio_rep1" REAL, "index_raw_ratio_rep2" REAL, "index_background_normalized_ratio_rep2" REAL, "index_ratio_rep2" REAL, reference_id TEXT NOT NULL REFERENCES "references"(reference_id), provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id), UNIQUE("series", "contrast", "gene"));
