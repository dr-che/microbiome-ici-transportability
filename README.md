# Microbiome–ICI transportability reproducibility release (v1.0.0)

This repository is the public, privacy-safe reproducibility release for the melanoma immune-checkpoint-inhibitor (ICI) microbiome transportability analyses.

## Release scope

The release contains:
- privacy-safe derived abundance matrices and harmonized metadata;
- frozen analysis outputs used for manuscript-level verification;
- figures;
- analysis and validation scripts;
- environment specifications;
- the Stage3 portable reproduction adapter.

Raw source datasets are not redistributed here. Source accessions and provenance are listed in `metadata/raw_data_accessions.tsv`.

## Reproduction

The portable Stage3 adapter is:

```text
scripts/portable/run_stage3_portable.py
```

A full clean Stage3 reproduction can be run from the repository root with:

```bash
python scripts/portable/run_stage3_portable.py \
  --work-dir ../stage3_all_work \
  --mode all \
  --clean
```

Detailed portability, privacy-safe identifier, intermediate-file, and numerical-verification contracts are documented in:

```text
scripts/portable/README_stage3_portability.md
```

## Verification boundary

The public runner verifies frozen public inputs and executed-source hashes before analysis. Reproduction checks are SHA256-first. Where byte-identical output is not portable across numerical environments, only narrowly audited finite floating-point differences are accepted under a fail-closed absolute tolerance of `1e-12`; structural, text, integer/discrete, file-set, non-finite, or above-tolerance differences fail the gate.

The original executed-source provenance scripts under `scripts/stage3/` are retained byte-for-byte. Their historical execution paths are not used directly by the portable adapter.

## Privacy

Study-local source subject identifiers are not required for public reproduction. The release uses privacy-safe public identifiers and an order-preserving synthetic key where the frozen bootstrap implementation requires stable lexical subject ordering.

## Repository contents

- `data_derived/` — privacy-safe derived matrices
- `metadata/` — harmonization, provenance, accession, and release metadata
- `results_frozen/` — frozen verification outputs
- `figures/` — manuscript/review figures
- `scripts/` — analysis, validation, and portable-reproduction code
- `environment/` — environment specifications and session information
- `docs/` — prespecified analysis and leakage-control documentation

## Release status

This v1.0.0 repository state has passed the internal public-release portability and clean-reproduction gates. Publication/hosting metadata such as the final GitHub/GitLab release URL and Zenodo DOI should be added only after those records are created.

## Data and reuse note

Third-party source datasets remain subject to the access and reuse terms of their original repositories and publications. This release does not redistribute raw source data.
<!-- STEP10_19H_R2_PRESENTATION_RESYNC -->
## Manuscript-facing presentation resynchronization

This repository snapshot was resynchronized to the final manuscript-facing evidence presentation without refitting or retuning any scientific model.

- Article: *Validation hierarchy and directed transfer reveal limited gut microbiome signature transportability across melanoma immunotherapy cohorts*
- Main Figures 1–6 use the publication-facing Stage4-final PNG/PDF assets.
- Table S17: `results_frozen/analysis6/Table_S17_metadata_completeness_public_safe.tsv`
- Figure S17: `figures/supplementary/Figure_S17_DEBIASM_secondary_benchmark.png`
- Table S18: `results_frozen/analysis7/Table_S18_DEBIASM_presentation.tsv`
- STORMS crosswalk: `docs/Step10_18C2_STORMS_reporting_crosswalk.docx`
- Validation-hierarchy/effective-unit reporting summary: `results_frozen/validation/Step10_18C2_validation_hierarchy_effective_unit_summary.tsv`
- This resynchronization changes presentation/reporting assets only; protected scientific payload remains unchanged.
<!-- STEP10_19H_RELEASE_METADATA -->
## v1.0.0 release metadata

- Article: *Validation hierarchy and directed transfer reveal limited gut microbiome signature transportability across melanoma immunotherapy cohorts*
- Repository contact / handling correspondence: Ping Che (dr.che@cqu.edu.cn)
- Creator order: Zengzhi Li; Yangming Liu; Po Liang; Zhangjun Wei; Kai Deng; Zhuoren Cheng; Ping Che.
- Analysis code: MIT License (LICENSE).
- Project-created derived tables, figures, and documentation: CC BY 4.0 (LICENSE_DERIVED_MATERIALS.md).
- Raw public sequencing data are not redistributed or relicensed; original provider terms continue to apply.
- Citation metadata: CITATION.cff and .zenodo.json.
- Zenodo DOI: not assigned yet. Do not cite a DOI until the GitHub v1.0.0 release has been archived and the real Zenodo record resolves.
