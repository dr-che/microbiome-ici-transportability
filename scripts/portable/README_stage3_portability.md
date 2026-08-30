# Stage3 portable reproduction adapter

The files under `scripts/stage3/analysis3/` and `scripts/stage3/analysis4/` are
the real executed-source provenance files. They are intentionally retained
byte-for-byte and therefore still contain historical execution paths.

Do **not** edit those provenance scripts in place.

`run_stage3_portable.py` creates a temporary compatibility workspace and
modifies only path assignments in temporary copies. It preserves the frozen
analysis logic, seeds, feature caps, bootstrap counts and QAP counts.

## Why an order-preserving synthetic subject key is included

The frozen Analysis4 code sorts held-out patients by its `patient_id` field
before target-patient bootstrap resampling. Publishing the original study-local
subject identifiers is unnecessary. The release therefore provides
`metadata/stage3_order_preserving_subject_keys.tsv`, which contains only
release `analysis_id` values and synthetic `stage3_sort_id` values. The
synthetic keys preserve the original lexical ordering required by the frozen
bootstrap implementation without exposing the original identifiers.

## Execution modes

Prepare and inspect the compatibility workspace without running a scientific
analysis:

```bash
python scripts/portable/run_stage3_portable.py \
  --work-dir ../stage3_portable_work \
  --mode prepare
```

Reproduce Analysis3:

```bash
python scripts/portable/run_stage3_portable.py \
  --work-dir ../stage3_analysis3_work \
  --mode analysis3
```

Reproduce Analysis4. This first regenerates the private-at-source Step10.5B
patient-level prediction table inside the temporary work directory using only
the privacy-safe public matrix and metadata. That regenerated patient-level
table is a transient reproducibility intermediate and is not part of the
public frozen results:

```bash
python scripts/portable/run_stage3_portable.py \
  --work-dir ../stage3_analysis4_work \
  --mode analysis4
```

Run both:

```bash
python scripts/portable/run_stage3_portable.py \
  --work-dir ../stage3_all_work \
  --mode all
```

Use `--clean` only when you intentionally want the adapter to remove and
recreate the specified work directory.

## Verification boundary

The adapter checks frozen public-input and executed-source hashes before
execution. The verification layer is fail-closed and separates exact
provenance contracts from platform-tolerant numeric reproduction.

For Step10.5B, the adapter regenerates all four transient outputs from the
privacy-safe public matrix and metadata. Before Analysis4 is allowed to run,
the PRIMARY / elastic-net directed-transfer metrics and QAP rows are compared
with the public frozen references. Headers, row counts, text fields and
integer-like values must match exactly; finite non-integer numeric values may
differ only within an absolute tolerance of `1e-12`. The
`primary_elastic_net_MRQAP.tsv` file must remain byte-identical by SHA256.
The regenerated patient-level prediction table must satisfy its fixed schema,
row-count, label and probability-range contracts. All regenerated Step10.5B
outputs are then propagated into Analysis4; frozen Step10.5B files are
verification references, not Analysis4 computational inputs.

Analysis3 has eight frozen release TSVs. The cap-specific
`Analysis3_downsample_*_cap{10,25,50,500}.tsv` files are required transient
intermediates: the adapter verifies that exactly those eight intermediate files
are present and that they concatenate exactly, in cap order, into the two
merged frozen-output tables. They are not counted as frozen release outputs.
Each of the eight frozen Analysis3 TSVs is checked by SHA256 first. If SHA256
differs, semantic fallback is permitted only for the five audited tables and
only for their Brier-derived floating-point columns (`brier`, `mean_brier`,
`mean_brier_mean`, or `mean_brier_sd`, as applicable). All other columns must
remain exact; permitted finite non-integer numeric differences are bounded by
absolute tolerance `1e-12`.

Analysis4 TSV outputs are checked by SHA256 first. If a file is not
byte-identical, the adapter applies the fail-closed semantic comparison:
the frozen file set, headers, row counts, text and integer-like values must
remain exact, while finite non-integer numeric values may differ only within
absolute tolerance `1e-12`. Any missing frozen output, unexpected Analysis3
intermediate, structural difference, discrete difference, non-finite mismatch,
or numeric difference above the tolerance fails the release gate.

This boundary addresses platform-level floating-point and linear-algebra
rounding only. It does not permit model retuning, altered seeds, changed
scientific parameters, or changed scientific interpretation. The original
executed-source files remain immutable provenance; the adapter remains a
release-engineering portability layer.
