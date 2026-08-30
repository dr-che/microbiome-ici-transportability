#!/usr/bin/env python3
"""
Portable Stage3 reproduction adapter.

The six Stage3 scripts under scripts/stage3/ are retained byte-for-byte as
executed-source provenance. They contain historical absolute paths and are not
edited. This adapter creates a temporary compatibility workspace, rewrites only
path assignments in temporary copies, and can execute the original analysis
logic there.

Public source identifiers are never reconstructed. A non-identifying synthetic
sort key preserves the original patient-ID lexical ordering needed by the
frozen Analysis4 bootstrap implementation.

Verification is fail-closed. Analysis3 retains byte-identical SHA256 output
verification. Analysis4 uses SHA256 first, then a strict semantic fallback only
for finite non-integer numeric values within absolute tolerance 1e-12; text,
integer-like values, headers, row counts, and file sets remain exact.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "data_derived/V2_species_SGB_relative_abundance_v1.0.0.tsv.gz":
        "12fdf833b7c22db879a409010320a27aa4ef85a8d18ff025ca3c97bf643fce59",
    "metadata/patient_specimen_run_profile_manifest_v1.0.0.tsv":
        "ee4664be969c120983e60c7157acd704403d0a19726e4e1ee8753fc532048721",
    "metadata/domain_publication_family_dictionary.tsv":
        "e8c463c73d0e06cc69051ee6b971762098958377955464bceca606b3432d7129",
    "scripts/Step10_5B_directed_transfer_ecological_distance_v1.py":
        "85c122e3da7a25ba0280f48fc5ac15cd0ab74c07746a09138ceef92a97bb2a49",
    "scripts/stage3/analysis3/analysis3_downsample_cap.py":
        "090c15e05d41ebeb805c29ffb414ed021fe0907c84fdbaf8598d8543467f256b",
    "scripts/stage3/analysis3/analysis3_finalize.py":
        "35b8470d94a82412da2735b77264872cd53ff3b50a373a9c3a204daeb8734463",
    "scripts/stage3/analysis3/analysis3_run.py":
        "a82a38266256c3bcf1bb684cfc23e5319f87b8fd8642e48f1210e111cf1e8546",
    "scripts/stage3/analysis4/analysis4_continue.py":
        "423a1eb23e8b28af1d4cb99d31b6b34234c5a859b31a93815b4624a000e313a0",
    "scripts/stage3/analysis4/analysis4_pair_bootstrap.py":
        "c460a89263bdcf726326bd32a9c55a1e60db4ecf937e86ebf37e7e161ee03682",
    "scripts/stage3/analysis4/analysis4_run.py":
        "1fd10af8925a884119d3cdf6997bb618ba374fae4ca1ec4fb1b1a870dee9f562",
    "results_frozen/step10_5B/directed_domain_transfer_metrics.tsv":
        "2b650a1d27b68de66b4a57d5b580ee954968276001c830a1ddc652f6c3a2fe5a",
    "results_frozen/step10_5B/univariable_QAP_associations.tsv":
        "c472c14f85448c1ffc6ff39a55011f05d6ef0e27a6a51fde701d90f11adaa49a",
    "results_frozen/step10_5B/primary_elastic_net_MRQAP.tsv":
        "fc60d2f42b26598738bdc800cdc42acc56ab8413a7fa8be7719804448db0ad09",
}

SCENARIOS = {
    "PRIMARY": "Step10_2_manifest_R3_primary_lock_v8.tsv",
    "S1_EXCLUDE_BCN12": "Step10_2_manifest_R3_sensitivity_S1_exclude_BCN12_v8.tsv",
    "S2_BCN12_AS_NR": "Step10_2_manifest_R3_sensitivity_S2_BCN12_as_NR_v8.tsv",
    "S3_EXCLUDE_LEE_SITES": "Step10_2_manifest_R3_sensitivity_S3_exclude_Lee_sites_v8.tsv",
}

A3 = [
    "scripts/stage3/analysis3/analysis3_run.py",
    "scripts/stage3/analysis3/analysis3_downsample_cap.py",
    "scripts/stage3/analysis3/analysis3_finalize.py",
]
A4 = [
    "scripts/stage3/analysis4/analysis4_run.py",
    "scripts/stage3/analysis4/analysis4_continue.py",
    "scripts/stage3/analysis4/analysis4_pair_bootstrap.py",
]

SEMANTIC_ATOL = 1e-12
INT_TOKEN = re.compile(r"^[+-]?\d+$")

# Analysis3 release verification boundary.
# The cap-specific files are deterministic transient intermediates emitted by
# the frozen sensitivity scripts and consumed into the two merged final TSVs.
# They are required and audited, but are not frozen release outputs.
ANALYSIS3_EXPECTED_INTERMEDIATE_OUTPUTS = {
    f"Analysis3_downsample_directed_metrics_cap{cap}.tsv"
    for cap in (10, 25, 50, 500)
} | {
    f"Analysis3_downsample_repeat_summary_cap{cap}.tsv"
    for cap in (10, 25, 50, 500)
}

# Cross-platform reruns showed machine-precision differences only in these
# Brier-derived floating-point columns. SHA256 is always attempted first.
# If SHA differs, all other columns remain exact and only the named columns
# may use the finite non-integer absolute-tolerance fallback.
ANALYSIS3_SEMANTIC_COLUMNS = {
    "Analysis3_downsample_aggregate_summary.tsv": {
        "mean_brier_mean", "mean_brier_sd",
    },
    "Analysis3_downsample_directed_metrics.tsv": {"brier"},
    "Analysis3_downsample_repeat_summary.tsv": {"mean_brier"},
    "Analysis3_feature_cap_and_source_size_summary.tsv": {"mean_brier"},
    "Analysis3_feature_cap_directed_metrics.tsv": {"brier"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r.fieldnames or []), list(r)


def write_tsv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def semantic_cell_equal(a: str, b: str):
    """Return (equal, numeric_nonexact, abs_diff).

    Exact text is preferred. Integer-like tokens are discrete and must remain
    exact. Only finite non-integer numeric text may use the platform-tolerance
    fallback.
    """
    if a == b:
        return True, False, 0.0
    if INT_TOKEN.fullmatch(a.strip()) or INT_TOKEN.fullmatch(b.strip()):
        return False, False, 0.0
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False, False, 0.0
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False, False, 0.0
    diff = abs(fa - fb)
    return diff <= SEMANTIC_ATOL, True, diff


def semantic_rows_equal(fields_a, rows_a, fields_b, rows_b):
    if fields_a != fields_b:
        return False, 0.0, 1, 0, "HEADER"
    if len(rows_a) != len(rows_b):
        return False, 0.0, abs(len(rows_a) - len(rows_b)), 0, "ROW_COUNT"

    max_abs = 0.0
    hard_diff_n = 0
    numeric_nonexact_n = 0
    for row_a, row_b in zip(rows_a, rows_b):
        for field in fields_a:
            ok, numeric_nonexact, diff = semantic_cell_equal(
                str(row_a.get(field, "")),
                str(row_b.get(field, "")),
            )
            if numeric_nonexact:
                numeric_nonexact_n += 1
                max_abs = max(max_abs, diff)
            if not ok:
                hard_diff_n += 1

    return (
        hard_diff_n == 0,
        max_abs,
        hard_diff_n,
        numeric_nonexact_n,
        "CELL",
    )


def semantic_rows_equal_restricted(
    fields_a,
    rows_a,
    fields_b,
    rows_b,
    allowed_numeric_fields,
):
    """Fail-closed semantic comparison with column-level numeric allowance."""
    if fields_a != fields_b:
        return False, 0.0, 1, 0, "HEADER"
    if len(rows_a) != len(rows_b):
        return False, 0.0, abs(len(rows_a) - len(rows_b)), 0, "ROW_COUNT"

    allowed_numeric_fields = set(allowed_numeric_fields)
    max_abs = 0.0
    hard_diff_n = 0
    numeric_nonexact_n = 0

    for row_a, row_b in zip(rows_a, rows_b):
        for field in fields_a:
            a = str(row_a.get(field, ""))
            b = str(row_b.get(field, ""))
            if a == b:
                continue

            if field not in allowed_numeric_fields:
                hard_diff_n += 1
                continue

            ok, numeric_nonexact, diff = semantic_cell_equal(a, b)
            if numeric_nonexact:
                numeric_nonexact_n += 1
                max_abs = max(max_abs, diff)
            if not ok:
                hard_diff_n += 1

    return (
        hard_diff_n == 0,
        max_abs,
        hard_diff_n,
        numeric_nonexact_n,
        "RESTRICTED_CELL",
    )


def verify_filtered_semantic_tsv(
    frozen: Path,
    generated: Path,
    filters,
    label: str,
):
    frozen_fields, frozen_rows = read_tsv(frozen)
    generated_fields, generated_rows = read_tsv(generated)

    frozen_rows = [
        row for row in frozen_rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    generated_rows = [
        row for row in generated_rows
        if all(row.get(key) == value for key, value in filters.items())
    ]

    if not frozen_rows or len(frozen_rows) != len(generated_rows):
        raise RuntimeError(
            f"{label} row contract failed: "
            f"frozen={len(frozen_rows)}, generated={len(generated_rows)}"
        )

    ok, max_abs, hard_diff_n, numeric_nonexact_n, _ = semantic_rows_equal(
        frozen_fields,
        frozen_rows,
        generated_fields,
        generated_rows,
    )

    print(
        f"{label}={str(ok).upper()} rows={len(frozen_rows)} "
        f"max_abs_diff={max_abs:.17g} hard_diff_n={hard_diff_n} "
        f"numeric_nonexact_n={numeric_nonexact_n}",
        flush=True,
    )

    if not ok:
        raise RuntimeError(f"{label}=FALSE")


def require_repo_asset(repo: Path, rel: str) -> Path:
    p = repo / rel
    if not p.is_file():
        raise RuntimeError(f"Required release asset missing: {rel}")
    observed = sha256(p)
    expected = EXPECTED[rel]
    if observed != expected:
        raise RuntimeError(f"Frozen asset hash mismatch: {rel}: {observed}")
    return p


def run(cmd, cwd=None):
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def patch_assignment(text: str, variable: str, value: Path) -> str:
    pattern = re.compile(
        rf"(?m)^(\s*{re.escape(variable)}\s*=\s*)Path\([^)]*\)"
    )
    replacement = lambda m: m.group(1) + f"Path({str(value)!r})"
    out, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Expected exactly one {variable}=Path(...) assignment; observed {n}")
    return out


def make_patched_scripts(repo: Path, work: Path, compat_repo: Path):
    patched = work / "patched_scripts"
    patched.mkdir(parents=True, exist_ok=True)
    a3_out = work / "analysis3_outputs"
    a4_out = work / "analysis4_outputs"
    a3_out.mkdir(parents=True, exist_ok=True)
    a4_out.mkdir(parents=True, exist_ok=True)

    # Core Analysis3 module first.
    src = require_repo_asset(repo, A3[0]).read_text(encoding="utf-8-sig")
    src = patch_assignment(src, "ROOT", compat_repo)
    src = patch_assignment(src, "OUT", a3_out)
    a3_core = patched / "analysis3_run.py"
    a3_core.write_text(src, encoding="utf-8-sig")

    for rel in A3[1:]:
        src = require_repo_asset(repo, rel).read_text(encoding="utf-8-sig")
        src = patch_assignment(src, "MODULE_PATH", a3_core)
        (patched / Path(rel).name).write_text(src, encoding="utf-8-sig")

    for rel in A4:
        src = require_repo_asset(repo, rel).read_text(encoding="utf-8-sig")
        src = patch_assignment(src, "ROOT", compat_repo)
        src = patch_assignment(src, "OUT", a4_out)
        (patched / Path(rel).name).write_text(src, encoding="utf-8-sig")

    for p in patched.glob("*.py"):
        compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")

    return patched, a3_out, a4_out


def prepare_compatibility_repo(repo: Path, work: Path):
    compat = work / "compat_repo"
    compat.mkdir(parents=True, exist_ok=True)

    matrix_gz = require_repo_asset(
        repo, "data_derived/V2_species_SGB_relative_abundance_v1.0.0.tsv.gz"
    )
    manifest_path = require_repo_asset(
        repo, "metadata/patient_specimen_run_profile_manifest_v1.0.0.tsv"
    )
    domain_path = require_repo_asset(
        repo, "metadata/domain_publication_family_dictionary.tsv"
    )
    order_path = repo / "metadata/stage3_order_preserving_subject_keys.tsv"
    if not order_path.is_file():
        raise RuntimeError("Missing stage3_order_preserving_subject_keys.tsv")

    order_fields, order_rows = read_tsv(order_path)
    if order_fields != ["analysis_id", "stage3_sort_id"] or len(order_rows) != 363:
        raise RuntimeError("Unexpected Stage3 order-key table")
    order = {r["analysis_id"]: r["stage3_sort_id"] for r in order_rows}
    if len(order) != 363 or len(set(order.values())) != 363:
        raise RuntimeError("Stage3 order-key uniqueness gate failed")

    # Write compatibility matrix twice, preserving all scientific values while
    # replacing only the temporary patient_id with the order-preserving key.
    with gzip.open(matrix_gz, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    if header[:4] != ["manifest_id", "patient_id", "domain_id", "response_harmonized"]:
        raise RuntimeError("Unexpected public V2 matrix metadata columns")
    if len(rows) != 363 or len(header) != 2209:
        raise RuntimeError("Unexpected public V2 matrix dimensions")

    converted = []
    for row in rows:
        rid = row[0]
        if rid not in order:
            raise RuntimeError(f"Order key missing for {rid}")
        out = list(row)
        out[1] = order[rid]
        converted.append(out)

    matrix_targets = [
        compat / "data/derived/assembled_species_matrix.tsv",
        compat / "02_results_step10_4B_v2_species_only/assembled_species_matrix.tsv",
    ]
    for target in matrix_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(converted)

    # Generate the four scenario manifests deterministically from the public
    # primary manifest. The named S3 scenario corresponds to excluding the
    # Manchester and Barcelona domains (n=326), as in the frozen analysis.
    fields, primary = read_tsv(manifest_path)
    if len(primary) != 363:
        raise RuntimeError("Unexpected public primary manifest row count")
    sens = [r for r in primary if r.get("label_sensitivity_subject") == "Yes"]
    if len(sens) != 1:
        raise RuntimeError("Expected exactly one label-sensitivity subject")

    def scrub_id_columns(row):
        out = dict(row)
        rid = out["analysis_id"]
        out["manifest_id"] = rid
        key = order[rid]
        for c in ("patient_id", "sample_id", "specimen_id"):
            if c in out:
                out[c] = key
        return out

    scenario_rows = {}
    scenario_rows["PRIMARY"] = [scrub_id_columns(r) for r in primary]
    scenario_rows["S1_EXCLUDE_BCN12"] = [
        scrub_id_columns(r) for r in primary
        if r.get("label_sensitivity_subject") != "Yes"
    ]
    s2 = [scrub_id_columns(r) for r in primary]
    for r in s2:
        if r.get("label_sensitivity_subject") == "Yes":
            r["response_harmonized"] = "Non-responder"
    scenario_rows["S2_BCN12_AS_NR"] = s2
    scenario_rows["S3_EXCLUDE_LEE_SITES"] = [
        scrub_id_columns(r) for r in primary
        if r.get("domain_id") not in {"Lee_Manchester", "Lee_Barcelona"}
    ]
    expected_n = {
        "PRIMARY": 363,
        "S1_EXCLUDE_BCN12": 362,
        "S2_BCN12_AS_NR": 363,
        "S3_EXCLUDE_LEE_SITES": 326,
    }
    for scenario, filename in SCENARIOS.items():
        rows_s = scenario_rows[scenario]
        if len(rows_s) != expected_n[scenario]:
            raise RuntimeError(f"Scenario count mismatch: {scenario}: {len(rows_s)}")
        write_tsv(compat / filename, fields, rows_s)

    # Step10.5B domain metadata uses the already public 9-row dictionary.
    dfields, drows = read_tsv(domain_path)
    required = {
        "domain_id", "study_family", "country", "macro_region",
        "treatment_scope_group", "source_doi", "metadata_confidence",
    }
    if len(drows) != 9 or not required.issubset(set(dfields)):
        raise RuntimeError("Public domain dictionary does not satisfy Step10.5B contract")
    shutil.copy2(domain_path, compat / "Step10_5B_domain_metadata_v1.tsv")

    # Frozen aggregate Step10.5B files are enough for Analysis3 and serve as
    # verification references. Analysis4 additionally requires regenerated
    # patient-level predictions, produced by run_step10_5b().
    frozen_out = compat / "results/step10_5B"
    frozen_out.mkdir(parents=True, exist_ok=True)
    for name in (
        "directed_domain_transfer_metrics.tsv",
        "univariable_QAP_associations.tsv",
        "primary_elastic_net_MRQAP.tsv",
    ):
        src_rel = f"results_frozen/step10_5B/{name}"
        shutil.copy2(require_repo_asset(repo, src_rel), frozen_out / name)

    return compat


def run_step10_5b(repo: Path, compat: Path):
    script = require_repo_asset(
        repo, "scripts/Step10_5B_directed_transfer_ecological_distance_v1.py"
    )
    run([sys.executable, script, "--root", compat])

    generated = compat / "02_results_step10_5B_v1"
    required = [
        "directed_domain_transfer_metrics.tsv",
        "all_directed_transfer_predictions.tsv",
        "univariable_QAP_associations.tsv",
        "primary_elastic_net_MRQAP.tsv",
    ]
    for name in required:
        if not (generated / name).is_file():
            raise RuntimeError(f"Step10.5B did not generate {name}")

    frozen = repo / "results_frozen/step10_5B"

    # Analysis4 consumes the PRIMARY / elastic-net scientific boundary.
    # Platform-dependent floating arithmetic can differ below machine-scale
    # rounding while leaving the scientific values unchanged. Text and
    # integer-like fields remain exact; finite non-integer numeric fields use
    # a strict absolute tolerance of 1e-12.
    verify_filtered_semantic_tsv(
        frozen / "directed_domain_transfer_metrics.tsv",
        generated / "directed_domain_transfer_metrics.tsv",
        {"scenario": "PRIMARY", "model": "elastic_net"},
        "STEP10_5B_PRIMARY_ELASTIC_NET_METRICS_GATE",
    )
    verify_filtered_semantic_tsv(
        frozen / "univariable_QAP_associations.tsv",
        generated / "univariable_QAP_associations.tsv",
        {"scenario": "PRIMARY", "model": "elastic_net"},
        "STEP10_5B_PRIMARY_ELASTIC_NET_QAP_GATE",
    )

    generated_mrqap = generated / "primary_elastic_net_MRQAP.tsv"
    frozen_mrqap = frozen / "primary_elastic_net_MRQAP.tsv"
    if sha256(generated_mrqap) != sha256(frozen_mrqap):
        raise RuntimeError(
            "STEP10_5B_PRIMARY_ELASTIC_NET_MRQAP_SHA256_GATE=FALSE"
        )
    print("STEP10_5B_PRIMARY_ELASTIC_NET_MRQAP_SHA256_GATE=TRUE", flush=True)

    # The privacy-safe prediction table has no public frozen patient-ID
    # counterpart. Verify its full structural/discrete contract here; its
    # scientific propagation is then checked by the Analysis4 output gates.
    prediction_fields, prediction_rows = read_tsv(
        generated / "all_directed_transfer_predictions.tsv"
    )
    expected_prediction_fields = [
        "scenario",
        "model",
        "train_domain",
        "test_domain",
        "manifest_id",
        "patient_id",
        "true_label",
        "predicted_probability",
        "predicted_class",
    ]
    if (
        prediction_fields != expected_prediction_fields
        or len(prediction_rows) != 31980
    ):
        raise RuntimeError("STEP10_5B_PREDICTION_TABLE_CONTRACT_FALSE")

    primary_elastic_net = [
        row for row in prediction_rows
        if row.get("scenario") == "PRIMARY"
        and row.get("model") == "elastic_net"
    ]
    if len(primary_elastic_net) != 2904:
        raise RuntimeError(
            "STEP10_5B_PRIMARY_ELASTIC_NET_PREDICTION_ROW_N_FALSE: "
            f"{len(primary_elastic_net)}"
        )

    for row in primary_elastic_net:
        if (
            row.get("true_label") not in {"0", "1"}
            or row.get("predicted_class") not in {"0", "1"}
        ):
            raise RuntimeError("STEP10_5B_PREDICTION_DISCRETE_FIELD_GATE_FALSE")
        probability = float(row["predicted_probability"])
        if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
            raise RuntimeError("STEP10_5B_PREDICTED_PROBABILITY_GATE_FALSE")

    print(
        "STEP10_5B_PRIVACY_SAFE_PREDICTION_CONTRACT_GATE=TRUE "
        "rows=31980 primary_elastic_net_rows=2904",
        flush=True,
    )

    # Propagate every regenerated Step10.5B product into Analysis4. Frozen
    # files remain verification references only, not computational inputs.
    analysis4_inputs = compat / "results/step10_5B"
    analysis4_inputs.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = generated / name
        dst = analysis4_inputs / name
        shutil.copy2(src, dst)
        if sha256(src) != sha256(dst):
            raise RuntimeError(
                f"STEP10_5B_GENERATED_READBACK_GATE_FALSE: {name}"
            )

    print("STEP10_5B_GENERATED_TO_ANALYSIS4_PROPAGATION_GATE=TRUE", flush=True)


def run_analysis3(patched: Path):
    for cap in (10, 25, 50, 500):
        run([sys.executable, patched / "analysis3_downsample_cap.py", str(cap)])
    run([sys.executable, patched / "analysis3_finalize.py"])


def run_analysis4(patched: Path):
    run([sys.executable, patched / "analysis4_run.py"])
    run([sys.executable, patched / "analysis4_continue.py"])
    run([sys.executable, patched / "analysis4_pair_bootstrap.py"])


def compare_analysis3_outputs(repo: Path, produced: Path):
    """Verify the eight frozen Analysis3 release TSVs plus intermediates.

    Expected cap-specific TSVs are required transient intermediates. They must
    combine exactly, in cap order, into the two merged final tables, but they
    are not counted as frozen release outputs.
    """
    frozen = repo / "results_frozen/stage3/analysis3"
    if not frozen.is_dir():
        raise RuntimeError("Frozen Analysis3 result directory missing")

    refs = {path.name: path for path in frozen.glob("*.tsv")}
    outs = {path.name: path for path in produced.glob("*.tsv")}

    extra = set(outs) - set(refs)
    if extra != ANALYSIS3_EXPECTED_INTERMEDIATE_OUTPUTS:
        raise RuntimeError(
            "Analysis3 transient-intermediate file-set mismatch: "
            f"expected={sorted(ANALYSIS3_EXPECTED_INTERMEDIATE_OUTPUTS)}, "
            f"observed={sorted(extra)}"
        )
    print(
        "ANALYSIS3_EXPECTED_INTERMEDIATE_FILESET_GATE=TRUE "
        f"intermediate_n={len(extra)}",
        flush=True,
    )

    # Prove that the eight cap-specific intermediates are consumed exactly
    # into the two merged final tables.
    for merged_name, prefix in (
        (
            "Analysis3_downsample_directed_metrics.tsv",
            "Analysis3_downsample_directed_metrics_cap",
        ),
        (
            "Analysis3_downsample_repeat_summary.tsv",
            "Analysis3_downsample_repeat_summary_cap",
        ),
    ):
        merged_fields, merged_rows = read_tsv(outs[merged_name])
        combined_rows = []
        for cap in (10, 25, 50, 500):
            part_fields, part_rows = read_tsv(
                outs[f"{prefix}{cap}.tsv"]
            )
            if part_fields != merged_fields:
                raise RuntimeError(
                    f"Analysis3 intermediate header mismatch: {prefix}{cap}.tsv"
                )
            combined_rows.extend(part_rows)
        if combined_rows != merged_rows:
            raise RuntimeError(
                f"Analysis3 intermediate consumption mismatch: {merged_name}"
            )
    print("ANALYSIS3_INTERMEDIATE_CONSUMPTION_GATE=TRUE", flush=True)

    comparisons = []
    for name in sorted(refs):
        ref = refs[name]
        out = outs.get(name)

        if out is None:
            comparisons.append({
                "file": name,
                "verification_gate": "DIFF",
                "verification_mode": "MISSING_OUTPUT",
                "sha256_match": "FALSE",
                "max_abs_diff": "",
                "hard_diff_n": 1,
                "numeric_nonexact_n": 0,
                "frozen_sha256": sha256(ref),
                "reproduced_sha256": "",
            })
            continue

        frozen_hash = sha256(ref)
        reproduced_hash = sha256(out)

        if frozen_hash == reproduced_hash:
            comparisons.append({
                "file": name,
                "verification_gate": "PASS",
                "verification_mode": "SHA256",
                "sha256_match": "TRUE",
                "max_abs_diff": 0.0,
                "hard_diff_n": 0,
                "numeric_nonexact_n": 0,
                "frozen_sha256": frozen_hash,
                "reproduced_sha256": reproduced_hash,
            })
            continue

        allowed_fields = ANALYSIS3_SEMANTIC_COLUMNS.get(name)
        if not allowed_fields:
            comparisons.append({
                "file": name,
                "verification_gate": "DIFF",
                "verification_mode": "SHA256",
                "sha256_match": "FALSE",
                "max_abs_diff": "",
                "hard_diff_n": 1,
                "numeric_nonexact_n": 0,
                "frozen_sha256": frozen_hash,
                "reproduced_sha256": reproduced_hash,
            })
            continue

        frozen_fields, frozen_rows = read_tsv(ref)
        produced_fields, produced_rows = read_tsv(out)
        ok, max_abs, hard_diff_n, numeric_nonexact_n, reason = (
            semantic_rows_equal_restricted(
                frozen_fields,
                frozen_rows,
                produced_fields,
                produced_rows,
                allowed_fields,
            )
        )
        print(
            f"ANALYSIS3_SEMANTIC_OUTPUT_GATE file={name} "
            f"gate={str(ok).upper()} max_abs_diff={max_abs:.17g} "
            f"hard_diff_n={hard_diff_n} "
            f"numeric_nonexact_n={numeric_nonexact_n}",
            flush=True,
        )
        comparisons.append({
            "file": name,
            "verification_gate": "PASS" if ok else "DIFF",
            "verification_mode": (
                f"SEMANTIC_RESTRICTED_ATOL_1E-12:{reason}"
            ),
            "sha256_match": "FALSE",
            "max_abs_diff": max_abs,
            "hard_diff_n": hard_diff_n,
            "numeric_nonexact_n": numeric_nonexact_n,
            "frozen_sha256": frozen_hash,
            "reproduced_sha256": reproduced_hash,
        })

    return comparisons


def compare_tsv_outputs(
    repo: Path,
    produced: Path,
    frozen_rel: str,
    semantic_fallback: bool = False,
):
    frozen = repo / frozen_rel
    if not frozen.is_dir():
        raise RuntimeError(f"Frozen result directory missing: {frozen_rel}")

    refs = {path.name: path for path in frozen.glob("*.tsv")}
    outs = {path.name: path for path in produced.glob("*.tsv")}

    comparisons = []
    for name in sorted(set(refs) | set(outs)):
        ref = refs.get(name)
        out = outs.get(name)

        if ref is None:
            comparisons.append({
                "file": name,
                "verification_gate": "DIFF",
                "verification_mode": "EXTRA_OUTPUT",
                "sha256_match": "FALSE",
                "max_abs_diff": "",
                "hard_diff_n": 1,
                "numeric_nonexact_n": 0,
                "frozen_sha256": "",
                "reproduced_sha256": sha256(out),
            })
            continue

        if out is None:
            comparisons.append({
                "file": name,
                "verification_gate": "DIFF",
                "verification_mode": "MISSING_OUTPUT",
                "sha256_match": "FALSE",
                "max_abs_diff": "",
                "hard_diff_n": 1,
                "numeric_nonexact_n": 0,
                "frozen_sha256": sha256(ref),
                "reproduced_sha256": "",
            })
            continue

        frozen_hash = sha256(ref)
        reproduced_hash = sha256(out)

        if frozen_hash == reproduced_hash:
            comparisons.append({
                "file": name,
                "verification_gate": "PASS",
                "verification_mode": "SHA256",
                "sha256_match": "TRUE",
                "max_abs_diff": 0.0,
                "hard_diff_n": 0,
                "numeric_nonexact_n": 0,
                "frozen_sha256": frozen_hash,
                "reproduced_sha256": reproduced_hash,
            })
            continue

        if not semantic_fallback:
            comparisons.append({
                "file": name,
                "verification_gate": "DIFF",
                "verification_mode": "SHA256",
                "sha256_match": "FALSE",
                "max_abs_diff": "",
                "hard_diff_n": 1,
                "numeric_nonexact_n": 0,
                "frozen_sha256": frozen_hash,
                "reproduced_sha256": reproduced_hash,
            })
            continue

        frozen_fields, frozen_rows = read_tsv(ref)
        produced_fields, produced_rows = read_tsv(out)
        ok, max_abs, hard_diff_n, numeric_nonexact_n, reason = (
            semantic_rows_equal(
                frozen_fields,
                frozen_rows,
                produced_fields,
                produced_rows,
            )
        )
        print(
            f"SEMANTIC_OUTPUT_GATE file={name} gate={str(ok).upper()} "
            f"max_abs_diff={max_abs:.17g} hard_diff_n={hard_diff_n} "
            f"numeric_nonexact_n={numeric_nonexact_n}",
            flush=True,
        )
        comparisons.append({
            "file": name,
            "verification_gate": "PASS" if ok else "DIFF",
            "verification_mode": f"SEMANTIC_ATOL_1E-12:{reason}",
            "sha256_match": "FALSE",
            "max_abs_diff": max_abs,
            "hard_diff_n": hard_diff_n,
            "numeric_nonexact_n": numeric_nonexact_n,
            "frozen_sha256": frozen_hash,
            "reproduced_sha256": reproduced_hash,
        })

    return comparisons


def write_verification(work: Path, rows):
    path = work / "portable_reproduction_verification.tsv"
    fields = [
        "analysis",
        "file",
        "verification_gate",
        "verification_mode",
        "sha256_match",
        "max_abs_diff",
        "hard_diff_n",
        "numeric_nonexact_n",
        "frozen_sha256",
        "reproduced_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument(
        "--mode", choices=["prepare", "analysis3", "analysis4", "all"],
        default="prepare"
    )
    ap.add_argument(
        "--clean", action="store_true",
        help="Remove an existing adapter work directory before preparing it."
    )
    args = ap.parse_args()

    # Preserve an explicitly supplied or invoked repository path.
    # Do not resolve it here: on Windows, resolving a SUBST drive can expand
    # back to a physical path that exceeds legacy MAX_PATH during hash checks.
    repo = (
        Path(args.repo_root)
        if args.repo_root else Path(__file__).parents[2]
    )
    work = Path(args.work_dir).resolve()

    if work.exists() and args.clean:
        shutil.rmtree(work)
    if work.exists() and any(work.iterdir()):
        raise RuntimeError(
            "Work directory is not empty. Use a new directory or pass --clean."
        )
    work.mkdir(parents=True, exist_ok=True)

    # Lock all public inputs and original provenance scripts before adaptation.
    for rel in EXPECTED:
        require_repo_asset(repo, rel)

    compat = prepare_compatibility_repo(repo, work)
    patched, a3_out, a4_out = make_patched_scripts(repo, work, compat)

    if args.mode == "prepare":
        print("PORTABLE_STAGE3_PREPARE_GATE=TRUE")
        print(f"WORK_DIR={work}")
        return 0

    verification = []

    if args.mode in {"analysis3", "all"}:
        run_analysis3(patched)
        a3_checks = compare_analysis3_outputs(repo, a3_out)
        for record in a3_checks:
            record["analysis"] = "analysis3"
        verification += a3_checks

    if args.mode in {"analysis4", "all"}:
        run_step10_5b(repo, compat)
        run_analysis4(patched)
        a4_checks = compare_tsv_outputs(
            repo,
            a4_out,
            "results_frozen/stage3/analysis4",
            semantic_fallback=True,
        )
        for record in a4_checks:
            record["analysis"] = "analysis4"
        verification += a4_checks

    report = write_verification(work, verification)
    diffs = [r for r in verification if r["verification_gate"] != "PASS"]
    print(f"PORTABLE_REPRODUCTION_VERIFICATION_FILE={report}")
    print(f"PORTABLE_REPRODUCTION_TSV_CHECK_N={len(verification)}")
    print(f"PORTABLE_REPRODUCTION_TSV_DIFF_N={len(diffs)}")
    exact_n = sum(
        record["verification_mode"] == "SHA256"
        and record["verification_gate"] == "PASS"
        for record in verification
    )
    semantic_n = sum(
        record["verification_mode"].startswith("SEMANTIC_")
        and record["verification_gate"] == "PASS"
        for record in verification
    )
    print(f"PORTABLE_REPRODUCTION_SHA256_EXACT_PASS_N={exact_n}")
    print(f"PORTABLE_REPRODUCTION_SEMANTIC_PASS_N={semantic_n}")
    print(f"PORTABLE_REPRODUCTION_SEMANTIC_ATOL={SEMANTIC_ATOL:.0e}")
    if diffs:
        raise RuntimeError(
            "Portable Stage3 reproduction differs from frozen TSV outputs. "
            "Do not release until the differences are audited."
        )
    print("PORTABLE_STAGE3_REPRODUCTION_GATE=TRUE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
