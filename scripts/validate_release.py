from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path

EXPECTED_RELEASE_FILE_N = 207
EXPECTED_MANIFEST_ROW_N = 205
EXPECTED_SUM_ROW_N = 206
EXPECTED_V2_PATIENT_N = 363
EXPECTED_V2_RESPONDER_N = 217
EXPECTED_V2_NONRESPONDER_N = 146
EXPECTED_V2_FEATURE_N = 2205
EXPECTED_ANALYSIS7_PREDICTION_N = 7260
EXPECTED_ANALYSIS7_FIT_N = 140

REQUIRED = {
    ".zenodo.json",
    "AUTHORS_AND_CONTRIBUTIONS.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE_DERIVED_MATERIALS.md",
    "README.md",
    "RELEASE_MANIFEST.tsv",
    "SHA256SUMS.txt",
    "metadata/release_authors_v1.0.0.tsv",
    "metadata/raw_data_accessions.tsv",
    "metadata/patient_specimen_run_profile_manifest_v1.0.0.tsv",
    "data_derived/V2_species_SGB_relative_abundance_v1.0.0.tsv.gz",
    "results_frozen/analysis7/predictions.tsv",
    "results_frozen/analysis7/fit_audit.tsv",
    "results_frozen/analysis7/fit_failures.tsv",
    "results_frozen/analysis7/benchmark_status.json",
    "results_frozen/analysis6/Table_S17_metadata_completeness_public_safe.tsv",
    "results_frozen/analysis7/Table_S18_DEBIASM_presentation.tsv",
    "results_frozen/validation/Step10_18C2_validation_hierarchy_effective_unit_summary.tsv",
    "figures/supplementary/Figure_S17_DEBIASM_secondary_benchmark.png",
    "docs/Step10_18C2_STORMS_reporting_crosswalk.docx",
}
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?i)(^|/)Step10_4B_results_v1\.zip$"),
    re.compile(r"(?i)(^|/)pilot(/|$)"),
    re.compile(r"(?i)(^|/)(private|secrets?|credentials?)(/|$)"),
    re.compile(r"(?i)\.(fastq|fq|bam|cram)(\.gz)?$"),
    re.compile(r"(?i)(^|/)\.env$"),
)
RELEASE_FACING_TEXT = (
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "AUTHORS_AND_CONTRIBUTIONS.md",
    "LICENSE_DERIVED_MATERIALS.md",
    "metadata/release_authors_v1.0.0.tsv",
)
PLACEHOLDER_PATTERNS = (
    re.compile(r"\[[A-Z0-9_ -]*(?:PENDING|PLACEHOLDER|TODO|TBD)[A-Z0-9_ -]*\]", re.I),
    re.compile(r"GITHUB_REPOSITORY_URL|ZENODO_DOI", re.I),
)
REPO_URL = "https://github.com/dr-che/microbiome-ici-transportability"
PING_ORCID = "0009-0007-4037-9982"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    warnings: list[str] = []

    # Integrity tables define the release file set and are safe in a Git checkout
    # where .git/ and local runtime files may also exist.
    manifest_path = root / "RELEASE_MANIFEST.tsv"
    sums_path = root / "SHA256SUMS.txt"
    if not manifest_path.is_file() or not sums_path.is_file():
        print(json.dumps({"status": "FAIL", "failures": ["INTEGRITY_FILES_MISSING"]}, indent=2))
        return 1

    manifest = read_tsv(manifest_path)
    manifest_map = {r["relative_path"]: r for r in manifest}
    sums_map = {}
    for line in sums_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            failures.append("SHA256SUMS_PARSE_ERROR")
            continue
        sums_map[rel] = digest.lower()

    if len(manifest) != EXPECTED_MANIFEST_ROW_N:
        failures.append(f"MANIFEST_ROW_N={len(manifest)}")
    if len(sums_map) != EXPECTED_SUM_ROW_N:
        failures.append(f"SHA256SUM_ROW_N={len(sums_map)}")

    tracked = set(manifest_map) | {"RELEASE_MANIFEST.tsv", "SHA256SUMS.txt"}
    if len(tracked) != EXPECTED_RELEASE_FILE_N:
        failures.append(f"TRACKED_RELEASE_FILE_N={len(tracked)}")

    missing_required = sorted(REQUIRED - tracked)
    if missing_required:
        failures.append("MISSING_REQUIRED=" + ";".join(missing_required))

    for rel, row in manifest_map.items():
        p = root / Path(rel)
        if not p.is_file():
            failures.append(f"MANIFEST_FILE_MISSING={rel}")
            continue
        observed_size = p.stat().st_size
        observed_hash = sha256(p)
        if observed_size != int(row["size_bytes"]):
            failures.append(f"MANIFEST_SIZE_FALSE={rel}")
        if observed_hash != row["sha256"].lower():
            failures.append(f"MANIFEST_HASH_FALSE={rel}")

    for rel, expected in sums_map.items():
        p = root / Path(rel)
        if not p.is_file():
            failures.append(f"SHA256SUM_FILE_MISSING={rel}")
            continue
        if sha256(p) != expected:
            failures.append(f"SHA256SUM_HASH_FALSE={rel}")

    expected_sum_paths = set(manifest_map) | {"RELEASE_MANIFEST.tsv"}
    if set(sums_map) != expected_sum_paths:
        failures.append("SHA256SUM_PATH_SET_FALSE")

    for rel in sorted(tracked):
        for pat in FORBIDDEN_PATH_PATTERNS:
            if pat.search(rel):
                failures.append(f"FORBIDDEN_PATH={rel}")

    # Release-facing text hygiene.
    for rel in RELEASE_FACING_TEXT:
        text = (root / rel).read_text(encoding="utf-8-sig", errors="strict")
        controls = [ord(c) for c in text if ord(c) < 32 and c not in "\t\n\r"]
        if controls:
            failures.append(f"CONTROL_CHARACTER={rel}")
        for pat in PLACEHOLDER_PATTERNS:
            if pat.search(text):
                failures.append(f"PLACEHOLDER={rel}")
        if re.search(r"[A-Za-z]:\\|/mnt/data/|/home/oai/", text):
            failures.append(f"LOCAL_PATH_RELEASE_FACING={rel}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    for rel in (
        "results_frozen/analysis6/Table_S17_metadata_completeness_public_safe.tsv",
        "figures/supplementary/Figure_S17_DEBIASM_secondary_benchmark.png",
        "results_frozen/analysis7/Table_S18_DEBIASM_presentation.tsv",
        "docs/Step10_18C2_STORMS_reporting_crosswalk.docx",
        "results_frozen/validation/Step10_18C2_validation_hierarchy_effective_unit_summary.tsv",
    ):
        if rel not in readme:
            failures.append(f"README_ARTIFACT_MAP_MISSING={rel}")

    # Metadata consistency using standard library only.
    zen = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    if len(zen.get("creators", [])) != 7:
        failures.append("ZENODO_CREATOR_N_FALSE")
    if zen.get("version") != "1.0.0":
        failures.append("ZENODO_VERSION_FALSE")
    if str(zen.get("license", "")).lower() != "mit":
        failures.append("ZENODO_LICENSE_FALSE")
    if any(k.lower() == "doi" for k in zen):
        failures.append("ZENODO_DOI_PRESENT_BEFORE_MINT")
    if zen.get("creators", [])[-1].get("orcid") != PING_ORCID:
        failures.append("ZENODO_PING_ORCID_FALSE")

    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    for required_text in ("cff-version: 1.2.0", "version: 1.0.0", "license: MIT", REPO_URL, PING_ORCID):
        if required_text not in cff:
            failures.append("CFF_REQUIRED_TEXT_MISSING=" + required_text)
    if re.search(r"(?mi)^\s*doi\s*:", cff):
        failures.append("CFF_DOI_PRESENT_BEFORE_MINT")

    authors = read_tsv(root / "metadata/release_authors_v1.0.0.tsv")
    if len(authors) != 7:
        failures.append("RELEASE_AUTHOR_N_FALSE")
    if [r["order"] for r in authors] != [str(i) for i in range(1, 8)]:
        failures.append("RELEASE_AUTHOR_ORDER_FALSE")
    if authors[-1]["orcid"] != PING_ORCID or authors[-1]["repository_contact"] != "TRUE":
        failures.append("RELEASE_PING_METADATA_FALSE")

    # Privacy-safe V2 matrix identity/shape.
    v2 = root / "data_derived/V2_species_SGB_relative_abundance_v1.0.0.tsv.gz"
    responder = nonresponder = row_n = 0
    with gzip.open(v2, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        feature_n = sum(1 for c in header if c.startswith("s__"))
        for row in reader:
            row_n += 1
            if not re.fullmatch(r"V2A\d{4}", row.get("manifest_id", "")):
                failures.append("V2_UNSAFE_MANIFEST_ID")
                break
            if row.get("patient_id") != row.get("manifest_id"):
                failures.append("V2_PATIENT_ID_TRANSFORM_FALSE")
                break
            if row.get("response_harmonized") == "Responder":
                responder += 1
            elif row.get("response_harmonized") == "Non-responder":
                nonresponder += 1
    if row_n != EXPECTED_V2_PATIENT_N:
        failures.append(f"V2_PATIENT_N={row_n}")
    if feature_n != EXPECTED_V2_FEATURE_N:
        failures.append(f"V2_FEATURE_N={feature_n}")
    if responder != EXPECTED_V2_RESPONDER_N or nonresponder != EXPECTED_V2_NONRESPONDER_N:
        failures.append(f"V2_RESPONSE_COUNTS={responder}/{nonresponder}")

    preds = read_tsv(root / "results_frozen/analysis7/predictions.tsv")
    if len(preds) != EXPECTED_ANALYSIS7_PREDICTION_N:
        failures.append(f"ANALYSIS7_PREDICTION_N={len(preds)}")
    if any(not re.fullmatch(r"V2A\d{4}", r.get("manifest_id", "")) for r in preds):
        failures.append("ANALYSIS7_UNSAFE_MANIFEST_ID")

    fits = read_tsv(root / "results_frozen/analysis7/fit_audit.tsv")
    if len(fits) != EXPECTED_ANALYSIS7_FIT_N:
        failures.append(f"ANALYSIS7_FIT_N={len(fits)}")
    failures_rows = read_tsv(root / "results_frozen/analysis7/fit_failures.tsv")
    if failures_rows:
        failures.append(f"ANALYSIS7_FIT_FAILURE_N={len(failures_rows)}")

    status = json.loads((root / "results_frozen/analysis7/benchmark_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("pilot_only") is not False:
        failures.append("ANALYSIS7_STATUS_FALSE")

    # Python syntax only — no scientific execution.
    syntax_fail = []
    for p in sorted((root / "scripts").rglob("*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        except Exception as exc:
            syntax_fail.append(f"{p.relative_to(root).as_posix()}:{exc}")
    if syntax_fail:
        failures.append("PYTHON_SYNTAX_FAILURE=" + ";".join(syntax_fail))

    report = {
        "status": "PASS" if not failures else "FAIL",
        "release_file_n": len(tracked),
        "manifest_row_n": len(manifest),
        "sha256sum_row_n": len(sums_map),
        "v2_patient_n": row_n,
        "v2_responder_n": responder,
        "v2_nonresponder_n": nonresponder,
        "v2_species_feature_n": feature_n,
        "analysis7_fit_n": len(fits),
        "analysis7_prediction_n": len(preds),
        "failures": failures,
        "warnings": warnings,
        "scientific_rerun_executed": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
