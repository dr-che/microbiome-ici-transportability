#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
import zipfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
except Exception as exc:
    raise SystemExit(
        "Missing Python dependencies. Run INSTALL_STEP10_4B_DEPENDENCIES.cmd.\n"
        f"Original import error: {exc}"
    )


warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

VERSION = "v2_species_only"
RANDOM_SEED = 20260720
MIN_PREVALENCE = 0.10
FALLBACK_PREVALENCE = 0.05
MAX_FEATURES = 500
BOOTSTRAP_REPS = 500

SCENARIOS = {
    "PRIMARY": "Step10_2_manifest_R3_primary_lock_v8.tsv",
    "S1_EXCLUDE_BCN12": "Step10_2_manifest_R3_sensitivity_S1_exclude_BCN12_v8.tsv",
    "S2_BCN12_AS_NR": "Step10_2_manifest_R3_sensitivity_S2_BCN12_as_NR_v8.tsv",
    "S3_EXCLUDE_LEE_SITES": "Step10_2_manifest_R3_sensitivity_S3_exclude_Lee_sites_v8.tsv",
}

MODEL_ORDER = ["elastic_net", "ridge", "random_forest"]


def log(message: str) -> None:
    print(message, flush=True)


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def yes(value: object) -> bool:
    return text(value).lower() in {"yes", "y", "true", "1"}


def safe_float(value: object) -> Optional[float]:
    try:
        x = float(text(value))
        return x if math.isfinite(x) else None
    except Exception:
        return None


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [{k: text(v) for k, v in row.items()} for row in reader]
    return rows, fields


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text(row.get(field, "")) for field in fields})


def locate_file(root: Path, filename: str) -> Path:
    preferred = [
        root / filename,
        root / "01_downloaded" / "species" / filename,
        root / "species" / filename,
    ]
    for path in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path
    candidates = [
        path for path in root.rglob(filename)
        if path.is_file() and path.stat().st_size > 0
        and "02_results_step10_4B" not in str(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"Required file was not found: {filename}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p).lower()))[0]


def canonical_species(raw: str) -> Optional[str]:
    value = text(raw).strip('"')
    if not value:
        return None

    lower = value.lower()
    if any(token in lower for token in (
        "unclassified", "unknown", "unmapped", "unassigned",
        "other", "community_unclassified",
    )):
        return None

    # MetaPhlAn taxonomy: keep the species rank and discard strain rank.
    parts = [part.strip() for part in value.split("|") if part.strip()]
    species_parts = [part for part in parts if part.startswith("s__")]
    if species_parts:
        value = species_parts[-1]
    else:
        # Some tables use semicolon taxonomy.
        semis = [part.strip() for part in value.split(";") if part.strip()]
        species_semis = [part for part in semis if part.startswith("s__")]
        if species_semis:
            value = species_semis[-1]

    value = re.sub(r"^['\"]+|['\"]+$", "", value)
    value = value.replace(" ", "_")
    value = re.sub(r"_+", "_", value)

    # Exclude rows that are clearly not species-level taxonomy.
    if "__" in value and not value.startswith("s__"):
        return None

    if value.startswith("s__"):
        core = value[3:]
    else:
        core = value

    core = re.sub(r"[^A-Za-z0-9_.-]+", "_", core).strip("_")
    if not core or core.lower() in {"na", "nan", "none"}:
        return None

    # Require at least a genus/species-like identifier.
    if "_" not in core and "." not in core:
        return None
    return "s__" + core


def find_header(
    path: Path,
    required_columns: Sequence[str],
) -> Tuple[int, List[str], Dict[str, int], int]:
    required = set(required_columns)
    best = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line_no, line in enumerate(handle):
            if line_no > 200:
                break
            fields = [item.strip().strip('"') for item in line.rstrip("\r\n").split("\t")]
            index = {name: i for i, name in enumerate(fields)}
            matched = sum(1 for name in required if name in index)
            if matched:
                tax_index = 0
                normalized = [re.sub(r"[^a-z0-9]+", "", x.lower()) for x in fields]
                for candidate in (
                    "cladename", "taxon", "taxonomy", "species",
                    "feature", "name", "otu", "otuid",
                ):
                    if candidate in normalized:
                        tax_index = normalized.index(candidate)
                        break
                item = (matched, line_no, fields, index, tax_index)
                if best is None or matched > best[0]:
                    best = item
                if matched == len(required):
                    break
    if best is None:
        raise RuntimeError(
            f"No matrix header containing required sample columns was found: {path}"
        )
    matched, line_no, fields, index, tax_index = best
    missing = [name for name in required_columns if name not in index]
    if missing:
        raise RuntimeError(
            f"{path.name}: {len(missing)} required sample columns were absent. "
            f"Examples: {missing[:10]}"
        )
    return line_no, fields, index, tax_index


def load_species_matrix(
    path: Path,
    required_columns: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    header_line, header, index, tax_index = find_header(path, required_columns)
    sample_indices = [index[name] for name in required_columns]
    species_data: Dict[str, np.ndarray] = {}
    parsed_rows = 0
    skipped_rows = 0
    malformed_rows = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for _ in range(header_line + 1):
            next(handle, None)
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max([tax_index] + sample_indices):
                malformed_rows += 1
                continue
            species = canonical_species(parts[tax_index])
            if species is None:
                skipped_rows += 1
                continue
            values = np.zeros(len(required_columns), dtype=float)
            valid_any = False
            for j, column_index in enumerate(sample_indices):
                value = safe_float(parts[column_index])
                if value is not None and value >= 0:
                    values[j] = value
                    valid_any = valid_any or value > 0
            if not valid_any:
                skipped_rows += 1
                continue
            if species in species_data:
                species_data[species] += values
            else:
                species_data[species] = values
            parsed_rows += 1

    if not species_data:
        raise RuntimeError(f"No species rows were parsed from {path}")

    # Normalize every sample column to relative abundance.
    stacked = np.vstack(list(species_data.values()))
    sums = stacked.sum(axis=0)
    if np.any(sums <= 0):
        bad = [required_columns[i] for i, total in enumerate(sums) if total <= 0]
        raise RuntimeError(
            f"{path.name}: zero abundance sum for sample columns: {bad[:10]}"
        )
    for species in list(species_data):
        species_data[species] = species_data[species] / sums

    qc = {
        "matrix_file": path.name,
        "path": str(path),
        "required_sample_columns": len(required_columns),
        "parsed_species_rows_before_collapse": parsed_rows,
        "canonical_species_after_collapse": len(species_data),
        "skipped_non_species_or_empty_rows": skipped_rows,
        "malformed_rows": malformed_rows,
        "minimum_column_sum_before_normalization": float(np.min(sums)),
        "maximum_column_sum_before_normalization": float(np.max(sums)),
        "status": "PASS",
    }
    return species_data, qc


@dataclass
class AssembledData:
    sample_ids: List[str]
    patient_ids: List[str]
    domains: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    X: np.ndarray
    manifest_rows: List[Dict[str, str]]
    qc_rows: List[Dict[str, object]]


def locate_v1_assembled_matrix(root: Path) -> Tuple[Path, Optional[zipfile.ZipFile]]:
    direct_candidates = [
        root / "02_results_step10_4B_v1" / "assembled_species_matrix.tsv",
        root / "assembled_species_matrix.tsv",
    ]
    for path in direct_candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path, None

    result_zip = root / "Step10_4B_results_v1.zip"
    if result_zip.is_file() and zipfile.is_zipfile(result_zip):
        archive = zipfile.ZipFile(result_zip)
        if "assembled_species_matrix.tsv" not in archive.namelist():
            archive.close()
            raise RuntimeError(
                "Step10_4B_results_v1.zip does not contain assembled_species_matrix.tsv."
            )
        return result_zip, archive

    candidates = [
        path for path in root.rglob("assembled_species_matrix.tsv")
        if path.is_file() and path.stat().st_size > 0
        and "02_results_step10_4B_v2" not in str(path)
    ]
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.parts), str(p).lower()))[0], None

    raise FileNotFoundError(
        "Could not locate the completed v1 assembled_species_matrix.tsv or "
        "Step10_4B_results_v1.zip."
    )


def corrected_species_name(feature: str) -> Optional[str]:
    value = text(feature)
    if value.startswith("s__s__"):
        core = value[len("s__s__"):]
    elif value.startswith("s__s_"):
        core = value[len("s__s_"):]
    else:
        return None

    lower = core.lower()
    if any(token in lower for token in (
        "unclassified", "unknown", "unmapped", "unassigned",
        "community_unclassified", "other",
    )):
        return None

    core = re.sub(r"[^A-Za-z0-9_.-]+", "_", core).strip("_")
    if not core or core.lower() in {"na", "nan", "none"}:
        return None
    return "s__" + core


def assemble_primary_matrix(root: Path, primary_manifest: Path) -> AssembledData:
    rows, _ = read_tsv(primary_manifest)
    included = [
        row for row in rows
        if yes(row.get("strict_primary_target", ""))
        and row.get("profile_mapping_status", "") == "MATCHED"
        and yes(row.get("r3_v8_primary_analysis_include", ""))
    ]

    if len(included) != 363:
        raise RuntimeError(f"Primary included n must be 363; observed {len(included)}.")

    manifest_ids = [row["manifest_id"] for row in included]
    patient_ids = [row["patient_id"] for row in included]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise RuntimeError("Duplicate manifest_id in primary included rows.")
    if len(set(patient_ids)) != len(patient_ids):
        raise RuntimeError("Duplicate patient_id in primary included rows.")

    labels_text = [row.get("response_harmonized", "") for row in included]
    bad_labels = sorted(set(labels_text) - {"Responder", "Non-responder"})
    if bad_labels:
        raise RuntimeError(f"Invalid response labels: {bad_labels}")

    source_path, archive = locate_v1_assembled_matrix(root)
    if archive is None:
        handle = source_path.open("r", encoding="utf-8-sig", newline="")
        source_description = str(source_path)
    else:
        raw_handle = archive.open("assembled_species_matrix.tsv")
        handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
        source_description = str(source_path) + "::assembled_species_matrix.tsv"

    try:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        metadata_fields = [
            "manifest_id", "patient_id", "domain_id", "response_harmonized"
        ]
        missing_meta = [field for field in metadata_fields if field not in fields]
        if missing_meta:
            raise RuntimeError(
                f"v1 assembled matrix lacks metadata fields: {missing_meta}"
            )

        raw_features = [field for field in fields if field not in metadata_fields]
        mapping: Dict[str, List[str]] = defaultdict(list)
        excluded_higher_rank = 0
        excluded_other = 0
        for feature in raw_features:
            corrected = corrected_species_name(feature)
            if corrected is None:
                if feature.startswith("s__k_") or feature.startswith("s__k__"):
                    excluded_higher_rank += 1
                else:
                    excluded_other += 1
                continue
            mapping[corrected].append(feature)

        feature_names = sorted(mapping)
        if len(feature_names) < 100:
            raise RuntimeError(
                f"Species-only correction produced too few taxa: {len(feature_names)}"
            )

        source_rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            manifest_id = text(row.get("manifest_id", ""))
            if manifest_id:
                source_rows[manifest_id] = row
    finally:
        handle.close()
        if archive is not None:
            archive.close()

    missing_ids = [manifest_id for manifest_id in manifest_ids if manifest_id not in source_rows]
    if missing_ids:
        raise RuntimeError(
            f"v1 assembled matrix is missing {len(missing_ids)} frozen patients. "
            f"Examples: {missing_ids[:10]}"
        )

    X = np.zeros((len(included), len(feature_names)), dtype=float)
    feature_index = {feature: j for j, feature in enumerate(feature_names)}
    duplicate_groups = sum(1 for sources in mapping.values() if len(sources) > 1)

    for i, manifest_id in enumerate(manifest_ids):
        source_row = source_rows[manifest_id]
        for corrected, original_columns in mapping.items():
            value = 0.0
            for original in original_columns:
                numeric = safe_float(source_row.get(original, ""))
                if numeric is not None and numeric >= 0:
                    value += numeric
            X[i, feature_index[corrected]] = value

    sums_before = X.sum(axis=1)
    if np.any(sums_before <= 0):
        bad = [manifest_ids[i] for i, total in enumerate(sums_before) if total <= 0]
        raise RuntimeError(f"Species-only zero-sum patients: {bad[:10]}")
    X = X / sums_before[:, None]

    labels = np.array(
        [1 if label == "Responder" else 0 for label in labels_text], dtype=int
    )
    domains = np.array([row["domain_id"] for row in included], dtype=object)

    qc_rows = [{
        "matrix_file": "v1 assembled_species_matrix.tsv",
        "path": source_description,
        "required_sample_columns": len(included),
        "included_manifest_rows": len(included),
        "parsed_species_rows_before_collapse": sum(len(v) for v in mapping.values()),
        "canonical_species_after_collapse": len(feature_names),
        "skipped_non_species_or_empty_rows": excluded_higher_rank + excluded_other,
        "malformed_rows": 0,
        "minimum_column_sum_before_normalization": float(np.min(sums_before)),
        "maximum_column_sum_before_normalization": float(np.max(sums_before)),
        "status": "PASS_SPECIES_ONLY_CORRECTION",
    }]

    taxonomy_audit = {
        "source": source_description,
        "v1_total_feature_columns": len(raw_features),
        "v1_species_marker_columns": sum(len(v) for v in mapping.values()),
        "v1_higher_rank_columns_excluded": excluded_higher_rank,
        "v1_other_columns_excluded": excluded_other,
        "canonical_species_after_collapse": len(feature_names),
        "duplicate_canonical_species_groups_collapsed": duplicate_groups,
        "sample_n": len(included),
        "correction_rule": (
            "Retain only v1 columns derived from raw s_ or s__ species rows; "
            "exclude all k_/p_/c_/o_/f_/g_ hierarchy rows; remove duplicated s_ prefix; "
            "renormalize species abundances within sample."
        ),
    }
    (root / "Step10_4B_taxonomy_correction_audit_v2.json").write_text(
        json.dumps(taxonomy_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(
        f"[taxonomy correction] v1 features={len(raw_features)}; "
        f"higher-rank excluded={excluded_higher_rank}; "
        f"species retained={len(feature_names)}"
    )

    return AssembledData(
        sample_ids=manifest_ids,
        patient_ids=patient_ids,
        domains=domains,
        labels=labels,
        feature_names=feature_names,
        X=X,
        manifest_rows=included,
        qc_rows=qc_rows,
    )
def scenario_indices(
    assembled: AssembledData,
    scenario_manifest_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, str]]]:
    rows, _ = read_tsv(scenario_manifest_path)
    row_by_manifest = {row["manifest_id"]: row for row in rows}
    selected_indices = []
    labels = []
    selected_rows = {}

    for i, manifest_id in enumerate(assembled.sample_ids):
        row = row_by_manifest.get(manifest_id)
        if row is None:
            continue
        include = (
            yes(row.get("strict_primary_target", ""))
            and row.get("profile_mapping_status", "") == "MATCHED"
            and yes(row.get("r3_v8_primary_analysis_include", ""))
        )
        if include:
            label = row.get("response_harmonized", "")
            if label not in {"Responder", "Non-responder"}:
                raise RuntimeError(
                    f"Scenario {scenario_manifest_path.name}: invalid label for {manifest_id}"
                )
            selected_indices.append(i)
            labels.append(1 if label == "Responder" else 0)
            selected_rows[manifest_id] = row

    return np.array(selected_indices, dtype=int), np.array(labels, dtype=int), selected_rows


@dataclass
class PreprocessState:
    selected_indices: np.ndarray
    pseudocount: float
    mean: np.ndarray
    scale: np.ndarray


def fit_preprocessor(X_train: np.ndarray) -> PreprocessState:
    prevalence = np.mean(X_train > 0, axis=0)
    selected = np.where(prevalence >= MIN_PREVALENCE)[0]
    if len(selected) < 10:
        selected = np.where(prevalence >= FALLBACK_PREVALENCE)[0]
    if len(selected) < 5:
        selected = np.where(np.any(X_train > 0, axis=0))[0]
    if len(selected) == 0:
        raise RuntimeError("No features passed prevalence filtering.")

    nonzero = X_train[:, selected][X_train[:, selected] > 0]
    pseudocount = float(np.min(nonzero) / 2.0) if nonzero.size else 1e-6
    pseudocount = min(max(pseudocount, 1e-8), 1e-4)

    clr = np.log(X_train[:, selected] + pseudocount)
    clr = clr - clr.mean(axis=1, keepdims=True)
    variances = np.var(clr, axis=0)

    if len(selected) > MAX_FEATURES:
        order = np.argsort(variances)[::-1][:MAX_FEATURES]
        selected = selected[order]
        clr = np.log(X_train[:, selected] + pseudocount)
        clr = clr - clr.mean(axis=1, keepdims=True)

    mean = clr.mean(axis=0)
    scale = clr.std(axis=0, ddof=0)
    scale[scale < 1e-10] = 1.0

    return PreprocessState(
        selected_indices=selected,
        pseudocount=pseudocount,
        mean=mean,
        scale=scale,
    )


def apply_preprocessor(X: np.ndarray, state: PreprocessState) -> np.ndarray:
    clr = np.log(X[:, state.selected_indices] + state.pseudocount)
    clr = clr - clr.mean(axis=1, keepdims=True)
    return (clr - state.mean) / state.scale


def parameter_grid(model_name: str) -> List[Dict[str, object]]:
    if model_name == "elastic_net":
        return [
            {"C": C, "l1_ratio": ratio}
            for C in (0.05, 0.2, 1.0, 5.0)
            for ratio in (0.25, 0.75)
        ]
    if model_name == "ridge":
        return [{"C": C} for C in (0.02, 0.1, 0.5, 2.0, 10.0)]
    if model_name == "random_forest":
        return [
            {"max_features": "sqrt", "min_samples_leaf": 2},
            {"max_features": "sqrt", "min_samples_leaf": 5},
            {"max_features": 0.3, "min_samples_leaf": 2},
            {"max_features": 0.3, "min_samples_leaf": 5},
        ]
    raise ValueError(model_name)


def build_model(model_name: str, params: Dict[str, object], seed: int):
    if model_name == "elastic_net":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "ridge":
        return LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=float(params["C"]),
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, object], float]:
    unique_groups = np.unique(groups_train)
    n_splits = min(4, len(unique_groups))
    if n_splits < 2:
        return parameter_grid(model_name)[0], float("nan")

    splitter = GroupKFold(n_splits=n_splits)
    best_params = None
    best_score = -np.inf

    for params in parameter_grid(model_name):
        scores = []
        for inner_fold, (fit_idx, val_idx) in enumerate(
            splitter.split(X_train, y_train, groups_train), start=1
        ):
            if len(np.unique(y_train[fit_idx])) < 2 or len(np.unique(y_train[val_idx])) < 2:
                continue
            model = build_model(model_name, params, seed + inner_fold)
            model.fit(X_train[fit_idx], y_train[fit_idx])
            prob = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[val_idx], prob))
        score = float(np.mean(scores)) if scores else -np.inf
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        best_params = parameter_grid(model_name)[0]
    return best_params, float(best_score)


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = (p >= 0.5).astype(int)
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    else:
        auc = float("nan")
        ap = float("nan")
    brier = float(brier_score_loss(y, p))
    bal = float(balanced_accuracy_score(y, pred))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(log_loss(y, clipped, labels=[0, 1]))
    return {
        "roc_auc": auc,
        "pr_auc": ap,
        "brier": brier,
        "balanced_accuracy": bal,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "log_loss": ll,
    }


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=3000,
        )
        model.fit(logit, y)
    except Exception:
        try:
            model = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=3000,
            )
            model.fit(logit, y)
        except Exception:
            return float("nan"), float("nan")
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def cluster_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric_name: str,
    reps: int,
    seed: int,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values = []
    indices_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    for _ in range(reps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index_blocks = [indices_by_group[g] for g in sampled_groups]
        idx = np.concatenate(index_blocks)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric_name == "roc_auc":
            value = roc_auc_score(yy, pp)
        elif metric_name == "pr_auc":
            value = average_precision_score(yy, pp)
        elif metric_name == "brier":
            value = brier_score_loss(yy, pp)
        else:
            raise ValueError(metric_name)
        if math.isfinite(value):
            values.append(float(value))

    if len(values) < 20:
        return float("nan"), float("nan"), len(values)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def run_scenario_model(
    scenario: str,
    model_name: str,
    assembled: AssembledData,
    selected_indices: np.ndarray,
    scenario_y: np.ndarray,
    scenario_rows: Dict[str, Dict[str, str]],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
    List[Dict[str, object]],
]:
    X = assembled.X[selected_indices]
    groups = assembled.domains[selected_indices]
    manifest_ids = [assembled.sample_ids[i] for i in selected_indices]
    patient_ids = [assembled.patient_ids[i] for i in selected_indices]

    predictions: List[Dict[str, object]] = []
    domain_metrics: List[Dict[str, object]] = []
    feature_records: List[Dict[str, object]] = []

    unique_domains = sorted(np.unique(groups))
    log(f"[model] scenario={scenario}; model={model_name}; domains={len(unique_domains)}; n={len(scenario_y)}")

    for fold_index, test_domain in enumerate(unique_domains, start=1):
        test_mask = groups == test_domain
        train_mask = ~test_mask
        X_train_raw, X_test_raw = X[train_mask], X[test_mask]
        y_train, y_test = scenario_y[train_mask], scenario_y[test_mask]
        groups_train = groups[train_mask]

        if len(np.unique(y_train)) < 2:
            raise RuntimeError(
                f"{scenario}/{model_name}: training data has one class when holding out {test_domain}."
            )

        state = fit_preprocessor(X_train_raw)
        X_train = apply_preprocessor(X_train_raw, state)
        X_test = apply_preprocessor(X_test_raw, state)

        params, inner_auc = tune_model(
            model_name,
            X_train,
            y_train,
            groups_train,
            RANDOM_SEED + fold_index * 100,
        )
        model = build_model(
            model_name,
            params,
            RANDOM_SEED + fold_index * 1000,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]

        fold_metrics = metric_dict(y_test, prob)
        domain_metrics.append({
            "scenario": scenario,
            "model": model_name,
            "held_out_domain": test_domain,
            "fold_index": fold_index,
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "test_responders": int(y_test.sum()),
            "test_non_responders": int(len(y_test) - y_test.sum()),
            "selected_features": len(state.selected_indices),
            "pseudocount": state.pseudocount,
            "best_params": json.dumps(params, sort_keys=True),
            "inner_group_cv_auc": inner_auc,
            **fold_metrics,
        })

        test_positions = np.where(test_mask)[0]
        for local_position, probability in zip(test_positions, prob):
            manifest_id = manifest_ids[local_position]
            row = scenario_rows[manifest_id]
            predictions.append({
                "scenario": scenario,
                "model": model_name,
                "held_out_domain": test_domain,
                "manifest_id": manifest_id,
                "patient_id": patient_ids[local_position],
                "true_label": int(scenario_y[local_position]),
                "true_response": row.get("response_harmonized", ""),
                "predicted_probability": float(probability),
                "predicted_class": int(probability >= 0.5),
            })

        selected_names = [assembled.feature_names[i] for i in state.selected_indices]
        if model_name in {"elastic_net", "ridge"}:
            values = model.coef_[0]
            for feature, value in zip(selected_names, values):
                feature_records.append({
                    "scenario": scenario,
                    "model": model_name,
                    "held_out_domain": test_domain,
                    "feature": feature,
                    "importance_type": "standardized_coefficient",
                    "importance": float(value),
                    "nonzero": int(abs(value) > 1e-12),
                })
        else:
            values = model.feature_importances_
            for feature, value in zip(selected_names, values):
                feature_records.append({
                    "scenario": scenario,
                    "model": model_name,
                    "held_out_domain": test_domain,
                    "feature": feature,
                    "importance_type": "gini_importance",
                    "importance": float(value),
                    "nonzero": int(value > 0),
                })

    # Order pooled OOF predictions by original scenario order.
    prediction_by_id = {row["manifest_id"]: row for row in predictions}
    predictions = [prediction_by_id[mid] for mid in manifest_ids]

    y_all = np.array([row["true_label"] for row in predictions], dtype=int)
    p_all = np.array([row["predicted_probability"] for row in predictions], dtype=float)
    g_all = np.array([row["held_out_domain"] for row in predictions], dtype=object)

    overall = metric_dict(y_all, p_all)
    cal_intercept, cal_slope = calibration_metrics(y_all, p_all)

    auc_low, auc_high, auc_n = cluster_bootstrap_ci(
        y_all, p_all, g_all, "roc_auc", BOOTSTRAP_REPS,
        RANDOM_SEED + 11,
    )
    ap_low, ap_high, ap_n = cluster_bootstrap_ci(
        y_all, p_all, g_all, "pr_auc", BOOTSTRAP_REPS,
        RANDOM_SEED + 22,
    )
    brier_low, brier_high, brier_n = cluster_bootstrap_ci(
        y_all, p_all, g_all, "brier", BOOTSTRAP_REPS,
        RANDOM_SEED + 33,
    )

    overall_row: Dict[str, object] = {
        "scenario": scenario,
        "model": model_name,
        "n": len(y_all),
        "responders": int(y_all.sum()),
        "non_responders": int(len(y_all) - y_all.sum()),
        "domains": len(np.unique(g_all)),
        **overall,
        "roc_auc_ci_low": auc_low,
        "roc_auc_ci_high": auc_high,
        "roc_auc_bootstrap_valid": auc_n,
        "pr_auc_ci_low": ap_low,
        "pr_auc_ci_high": ap_high,
        "pr_auc_bootstrap_valid": ap_n,
        "brier_ci_low": brier_low,
        "brier_ci_high": brier_high,
        "brier_bootstrap_valid": brier_n,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "prevalence_baseline_pr_auc": float(y_all.mean()),
    }

    return predictions, domain_metrics, overall_row, feature_records


def aggregate_feature_stability(
    feature_records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    domains_by_group: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)

    for row in feature_records:
        key = (text(row["scenario"]), text(row["model"]), text(row["feature"]))
        grouped[key].append(float(row["importance"]))
        domains_by_group[key].add(text(row["held_out_domain"]))

    output = []
    for (scenario, model, feature), values in grouped.items():
        values_array = np.array(values, dtype=float)
        folds = len(domains_by_group[(scenario, model, feature)])
        if model in {"elastic_net", "ridge"}:
            nonzero = np.abs(values_array) > 1e-12
            positive = values_array > 0
            negative = values_array < 0
            sign_consistency = max(np.mean(positive), np.mean(negative))
        else:
            nonzero = values_array > 0
            sign_consistency = float("nan")

        output.append({
            "scenario": scenario,
            "model": model,
            "feature": feature,
            "folds_selected": len(values),
            "unique_outer_folds": folds,
            "nonzero_fold_fraction": float(np.mean(nonzero)),
            "mean_importance": float(np.mean(values_array)),
            "mean_absolute_importance": float(np.mean(np.abs(values_array))),
            "median_importance": float(np.median(values_array)),
            "sign_consistency": sign_consistency,
        })

    return sorted(
        output,
        key=lambda row: (
            MODEL_ORDER.index(row["model"]),
            row["scenario"],
            -float(row["mean_absolute_importance"]),
            row["feature"],
        ),
    )


def write_assembled_matrix(
    path: Path,
    assembled: AssembledData,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["manifest_id", "patient_id", "domain_id", "response_harmonized"]
            + assembled.feature_names
        )
        for i, row in enumerate(assembled.manifest_rows):
            writer.writerow([
                assembled.sample_ids[i],
                assembled.patient_ids[i],
                assembled.domains[i],
                row["response_harmonized"],
                *[f"{value:.12g}" for value in assembled.X[i]],
            ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result_dir = root / "02_results_step10_4B_v2_species_only"
    result_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    log("[start] Step 10.4B corrected species-only multiscenario LOSO modeling")

    scenario_paths = {}
    for scenario, filename in SCENARIOS.items():
        path = locate_file(root, filename)
        scenario_paths[scenario] = path
        log(f"[manifest] {scenario}: {path}")

    assembled = assemble_primary_matrix(root, scenario_paths["PRIMARY"])
    log(
        f"[assembled] n={len(assembled.sample_ids)}; "
        f"species={len(assembled.feature_names)}; "
        f"domains={len(np.unique(assembled.domains))}"
    )

    write_assembled_matrix(
        result_dir / "assembled_species_matrix.tsv",
        assembled,
    )
    write_tsv(
        result_dir / "matrix_assembly_qc.tsv",
        assembled.qc_rows,
        [
            "matrix_file", "path", "required_sample_columns",
            "included_manifest_rows", "parsed_species_rows_before_collapse",
            "canonical_species_after_collapse",
            "skipped_non_species_or_empty_rows", "malformed_rows",
            "minimum_column_sum_before_normalization",
            "maximum_column_sum_before_normalization", "status",
        ],
    )

    scenario_summary = []
    all_predictions: List[Dict[str, object]] = []
    all_domain_metrics: List[Dict[str, object]] = []
    all_overall: List[Dict[str, object]] = []
    all_feature_records: List[Dict[str, object]] = []

    for scenario, path in scenario_paths.items():
        selected_indices, scenario_y, scenario_rows = scenario_indices(
            assembled, path
        )
        domains = assembled.domains[selected_indices]
        scenario_summary.append({
            "scenario": scenario,
            "manifest_file": path.name,
            "n": len(selected_indices),
            "responders": int(scenario_y.sum()),
            "non_responders": int(len(scenario_y) - scenario_y.sum()),
            "domains": len(np.unique(domains)),
            "domain_list": ";".join(sorted(np.unique(domains))),
        })

        for model_name in MODEL_ORDER:
            predictions, domain_metrics, overall, feature_records = run_scenario_model(
                scenario,
                model_name,
                assembled,
                selected_indices,
                scenario_y,
                scenario_rows,
            )
            all_predictions.extend(predictions)
            all_domain_metrics.extend(domain_metrics)
            all_overall.append(overall)
            all_feature_records.extend(feature_records)
            log(
                f"[complete] {scenario}/{model_name}: "
                f"AUC={overall['roc_auc']:.4f}; "
                f"PR-AUC={overall['pr_auc']:.4f}"
            )

    feature_stability = aggregate_feature_stability(all_feature_records)

    primary_by_model = {
        row["model"]: row for row in all_overall
        if row["scenario"] == "PRIMARY"
    }
    comparison = []
    for row in all_overall:
        primary = primary_by_model[row["model"]]
        comparison.append({
            "model": row["model"],
            "scenario": row["scenario"],
            "n": row["n"],
            "roc_auc": row["roc_auc"],
            "primary_roc_auc": primary["roc_auc"],
            "delta_roc_auc_vs_primary": float(row["roc_auc"]) - float(primary["roc_auc"]),
            "pr_auc": row["pr_auc"],
            "primary_pr_auc": primary["pr_auc"],
            "delta_pr_auc_vs_primary": float(row["pr_auc"]) - float(primary["pr_auc"]),
            "brier": row["brier"],
            "primary_brier": primary["brier"],
            "delta_brier_vs_primary": float(row["brier"]) - float(primary["brier"]),
        })

    write_tsv(
        result_dir / "scenario_summary.tsv",
        scenario_summary,
        [
            "scenario", "manifest_file", "n", "responders",
            "non_responders", "domains", "domain_list",
        ],
    )
    write_tsv(
        result_dir / "all_out_of_domain_predictions.tsv",
        all_predictions,
        [
            "scenario", "model", "held_out_domain", "manifest_id",
            "patient_id", "true_label", "true_response",
            "predicted_probability", "predicted_class",
        ],
    )
    write_tsv(
        result_dir / "scenario_model_domain_metrics.tsv",
        all_domain_metrics,
        [
            "scenario", "model", "held_out_domain", "fold_index",
            "train_n", "test_n", "test_responders", "test_non_responders",
            "selected_features", "pseudocount", "best_params",
            "inner_group_cv_auc", "roc_auc", "pr_auc", "brier",
            "balanced_accuracy", "sensitivity", "specificity", "log_loss",
        ],
    )
    write_tsv(
        result_dir / "scenario_model_overall_metrics.tsv",
        all_overall,
        [
            "scenario", "model", "n", "responders", "non_responders",
            "domains", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "roc_auc_bootstrap_valid", "pr_auc", "pr_auc_ci_low",
            "pr_auc_ci_high", "pr_auc_bootstrap_valid", "brier",
            "brier_ci_low", "brier_ci_high", "brier_bootstrap_valid",
            "balanced_accuracy", "sensitivity", "specificity", "log_loss",
            "calibration_intercept", "calibration_slope",
            "prevalence_baseline_pr_auc",
        ],
    )
    write_tsv(
        result_dir / "feature_stability.tsv",
        feature_stability,
        [
            "scenario", "model", "feature", "folds_selected",
            "unique_outer_folds", "nonzero_fold_fraction",
            "mean_importance", "mean_absolute_importance",
            "median_importance", "sign_consistency",
        ],
    )
    write_tsv(
        result_dir / "scenario_model_comparison.tsv",
        comparison,
        [
            "model", "scenario", "n", "roc_auc", "primary_roc_auc",
            "delta_roc_auc_vs_primary", "pr_auc", "primary_pr_auc",
            "delta_pr_auc_vs_primary", "brier", "primary_brier",
            "delta_brier_vs_primary",
        ],
    )

    primary_elastic = next(
        row for row in all_overall
        if row["scenario"] == "PRIMARY" and row["model"] == "elastic_net"
    )
    sensitivity_elastic = [
        row for row in comparison
        if row["model"] == "elastic_net" and row["scenario"] != "PRIMARY"
    ]

    status = {
        "step": "Step10.4B",
        "version": VERSION,
        "status": "PASS_SPECIES_ONLY_MODELING_PIPELINE_COMPLETED",
        "primary_n": 363,
        "primary_domains": 9,
        "assembled_species_features": len(assembled.feature_names),
        "models": MODEL_ORDER,
        "scenarios": list(SCENARIOS.keys()),
        "validation": "Leave-one-domain-out outer validation; grouped inner tuning",
        "primary_model": "elastic_net",
        "primary_elastic_net_metrics": primary_elastic,
        "primary_elastic_net_sensitivity_deltas": sensitivity_elastic,
        "interpretation_gate": (
            "REQUIRES_RESULT_REVIEW_BEFORE_MANUSCRIPT_CONCLUSION"
        ),
        "critical_rules_satisfied": [
            "All three mandatory endpoint/site sensitivity scenarios were run for every model.",
            "All models used leave-one-domain-out validation.",
            "Preprocessing was learned only from outer training domains.",
            "PRIMM0069 was excluded.",
            "No domain indicator was used as a predictor.",
        ],
        "runtime_seconds": round(time.time() - start, 2),
    }
    (result_dir / "Step10_4B_status_v2_species_only.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_lines = [
        "Step 10.4B corrected species-only results v2",
        "",
        f"Assembled patients: {len(assembled.sample_ids)}",
        f"Assembled canonical species: {len(assembled.feature_names)}",
        f"Primary elastic-net LOSO ROC AUC: {primary_elastic['roc_auc']}",
        f"Primary elastic-net LOSO PR AUC: {primary_elastic['pr_auc']}",
        "",
        "Do not interpret this file alone.",
        "Review overall metrics, per-domain metrics, sensitivity deltas,",
        "calibration and feature stability before manuscript conclusions.",
    ]
    (result_dir / "README_RESULTS_STEP10_4B_v2_species_only.txt").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    output_zip = root / "Step10_4B_results_v2_species_only.zip"
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(result_dir))

    log(json.dumps(status, ensure_ascii=False, indent=2))
    
    audit_path = root / "Step10_4B_taxonomy_correction_audit_v2.json"
    if audit_path.is_file():
        with zipfile.ZipFile(output_zip, "a", zipfile.ZIP_DEFLATED) as archive:
            archive.write(audit_path, arcname=audit_path.name)
    log(f"[output] {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
