#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse the isolated Step 10.4B Python environment before importing scientific packages.
ROOT_HINT = Path(__file__).resolve().parent
LOCAL_ENV = ROOT_HINT / ".step10_4b_pydeps_cp312"
if LOCAL_ENV.is_dir():
    sys.path.insert(0, str(LOCAL_ENV))

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold, StratifiedKFold
except Exception as exc:
    raise SystemExit(
        "Missing NumPy/SciPy/scikit-learn. Keep the existing "
        ".step10_4b_pydeps_cp312 folder in the project root.\n"
        f"Original import error: {exc}"
    )

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VERSION = "v1"
RANDOM_SEED = 20260720
MIN_PREVALENCE = 0.10
FALLBACK_PREVALENCE = 0.05
MAX_FEATURES = 500
DEFAULT_RANDOM_REPEATS = 5
DEFAULT_PERMUTATIONS = 500

SCENARIOS = {
    "PRIMARY": "Step10_2_manifest_R3_primary_lock_v8.tsv",
    "S1_EXCLUDE_BCN12": "Step10_2_manifest_R3_sensitivity_S1_exclude_BCN12_v8.tsv",
    "S2_BCN12_AS_NR": "Step10_2_manifest_R3_sensitivity_S2_BCN12_as_NR_v8.tsv",
    "S3_EXCLUDE_LEE_SITES": "Step10_2_manifest_R3_sensitivity_S3_exclude_Lee_sites_v8.tsv",
}
MODEL_ORDER = ["elastic_net", "ridge", "random_forest"]

# Step 10.5A is a validation-design benchmark, not another hyperparameter search.
# Model complexity is locked a priori within the Step 10.4B grid so that random CV,
# LODO and study-family-out are compared with exactly the same model specification.
LOCKED_PARAMS = {
    "elastic_net": {"C": 0.2, "l1_ratio": 0.75},
    "ridge": {"C": 0.5},
    "random_forest": {"max_features": "sqrt", "min_samples_leaf": 5},
}

FAMILY_MAP = {
    "Frankel_US": "Frankel_2017",
    "Gopalakrishnan_US": "Gopalakrishnan_2018",
    "Matson_US": "Matson_2018",
    "Spencer_US": "Spencer_2021",
    "Lee_PRIMM_UK": "Lee_2022_family",
    "Lee_PRIMM_NL": "Lee_2022_family",
    "Lee_Manchester": "Lee_2022_family",
    "Lee_Leeds": "Lee_2022_family",
    "Lee_Barcelona": "Lee_2022_family",
}


def log(message: str) -> None:
    print(message, flush=True)


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def yes(value: object) -> bool:
    return text(value).lower() in {"yes", "y", "true", "1"}


def safe_float(value: object) -> float:
    try:
        value_float = float(text(value))
        return value_float if math.isfinite(value_float) else 0.0
    except Exception:
        return 0.0


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [{key: text(value) for key, value in row.items()} for row in reader]
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
        root / "02_results_step10_4B_v2_species_only" / filename,
        root / "02_results_step10_4B_v1" / filename,
    ]
    for path in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path
    candidates = [
        path for path in root.rglob(filename)
        if path.is_file() and path.stat().st_size > 0
        and "02_results_step10_5A" not in str(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"Required file not found: {filename}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p).lower()))[0]


def open_assembled_matrix(root: Path):
    direct_candidates = [
        root / "assembled_species_matrix.tsv",
        root / "02_results_step10_4B_v2_species_only" / "assembled_species_matrix.tsv",
    ]
    for path in direct_candidates:
        if path.is_file():
            return path.open("r", encoding="utf-8-sig", newline=""), str(path)

    zip_path = locate_file(root, "Step10_4B_results_v2_species_only.zip")
    archive = zipfile.ZipFile(zip_path, "r")
    if "assembled_species_matrix.tsv" not in archive.namelist():
        archive.close()
        raise RuntimeError(
            "Step10_4B_results_v2_species_only.zip does not contain "
            "assembled_species_matrix.tsv"
        )
    binary = archive.open("assembled_species_matrix.tsv", "r")
    import io
    text_handle = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    # Keep archive attached so it stays open until the text handle is closed.
    text_handle._source_archive = archive  # type: ignore[attr-defined]
    return text_handle, f"{zip_path}!assembled_species_matrix.tsv"


@dataclass
class DataSet:
    manifest_ids: List[str]
    patient_ids: List[str]
    domains: np.ndarray
    families: np.ndarray
    primary_y: np.ndarray
    feature_names: List[str]
    X: np.ndarray


def load_data(root: Path) -> DataSet:
    handle, source = open_assembled_matrix(root)
    try:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:4] != [
            "manifest_id", "patient_id", "domain_id", "response_harmonized"
        ]:
            raise RuntimeError(
                "Unexpected assembled matrix metadata columns: " + str(header[:4])
            )
        feature_names = header[4:]
        manifest_ids: List[str] = []
        patient_ids: List[str] = []
        domains: List[str] = []
        labels: List[int] = []
        values: List[List[float]] = []
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError(
                    f"Malformed assembled matrix row with {len(row)} columns; "
                    f"expected {len(header)}"
                )
            manifest_ids.append(row[0])
            patient_ids.append(row[1])
            domain = row[2]
            domains.append(domain)
            response = row[3]
            if response not in {"Responder", "Non-responder"}:
                raise RuntimeError(f"Invalid primary response: {response}")
            labels.append(1 if response == "Responder" else 0)
            values.append([safe_float(value) for value in row[4:]])
    finally:
        handle.close()

    if len(manifest_ids) != 363:
        raise RuntimeError(f"Expected 363 assembled patients; observed {len(manifest_ids)}")
    if len(feature_names) != 2205:
        raise RuntimeError(f"Expected 2205 species; observed {len(feature_names)}")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise RuntimeError("Duplicate manifest IDs in assembled matrix")
    unknown_domains = sorted(set(domains) - set(FAMILY_MAP))
    if unknown_domains:
        raise RuntimeError(f"Unmapped study domains: {unknown_domains}")

    X = np.asarray(values, dtype=float)
    row_sums = X.sum(axis=1)
    if np.any(row_sums <= 0):
        raise RuntimeError("At least one sample has zero species abundance")
    X = X / row_sums[:, None]
    domain_array = np.asarray(domains, dtype=object)
    family_array = np.asarray([FAMILY_MAP[d] for d in domains], dtype=object)
    log(
        f"[data] source={source}; n={len(manifest_ids)}; species={len(feature_names)}; "
        f"domains={len(np.unique(domain_array))}; families={len(np.unique(family_array))}"
    )
    return DataSet(
        manifest_ids=manifest_ids,
        patient_ids=patient_ids,
        domains=domain_array,
        families=family_array,
        primary_y=np.asarray(labels, dtype=int),
        feature_names=feature_names,
        X=X,
    )


def load_scenario(
    root: Path,
    data: DataSet,
    scenario: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, str]]]:
    manifest_path = locate_file(root, SCENARIOS[scenario])
    rows, _ = read_tsv(manifest_path)
    row_by_id = {row["manifest_id"]: row for row in rows}
    indices: List[int] = []
    labels: List[int] = []
    selected_rows: Dict[str, Dict[str, str]] = {}
    for index, manifest_id in enumerate(data.manifest_ids):
        row = row_by_id.get(manifest_id)
        if row is None:
            continue
        include = (
            yes(row.get("strict_primary_target", ""))
            and row.get("profile_mapping_status", "") == "MATCHED"
            and yes(row.get("r3_v8_primary_analysis_include", ""))
        )
        if not include:
            continue
        response = row.get("response_harmonized", "")
        if response not in {"Responder", "Non-responder"}:
            raise RuntimeError(f"Invalid scenario label for {manifest_id}: {response}")
        indices.append(index)
        labels.append(1 if response == "Responder" else 0)
        selected_rows[manifest_id] = row
    return np.asarray(indices, dtype=int), np.asarray(labels, dtype=int), selected_rows


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
        raise RuntimeError("No species passed prevalence filtering")

    nonzero = X_train[:, selected][X_train[:, selected] > 0]
    pseudocount = float(np.min(nonzero) / 2.0) if nonzero.size else 1e-6
    pseudocount = min(max(pseudocount, 1e-8), 1e-4)
    clr = np.log(X_train[:, selected] + pseudocount)
    clr -= clr.mean(axis=1, keepdims=True)
    variances = np.var(clr, axis=0)
    if len(selected) > MAX_FEATURES:
        order = np.argsort(variances)[::-1][:MAX_FEATURES]
        selected = selected[order]
        clr = np.log(X_train[:, selected] + pseudocount)
        clr -= clr.mean(axis=1, keepdims=True)
    mean = clr.mean(axis=0)
    scale = clr.std(axis=0, ddof=0)
    scale[scale < 1e-10] = 1.0
    return PreprocessState(selected, pseudocount, mean, scale)


def apply_preprocessor(X: np.ndarray, state: PreprocessState) -> np.ndarray:
    clr = np.log(X[:, state.selected_indices] + state.pseudocount)
    clr -= clr.mean(axis=1, keepdims=True)
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
            tol=1e-3,
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
            n_estimators=100,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = (p >= 0.5).astype(int)
    auc = safe_auc(y, p)
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    brier = float(brier_score_loss(y, p))
    balanced = float(balanced_accuracy_score(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    loss = float(log_loss(y, clipped, labels=[0, 1]))
    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "log_loss": loss,
    }


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_mode: str,
    inner_groups: Optional[np.ndarray],
    seed: int,
) -> Tuple[Dict[str, object], float]:
    if inner_mode == "stratified":
        minimum_class = int(np.min(np.bincount(y_train)))
        n_splits = max(2, min(4, minimum_class))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(X_train, y_train)
    else:
        if inner_groups is None:
            raise ValueError("Grouped tuning requires inner_groups")
        unique_groups = np.unique(inner_groups)
        n_splits = min(4, len(unique_groups))
        if n_splits < 2:
            return parameter_grid(model_name)[0], float("nan")
        splitter = GroupKFold(n_splits=n_splits)
        split_iterator = splitter.split(X_train, y_train, inner_groups)

    splits = list(split_iterator)
    best_params: Optional[Dict[str, object]] = None
    best_score = -np.inf
    for params in parameter_grid(model_name):
        scores: List[float] = []
        for inner_fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            if len(np.unique(y_train[fit_idx])) < 2 or len(np.unique(y_train[val_idx])) < 2:
                continue
            model = build_model(model_name, params, seed + inner_fold)
            model.fit(X_train[fit_idx], y_train[fit_idx])
            probability = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(float(roc_auc_score(y_train[val_idx], probability)))
        score = float(np.mean(scores)) if scores else -np.inf
        if score > best_score:
            best_score = score
            best_params = params
    if best_params is None:
        best_params = parameter_grid(model_name)[0]
    return best_params, float(best_score)


@dataclass
class FoldCache:
    scheme: str
    fold_id: str
    test_group: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    best_params: Dict[str, object]
    selected_features: int
    pseudocount: float


def build_outer_splits(
    scheme: str,
    y: np.ndarray,
    domains: np.ndarray,
    families: np.ndarray,
    repeat_id: int,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    splits: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if scheme == "RANDOM_STRATIFIED_5FOLD":
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_SEED + repeat_id * 1009,
        )
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
            splits.append((f"R{repeat_id}_F{fold_index}", train_idx, test_idx))
    elif scheme == "LEAVE_ONE_DOMAIN_OUT":
        for domain in sorted(np.unique(domains)):
            test_idx = np.where(domains == domain)[0]
            train_idx = np.where(domains != domain)[0]
            splits.append((domain, train_idx, test_idx))
    elif scheme == "LEAVE_ONE_STUDY_FAMILY_OUT":
        for family in sorted(np.unique(families)):
            test_idx = np.where(families == family)[0]
            train_idx = np.where(families != family)[0]
            splits.append((family, train_idx, test_idx))
    else:
        raise ValueError(scheme)
    return splits


def run_observed_validation(
    scenario: str,
    model_name: str,
    scheme: str,
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    families: np.ndarray,
    manifest_ids: Sequence[str],
    patient_ids: Sequence[str],
    repeat_id: int,
    keep_cache: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], List[FoldCache]]:
    outer_splits = build_outer_splits(scheme, y, domains, families, repeat_id)
    predictions: List[Dict[str, object]] = []
    fold_metrics: List[Dict[str, object]] = []
    caches: List[FoldCache] = []

    for fold_number, (fold_id, train_idx, test_idx) in enumerate(outer_splits, start=1):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"One-class training set in {scenario}/{model_name}/{scheme}/{fold_id}")

        state = fit_preprocessor(X_train_raw)
        X_train = apply_preprocessor(X_train_raw, state)
        X_test = apply_preprocessor(X_test_raw, state)

        params = dict(LOCKED_PARAMS[model_name])
        inner_auc = float("nan")
        model = build_model(
            model_name,
            params,
            RANDOM_SEED + repeat_id * 100000 + fold_number * 1000,
        )
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        metrics = metric_dict(y_test, probability)
        test_group = (
            fold_id if scheme != "RANDOM_STRATIFIED_5FOLD" else "mixed_domains"
        )
        fold_metrics.append({
            "scenario": scenario,
            "model": model_name,
            "validation_scheme": scheme,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "test_group": test_group,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "test_responders": int(y_test.sum()),
            "test_non_responders": int(len(y_test) - y_test.sum()),
            "selected_features": len(state.selected_indices),
            "pseudocount": state.pseudocount,
            "best_params": json.dumps(params, sort_keys=True),
            "inner_cv_auc": inner_auc,
            **metrics,
        })
        for local_position, probability_value in zip(test_idx, probability):
            predictions.append({
                "scenario": scenario,
                "model": model_name,
                "validation_scheme": scheme,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "manifest_id": manifest_ids[local_position],
                "patient_id": patient_ids[local_position],
                "domain_id": domains[local_position],
                "study_family": families[local_position],
                "true_label": int(y[local_position]),
                "predicted_probability": float(probability_value),
                "predicted_class": int(probability_value >= 0.5),
            })
        if keep_cache:
            caches.append(FoldCache(
                scheme=scheme,
                fold_id=fold_id,
                test_group=test_group,
                train_idx=train_idx.copy(),
                test_idx=test_idx.copy(),
                X_train=X_train,
                X_test=X_test,
                best_params=dict(params),
                selected_features=len(state.selected_indices),
                pseudocount=state.pseudocount,
            ))

    prediction_by_index: Dict[int, float] = {}
    for row in predictions:
        index = manifest_ids.index(text(row["manifest_id"]))
        prediction_by_index[index] = float(row["predicted_probability"])
    ordered_prob = np.asarray([prediction_by_index[i] for i in range(len(y))], dtype=float)
    overall = {
        "scenario": scenario,
        "model": model_name,
        "validation_scheme": scheme,
        "repeat_id": repeat_id,
        "n": len(y),
        "responders": int(y.sum()),
        "non_responders": int(len(y) - y.sum()),
        "domains": len(np.unique(domains)),
        "study_families": len(np.unique(families)),
        **metric_dict(y, ordered_prob),
        "prevalence_baseline_pr_auc": float(y.mean()),
    }
    return predictions, fold_metrics, overall, caches


def aggregate_benchmark(
    overall_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    keys = sorted({
        (text(row["scenario"]), text(row["model"]), text(row["validation_scheme"]))
        for row in overall_rows
    })
    metrics = [
        "roc_auc", "pr_auc", "brier", "balanced_accuracy",
        "sensitivity", "specificity", "log_loss",
    ]
    for scenario, model, scheme in keys:
        rows = [row for row in overall_rows if (
            row["scenario"] == scenario and row["model"] == model
            and row["validation_scheme"] == scheme
        )]
        base = rows[0]
        summary: Dict[str, object] = {
            "scenario": scenario,
            "model": model,
            "validation_scheme": scheme,
            "runs": len(rows),
            "n": base["n"],
            "responders": base["responders"],
            "non_responders": base["non_responders"],
            "domains": base["domains"],
            "study_families": base["study_families"],
            "prevalence_baseline_pr_auc": base["prevalence_baseline_pr_auc"],
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            summary[metric] = float(np.nanmean(values))
            summary[f"{metric}_sd_across_runs"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = float(np.nanmin(values))
            summary[f"{metric}_max"] = float(np.nanmax(values))
        output.append(summary)
    return output


def calculate_optimism_gaps(
    benchmark_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    lookup = {
        (text(row["scenario"]), text(row["model"]), text(row["validation_scheme"])): row
        for row in benchmark_rows
    }
    output: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        for model in MODEL_ORDER:
            random_row = lookup[(scenario, model, "RANDOM_STRATIFIED_5FOLD")]
            lodo_row = lookup[(scenario, model, "LEAVE_ONE_DOMAIN_OUT")]
            losfo_row = lookup[(scenario, model, "LEAVE_ONE_STUDY_FAMILY_OUT")]
            output.append({
                "scenario": scenario,
                "model": model,
                "random_cv_auc": random_row["roc_auc"],
                "lodo_auc": lodo_row["roc_auc"],
                "losfo_auc": losfo_row["roc_auc"],
                "random_minus_lodo_auc": float(random_row["roc_auc"]) - float(lodo_row["roc_auc"]),
                "random_minus_losfo_auc": float(random_row["roc_auc"]) - float(losfo_row["roc_auc"]),
                "lodo_minus_losfo_auc": float(lodo_row["roc_auc"]) - float(losfo_row["roc_auc"]),
                "random_cv_pr_auc": random_row["pr_auc"],
                "lodo_pr_auc": lodo_row["pr_auc"],
                "losfo_pr_auc": losfo_row["pr_auc"],
                "random_minus_losfo_pr_auc": float(random_row["pr_auc"]) - float(losfo_row["pr_auc"]),
                "random_cv_brier": random_row["brier"],
                "lodo_brier": lodo_row["brier"],
                "losfo_brier": losfo_row["brier"],
            })
    return output


def permute_within_domains(y: np.ndarray, domains: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = y.copy()
    for domain in np.unique(domains):
        indices = np.where(domains == domain)[0]
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def predict_locked_permutation(
    y_permuted: np.ndarray,
    caches: Sequence[FoldCache],
    seed_base: int,
) -> np.ndarray:
    probability = np.full(len(y_permuted), np.nan, dtype=float)
    for fold_number, cache in enumerate(caches, start=1):
        y_train = y_permuted[cache.train_idx]
        if len(np.unique(y_train)) < 2:
            probability[cache.test_idx] = float(np.mean(y_train))
            continue
        model = build_model(
            "elastic_net",
            cache.best_params,
            seed_base + fold_number * 1000,
        )
        model.fit(cache.X_train, y_train)
        probability[cache.test_idx] = model.predict_proba(cache.X_test)[:, 1]
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("Incomplete probabilities in locked permutation")
    return probability


def run_permutations(
    result_dir: Path,
    y: np.ndarray,
    domains: np.ndarray,
    lodo_caches: Sequence[FoldCache],
    losfo_caches: Sequence[FoldCache],
    observed_lodo_auc: float,
    observed_losfo_auc: float,
    permutations: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    checkpoint_path = result_dir / "permutation_checkpoint.tsv"
    rows: List[Dict[str, object]] = []
    completed = set()
    if checkpoint_path.is_file():
        existing, _ = read_tsv(checkpoint_path)
        for row in existing:
            index = int(row["permutation_index"])
            if index <= permutations:
                rows.append({
                    "permutation_index": index,
                    "lodo_roc_auc": float(row["lodo_roc_auc"]),
                    "losfo_roc_auc": float(row["losfo_roc_auc"]),
                })
                completed.add(index)
        if completed:
            log(f"[permutation] resuming from {len(completed)} completed iterations")

    for permutation_index in range(1, permutations + 1):
        if permutation_index in completed:
            continue
        rng = np.random.default_rng(RANDOM_SEED + permutation_index * 7919)
        y_permuted = permute_within_domains(y, domains, rng)
        lodo_probability = predict_locked_permutation(
            y_permuted,
            lodo_caches,
            RANDOM_SEED + permutation_index * 100000,
        )
        losfo_probability = predict_locked_permutation(
            y_permuted,
            losfo_caches,
            RANDOM_SEED + permutation_index * 200000,
        )
        rows.append({
            "permutation_index": permutation_index,
            "lodo_roc_auc": safe_auc(y_permuted, lodo_probability),
            "losfo_roc_auc": safe_auc(y_permuted, losfo_probability),
        })
        if permutation_index % 10 == 0 or permutation_index == permutations:
            rows = sorted(rows, key=lambda row: int(row["permutation_index"]))
            write_tsv(
                checkpoint_path,
                rows,
                ["permutation_index", "lodo_roc_auc", "losfo_roc_auc"],
            )
            log(f"[permutation] completed {permutation_index}/{permutations}")

    rows = sorted(rows, key=lambda row: int(row["permutation_index"]))
    output: List[Dict[str, object]] = []
    for scheme, field, observed in [
        ("LEAVE_ONE_DOMAIN_OUT", "lodo_roc_auc", observed_lodo_auc),
        ("LEAVE_ONE_STUDY_FAMILY_OUT", "losfo_roc_auc", observed_losfo_auc),
    ]:
        null_values = np.asarray([float(row[field]) for row in rows], dtype=float)
        output.append({
            "scenario": "PRIMARY",
            "model": "elastic_net",
            "validation_scheme": scheme,
            "permutation_strategy": "response labels shuffled separately within each of 9 domains",
            "hyperparameter_strategy": "prespecified locked Step10.5A benchmark hyperparameters",
            "permutations": len(null_values),
            "observed_roc_auc": observed,
            "null_mean_roc_auc": float(np.mean(null_values)),
            "null_sd_roc_auc": float(np.std(null_values, ddof=1)),
            "null_median_roc_auc": float(np.median(null_values)),
            "null_95th_percentile_roc_auc": float(np.quantile(null_values, 0.95)),
            "null_975th_percentile_roc_auc": float(np.quantile(null_values, 0.975)),
            "empirical_one_sided_p": float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1)),
        })
    return rows, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--support-random-repeats", type=int, default=2)
    parser.add_argument("--skip-permutations", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    root = Path(args.root.strip().strip('"').rstrip("\\/")).resolve()
    result_dir = root / "02_results_step10_5A_v1"
    result_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    random_repeats = 1 if args.smoke_test else max(1, args.random_repeats)
    permutations = 2 if args.smoke_test else max(0, args.permutations)
    support_random_repeats = 1 if args.smoke_test else max(1, args.support_random_repeats)
    active_models = ["elastic_net"] if args.smoke_test else MODEL_ORDER
    active_scenarios = ["PRIMARY"] if args.smoke_test else list(SCENARIOS)

    log("[start] Step 10.5A study-family-aware validation and structured permutation")
    data = load_data(root)

    family_rows = []
    for domain in sorted(FAMILY_MAP):
        mask = data.domains == domain
        family_rows.append({
            "domain_id": domain,
            "study_family": FAMILY_MAP[domain],
            "primary_n": int(mask.sum()),
            "primary_responders": int(data.primary_y[mask].sum()),
            "primary_non_responders": int(mask.sum() - data.primary_y[mask].sum()),
        })
    write_tsv(
        result_dir / "study_family_mapping.tsv",
        family_rows,
        ["domain_id", "study_family", "primary_n", "primary_responders", "primary_non_responders"],
    )

    all_predictions: List[Dict[str, object]] = []
    all_fold_metrics: List[Dict[str, object]] = []
    all_overall: List[Dict[str, object]] = []
    primary_lodo_caches: List[FoldCache] = []
    primary_losfo_caches: List[FoldCache] = []

    for scenario in active_scenarios:
        selected_indices, scenario_y, _ = load_scenario(root, data, scenario)
        X = data.X[selected_indices]
        domains = data.domains[selected_indices]
        families = data.families[selected_indices]
        manifest_ids = [data.manifest_ids[i] for i in selected_indices]
        patient_ids = [data.patient_ids[i] for i in selected_indices]
        log(
            f"[scenario] {scenario}: n={len(scenario_y)}; R={int(scenario_y.sum())}; "
            f"NR={int(len(scenario_y)-scenario_y.sum())}; domains={len(np.unique(domains))}; "
            f"families={len(np.unique(families))}"
        )

        for model_name in active_models:
            # Random patient-level CV intentionally allows study domains in both train and test.
            model_random_repeats = random_repeats if model_name == "elastic_net" else support_random_repeats
            for repeat_id in range(1, model_random_repeats + 1):
                predictions, folds, overall, _ = run_observed_validation(
                    scenario, model_name, "RANDOM_STRATIFIED_5FOLD",
                    X, scenario_y, domains, families, manifest_ids, patient_ids,
                    repeat_id=repeat_id, keep_cache=False,
                )
                all_predictions.extend(predictions)
                all_fold_metrics.extend(folds)
                all_overall.append(overall)
                log(
                    f"[complete] {scenario}/{model_name}/random repeat {repeat_id}: "
                    f"AUC={overall['roc_auc']:.4f}"
                )

            for scheme in ["LEAVE_ONE_DOMAIN_OUT", "LEAVE_ONE_STUDY_FAMILY_OUT"]:
                keep_cache = (
                    scenario == "PRIMARY" and model_name == "elastic_net"
                    and scheme in {"LEAVE_ONE_DOMAIN_OUT", "LEAVE_ONE_STUDY_FAMILY_OUT"}
                )
                predictions, folds, overall, caches = run_observed_validation(
                    scenario, model_name, scheme,
                    X, scenario_y, domains, families, manifest_ids, patient_ids,
                    repeat_id=1, keep_cache=keep_cache,
                )
                all_predictions.extend(predictions)
                all_fold_metrics.extend(folds)
                all_overall.append(overall)
                if keep_cache and scheme == "LEAVE_ONE_DOMAIN_OUT":
                    primary_lodo_caches = caches
                if keep_cache and scheme == "LEAVE_ONE_STUDY_FAMILY_OUT":
                    primary_losfo_caches = caches
                log(
                    f"[complete] {scenario}/{model_name}/{scheme}: "
                    f"AUC={overall['roc_auc']:.4f}"
                )

    benchmark = aggregate_benchmark(all_overall)
    optimism = calculate_optimism_gaps(benchmark) if not args.smoke_test else []

    write_tsv(
        result_dir / "observed_validation_run_metrics.tsv",
        all_overall,
        [
            "scenario", "model", "validation_scheme", "repeat_id", "n",
            "responders", "non_responders", "domains", "study_families",
            "roc_auc", "pr_auc", "brier", "balanced_accuracy", "sensitivity",
            "specificity", "log_loss", "prevalence_baseline_pr_auc",
        ],
    )
    benchmark_fields = [
        "scenario", "model", "validation_scheme", "runs", "n", "responders",
        "non_responders", "domains", "study_families", "roc_auc",
        "roc_auc_sd_across_runs", "roc_auc_min", "roc_auc_max", "pr_auc",
        "pr_auc_sd_across_runs", "pr_auc_min", "pr_auc_max", "brier",
        "brier_sd_across_runs", "brier_min", "brier_max", "balanced_accuracy",
        "balanced_accuracy_sd_across_runs", "balanced_accuracy_min",
        "balanced_accuracy_max", "sensitivity", "sensitivity_sd_across_runs",
        "sensitivity_min", "sensitivity_max", "specificity",
        "specificity_sd_across_runs", "specificity_min", "specificity_max",
        "log_loss", "log_loss_sd_across_runs", "log_loss_min", "log_loss_max",
        "prevalence_baseline_pr_auc",
    ]
    write_tsv(result_dir / "validation_scheme_benchmark_summary.tsv", benchmark, benchmark_fields)
    write_tsv(
        result_dir / "validation_scheme_fold_metrics.tsv",
        all_fold_metrics,
        [
            "scenario", "model", "validation_scheme", "repeat_id", "fold_id",
            "test_group", "train_n", "test_n", "test_responders",
            "test_non_responders", "selected_features", "pseudocount",
            "best_params", "inner_cv_auc", "roc_auc", "pr_auc", "brier",
            "balanced_accuracy", "sensitivity", "specificity", "log_loss",
        ],
    )
    write_tsv(
        result_dir / "all_validation_predictions.tsv",
        all_predictions,
        [
            "scenario", "model", "validation_scheme", "repeat_id", "fold_id",
            "manifest_id", "patient_id", "domain_id", "study_family",
            "true_label", "predicted_probability", "predicted_class",
        ],
    )
    if optimism:
        write_tsv(
            result_dir / "random_CV_external_validation_optimism_gap.tsv",
            optimism,
            [
                "scenario", "model", "random_cv_auc", "lodo_auc", "losfo_auc",
                "random_minus_lodo_auc", "random_minus_losfo_auc",
                "lodo_minus_losfo_auc", "random_cv_pr_auc", "lodo_pr_auc",
                "losfo_pr_auc", "random_minus_losfo_pr_auc", "random_cv_brier",
                "lodo_brier", "losfo_brier",
            ],
        )

    permutation_rows: List[Dict[str, object]] = []
    empirical_rows: List[Dict[str, object]] = []
    if not args.skip_permutations and permutations > 0:
        primary_indices, primary_y, _ = load_scenario(root, data, "PRIMARY")
        primary_domains = data.domains[primary_indices]
        lookup = {
            (row["scenario"], row["model"], row["validation_scheme"]): row
            for row in benchmark
        }
        observed_lodo_auc = float(lookup[("PRIMARY", "elastic_net", "LEAVE_ONE_DOMAIN_OUT")]["roc_auc"])
        observed_losfo_auc = float(lookup[("PRIMARY", "elastic_net", "LEAVE_ONE_STUDY_FAMILY_OUT")]["roc_auc"])
        permutation_rows, empirical_rows = run_permutations(
            result_dir,
            primary_y,
            primary_domains,
            primary_lodo_caches,
            primary_losfo_caches,
            observed_lodo_auc,
            observed_losfo_auc,
            permutations,
        )
        write_tsv(
            result_dir / "within_domain_permutation_null_distribution.tsv",
            permutation_rows,
            ["permutation_index", "lodo_roc_auc", "losfo_roc_auc"],
        )
        write_tsv(
            result_dir / "within_domain_permutation_empirical_pvalues.tsv",
            empirical_rows,
            [
                "scenario", "model", "validation_scheme", "permutation_strategy",
                "hyperparameter_strategy", "permutations", "observed_roc_auc",
                "null_mean_roc_auc", "null_sd_roc_auc", "null_median_roc_auc",
                "null_95th_percentile_roc_auc", "null_975th_percentile_roc_auc",
                "empirical_one_sided_p",
            ],
        )

    status = {
        "step": "Step10.5A",
        "version": VERSION,
        "status": "PASS_STEP10_5A_COMPLETED",
        "input": "Step10_4B_results_v2_species_only.zip / assembled_species_matrix.tsv",
        "primary_n": 363,
        "species_features": 2205,
        "domain_count": 9,
        "study_family_count": 5,
        "observed_models": active_models,
        "observed_scenarios": active_scenarios,
        "validation_schemes": [
            "RANDOM_STRATIFIED_5FOLD",
            "LEAVE_ONE_DOMAIN_OUT",
            "LEAVE_ONE_STUDY_FAMILY_OUT",
        ],
        "primary_elastic_net_random_cv_repeats": random_repeats,
        "support_model_random_cv_repeats": support_random_repeats,
        "locked_benchmark_hyperparameters": LOCKED_PARAMS,
        "permutation_primary_model_only": "elastic_net",
        "permutation_primary_scenario_only": "PRIMARY",
        "permutation_iterations": len(permutation_rows),
        "permutation_strategy": "shuffle response separately within each domain",
        "mandatory_sensitivity_rule": (
            "All three observed models were rerun under PRIMARY, S1, S2 and S3 "
            "for all three validation schemes. The permutation test targets the "
            "prespecified PRIMARY elastic-net estimand only."
        ),
        "leakage_controls": [
            "Feature filtering, CLR pseudocount and scaling learned only in each outer training set.",
            "Random CV intentionally mixes domains to quantify conventional-validation optimism.",
            "No response information from an outer test fold contributes to preprocessing or fitting.",
            "The same prespecified model complexity is used for random CV, LODO and study-family-out.",
            "Permutation preserves domain-specific sample size and responder prevalence.",
        ],
        "runtime_seconds": round(time.time() - start_time, 2),
    }
    (result_dir / "Step10_5A_status_v1.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = [
        "Step 10.5A results v1",
        "",
        "Primary outputs:",
        "- validation_scheme_benchmark_summary.tsv",
        "- random_CV_external_validation_optimism_gap.tsv",
        "- validation_scheme_fold_metrics.tsv",
        "- within_domain_permutation_empirical_pvalues.tsv",
        "- within_domain_permutation_null_distribution.tsv",
        "",
        "Scientific interpretation must compare random CV, LODO and study-family-out.",
        "The permutation test is a prespecified locked-complexity benchmark for elastic-net.",
    ]
    (result_dir / "README_RESULTS_STEP10_5A_v1.txt").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )

    output_zip = root / "Step10_5A_results_v1.zip"
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(result_dir))
    log(json.dumps(status, ensure_ascii=False, indent=2))
    log(f"[output] {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
