#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_HINT = Path(__file__).resolve().parent
LOCAL_ENV = ROOT_HINT / ".step10_4b_pydeps_cp312"
if LOCAL_ENV.is_dir():
    sys.path.insert(0, str(LOCAL_ENV))

try:
    import numpy as np
    from scipy.stats import rankdata, spearmanr
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

SCENARIOS = {
    "PRIMARY": "Step10_2_manifest_R3_primary_lock_v8.tsv",
    "S1_EXCLUDE_BCN12": "Step10_2_manifest_R3_sensitivity_S1_exclude_BCN12_v8.tsv",
    "S2_BCN12_AS_NR": "Step10_2_manifest_R3_sensitivity_S2_BCN12_as_NR_v8.tsv",
    "S3_EXCLUDE_LEE_SITES": "Step10_2_manifest_R3_sensitivity_S3_exclude_Lee_sites_v8.tsv",
}

MODEL_ORDER = ["elastic_net", "ridge", "random_forest"]
LOCKED_PARAMS = {
    "elastic_net": {"C": 0.2, "l1_ratio": 0.75},
    "ridge": {"C": 0.5},
    "random_forest": {"max_features": "sqrt", "min_samples_leaf": 5},
}

DISTANCE_COLUMNS = [
    "aitchison_centroid_distance",
    "bray_curtis_centroid_distance",
    "jensen_shannon_distance",
    "prevalence_l1_distance",
]

BINARY_RELATIONS = [
    "same_study_family",
    "same_country",
    "same_macro_region",
    "same_treatment_scope_group",
]

PRIMARY_MRQAP_PREDICTORS = [
    "aitchison_centroid_distance",
    "same_study_family",
    "same_country",
    "same_treatment_scope_group",
    "absolute_response_rate_difference",
    "log_train_n",
    "log_test_n",
]


def log(message: str) -> None:
    print(message, flush=True)


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def yes(value: object) -> bool:
    return text(value).lower() in {"yes", "y", "true", "1"}


def safe_float(value: object) -> float:
    try:
        x = float(text(value))
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


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
    ]
    for path in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path
    candidates = [
        path for path in root.rglob(filename)
        if path.is_file() and path.stat().st_size > 0
        and "02_results_step10_5B" not in str(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"Required file not found: {filename}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p).lower()))[0]


def open_assembled_matrix(root: Path):
    direct = root / "02_results_step10_4B_v2_species_only" / "assembled_species_matrix.tsv"
    if direct.is_file():
        return direct.open("r", encoding="utf-8-sig", newline=""), str(direct)

    zip_path = locate_file(root, "Step10_4B_results_v2_species_only.zip")
    archive = zipfile.ZipFile(zip_path, "r")
    if "assembled_species_matrix.tsv" not in archive.namelist():
        archive.close()
        raise RuntimeError("assembled_species_matrix.tsv absent from Step10_4B v2 ZIP")
    binary = archive.open("assembled_species_matrix.tsv", "r")
    handle = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    handle._source_archive = archive  # type: ignore[attr-defined]
    return handle, f"{zip_path}!assembled_species_matrix.tsv"


@dataclass
class DataSet:
    manifest_ids: List[str]
    patient_ids: List[str]
    domains: np.ndarray
    primary_y: np.ndarray
    feature_names: List[str]
    X: np.ndarray


def load_data(root: Path) -> DataSet:
    handle, source = open_assembled_matrix(root)
    try:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        expected = ["manifest_id", "patient_id", "domain_id", "response_harmonized"]
        if header[:4] != expected:
            raise RuntimeError(f"Unexpected assembled metadata columns: {header[:4]}")
        feature_names = header[4:]
        manifest_ids, patient_ids, domains, labels, values = [], [], [], [], []
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError(f"Malformed assembled row: {len(row)} vs {len(header)}")
            manifest_ids.append(row[0])
            patient_ids.append(row[1])
            domains.append(row[2])
            if row[3] not in {"Responder", "Non-responder"}:
                raise RuntimeError(f"Invalid response label: {row[3]}")
            labels.append(1 if row[3] == "Responder" else 0)
            values.append([float(value) for value in row[4:]])
    finally:
        handle.close()

    if len(manifest_ids) != 363:
        raise RuntimeError(f"Expected 363 primary patients; observed {len(manifest_ids)}")
    if len(feature_names) != 2205:
        raise RuntimeError(f"Expected 2205 species; observed {len(feature_names)}")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise RuntimeError("Duplicate manifest IDs")

    X = np.asarray(values, dtype=float)
    X[X < 0] = 0
    row_sums = X.sum(axis=1)
    if np.any(row_sums <= 0):
        raise RuntimeError("Zero-sum abundance sample")
    X /= row_sums[:, None]
    log(
        f"[data] source={source}; n={len(manifest_ids)}; "
        f"species={len(feature_names)}; domains={len(set(domains))}"
    )
    return DataSet(
        manifest_ids=manifest_ids,
        patient_ids=patient_ids,
        domains=np.asarray(domains, dtype=object),
        primary_y=np.asarray(labels, dtype=int),
        feature_names=feature_names,
        X=X,
    )


def load_scenario(root: Path, data: DataSet, scenario: str):
    path = locate_file(root, SCENARIOS[scenario])
    rows, _ = read_tsv(path)
    row_by_id = {row["manifest_id"]: row for row in rows}
    indices, labels = [], []
    for i, manifest_id in enumerate(data.manifest_ids):
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
            raise RuntimeError(f"Invalid scenario response: {manifest_id} {response}")
        indices.append(i)
        labels.append(1 if response == "Responder" else 0)
    return np.asarray(indices, dtype=int), np.asarray(labels, dtype=int), path


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
        raise RuntimeError("No features passed training-only prevalence filtering")

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


def build_model(model_name: str, seed: int):
    params = LOCKED_PARAMS[model_name]
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


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    brier = float(brier_score_loss(y, p))
    balanced = float(balanced_accuracy_score(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
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


def stratified_bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, reps: int, seed: int
) -> Tuple[float, float, int]:
    if reps <= 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    values: List[float] = []
    for _ in range(reps):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        values.append(float(roc_auc_score(y[idx], p[idx])))
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def read_domain_metadata(root: Path) -> Tuple[Dict[str, Dict[str, str]], Path]:
    path = locate_file(root, "Step10_5B_domain_metadata_v1.tsv")
    rows, fields = read_tsv(path)
    required = {
        "domain_id", "study_family", "country", "macro_region",
        "treatment_scope_group", "source_doi", "metadata_confidence",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"Domain metadata missing columns: {missing}")
    mapping = {row["domain_id"]: row for row in rows}
    return mapping, path


def compute_domain_ecology(
    data: DataSet,
    metadata: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    domains = sorted(np.unique(data.domains))
    missing = sorted(set(domains) - set(metadata))
    if missing:
        raise RuntimeError(f"Domain metadata absent for: {missing}")

    global_prevalence = np.mean(data.X > 0, axis=0)
    selected = np.where(global_prevalence >= 0.05)[0]
    if len(selected) < 50:
        selected = np.where(np.any(data.X > 0, axis=0))[0]
    X = data.X[:, selected]
    nonzero = X[X > 0]
    pseudocount = float(np.min(nonzero) / 2.0) if nonzero.size else 1e-6
    pseudocount = min(max(pseudocount, 1e-8), 1e-4)
    clr = np.log(X + pseudocount)
    clr -= clr.mean(axis=1, keepdims=True)

    profiles: Dict[str, np.ndarray] = {}
    clr_centroids: Dict[str, np.ndarray] = {}
    prevalence_vectors: Dict[str, np.ndarray] = {}
    stats: List[Dict[str, object]] = []

    for domain in domains:
        idx = np.where(data.domains == domain)[0]
        composition = data.X[idx].mean(axis=0)
        composition = composition / composition.sum()
        profiles[domain] = composition
        clr_centroids[domain] = clr[idx].mean(axis=0)
        prevalence_vectors[domain] = np.mean(data.X[idx] > 0, axis=0)
        stats.append({
            "domain_id": domain,
            "n": len(idx),
            "responders": int(data.primary_y[idx].sum()),
            "non_responders": int(len(idx) - data.primary_y[idx].sum()),
            "response_rate": float(data.primary_y[idx].mean()),
            "mean_detected_species_per_sample": float(np.mean(np.sum(data.X[idx] > 0, axis=1))),
            "global_ecology_features": len(selected),
            "global_ecology_pseudocount": pseudocount,
            **metadata[domain],
        })

    distances: Dict[str, Dict[str, float]] = {}
    for a in domains:
        for b in domains:
            key = f"{a}|||{b}"
            if a == b:
                distances[key] = {column: 0.0 for column in DISTANCE_COLUMNS}
                continue
            pa, pb = profiles[a], profiles[b]
            bray = float(np.sum(np.abs(pa - pb)) / np.sum(pa + pb))
            m = 0.5 * (pa + pb)
            eps = 1e-15
            kl_a = float(np.sum(pa * np.log((pa + eps) / (m + eps))))
            kl_b = float(np.sum(pb * np.log((pb + eps) / (m + eps))))
            js = math.sqrt(max(0.0, 0.5 * (kl_a + kl_b)))
            aitchison = float(
                np.linalg.norm(clr_centroids[a] - clr_centroids[b])
                / math.sqrt(len(selected))
            )
            prevalence_l1 = float(
                np.mean(np.abs(prevalence_vectors[a] - prevalence_vectors[b]))
            )
            distances[key] = {
                "aitchison_centroid_distance": aitchison,
                "bray_curtis_centroid_distance": bray,
                "jensen_shannon_distance": js,
                "prevalence_l1_distance": prevalence_l1,
            }

    vectors = {
        "global_selected_indices": selected,
        "global_clr_centroids": clr_centroids,
        "composition_centroids": profiles,
        "prevalence_vectors": prevalence_vectors,
    }
    return stats, distances, vectors


def pair_metadata(
    train_domain: str,
    test_domain: str,
    domain_stats: Dict[str, Dict[str, object]],
    metadata: Dict[str, Dict[str, str]],
    distances: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    train_meta = metadata[train_domain]
    test_meta = metadata[test_domain]
    train_stats = domain_stats[train_domain]
    test_stats = domain_stats[test_domain]

    combo_train = safe_float(train_meta.get("known_combination_ici_fraction", ""))
    combo_test = safe_float(test_meta.get("known_combination_ici_fraction", ""))
    combo_diff = (
        abs(combo_train - combo_test)
        if math.isfinite(combo_train) and math.isfinite(combo_test)
        else float("nan")
    )
    result: Dict[str, object] = {
        "train_domain": train_domain,
        "test_domain": test_domain,
        "train_study_family": train_meta["study_family"],
        "test_study_family": test_meta["study_family"],
        "train_country": train_meta["country"],
        "test_country": test_meta["country"],
        "train_macro_region": train_meta["macro_region"],
        "test_macro_region": test_meta["macro_region"],
        "train_treatment_scope_group": train_meta["treatment_scope_group"],
        "test_treatment_scope_group": test_meta["treatment_scope_group"],
        "same_study_family": int(train_meta["study_family"] == test_meta["study_family"]),
        "same_country": int(train_meta["country"] == test_meta["country"]),
        "same_macro_region": int(train_meta["macro_region"] == test_meta["macro_region"]),
        "same_treatment_scope_group": int(
            train_meta["treatment_scope_group"] == test_meta["treatment_scope_group"]
        ),
        "known_combination_ici_fraction_difference": combo_diff,
        "train_n_primary": int(train_stats["n"]),
        "test_n_primary": int(test_stats["n"]),
        "train_response_rate_primary": float(train_stats["response_rate"]),
        "test_response_rate_primary": float(test_stats["response_rate"]),
        "absolute_response_rate_difference": abs(
            float(train_stats["response_rate"]) - float(test_stats["response_rate"])
        ),
        "log_train_n": math.log(int(train_stats["n"])),
        "log_test_n": math.log(int(test_stats["n"])),
    }
    result.update(distances[f"{train_domain}|||{test_domain}"])
    return result


def run_directed_transfer(
    scenario: str,
    model_name: str,
    data: DataSet,
    indices: np.ndarray,
    y: np.ndarray,
    bootstrap_reps: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    X = data.X[indices]
    domains = data.domains[indices]
    manifest_ids = [data.manifest_ids[i] for i in indices]
    patient_ids = [data.patient_ids[i] for i in indices]
    unique_domains = sorted(np.unique(domains))
    metrics_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    pair_counter = 0

    for train_domain in unique_domains:
        train_idx = np.where(domains == train_domain)[0]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"One-class training domain: {scenario}/{train_domain}")
        state = fit_preprocessor(X[train_idx])
        X_train = apply_preprocessor(X[train_idx], state)

        # A directed transfer model is defined by its training domain.
        # Fit it once and apply the identical fitted model to every external
        # test domain. Re-fitting for each test domain would be redundant and
        # could introduce artificial pair-specific random variation.
        train_seed = (
            RANDOM_SEED
            + MODEL_ORDER.index(model_name) * 1000000
            + list(SCENARIOS).index(scenario) * 100000
            + unique_domains.index(train_domain) * 1000
        )
        model = build_model(model_name, train_seed)
        model.fit(X_train, y_train)

        for test_domain in unique_domains:
            if train_domain == test_domain:
                continue
            pair_counter += 1
            test_idx = np.where(domains == test_domain)[0]
            y_test = y[test_idx]
            if len(np.unique(y_test)) < 2:
                raise RuntimeError(f"One-class test domain: {scenario}/{test_domain}")
            X_test = apply_preprocessor(X[test_idx], state)
            probability = model.predict_proba(X_test)[:, 1]
            metrics = metric_dict(y_test, probability)
            ci_low, ci_high, ci_n = stratified_bootstrap_auc_ci(
                y_test,
                probability,
                bootstrap_reps,
                RANDOM_SEED + pair_counter * 7919 + MODEL_ORDER.index(model_name),
            )
            metrics_rows.append({
                "scenario": scenario,
                "model": model_name,
                "train_domain": train_domain,
                "test_domain": test_domain,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "train_responders": int(y_train.sum()),
                "train_non_responders": int(len(y_train) - y_train.sum()),
                "test_responders": int(y_test.sum()),
                "test_non_responders": int(len(y_test) - y_test.sum()),
                "selected_features": len(state.selected_indices),
                "pseudocount": state.pseudocount,
                "locked_params": json.dumps(LOCKED_PARAMS[model_name], sort_keys=True),
                "roc_auc_ci_low": ci_low,
                "roc_auc_ci_high": ci_high,
                "roc_auc_bootstrap_valid": ci_n,
                **metrics,
            })
            for local_i, prob in zip(test_idx, probability):
                prediction_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "manifest_id": manifest_ids[local_i],
                    "patient_id": patient_ids[local_i],
                    "true_label": int(y[local_i]),
                    "predicted_probability": float(prob),
                    "predicted_class": int(prob >= 0.5),
                })
    log(
        f"[complete] {scenario}/{model_name}: "
        f"directed_pairs={len(metrics_rows)}"
    )
    return metrics_rows, prediction_rows


def create_matrix_rows(
    pair_rows: Sequence[Dict[str, object]],
    scenario: str,
    model: str,
    value_column: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    subset = [
        row for row in pair_rows
        if row["scenario"] == scenario and row["model"] == model
    ]
    domains = sorted(
        set(text(row["train_domain"]) for row in subset)
        | set(text(row["test_domain"]) for row in subset)
    )
    lookup = {
        (text(row["train_domain"]), text(row["test_domain"])): row.get(value_column, "")
        for row in subset
    }
    rows = []
    for train_domain in domains:
        row: Dict[str, object] = {"train_domain": train_domain}
        for test_domain in domains:
            row[test_domain] = (
                "" if train_domain == test_domain
                else lookup.get((train_domain, test_domain), "")
            )
        rows.append(row)
    return rows, ["train_domain"] + domains


def reciprocal_summary(pair_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    keys = sorted(set((text(r["scenario"]), text(r["model"])) for r in pair_rows))
    for scenario, model in keys:
        subset = [r for r in pair_rows if r["scenario"] == scenario and r["model"] == model]
        lookup = {
            (text(r["train_domain"]), text(r["test_domain"])): r for r in subset
        }
        domains = sorted(set(d for pair in lookup for d in pair))
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                ab = lookup[(a, b)]
                ba = lookup[(b, a)]
                auc_ab = float(ab["roc_auc"])
                auc_ba = float(ba["roc_auc"])
                output.append({
                    "scenario": scenario,
                    "model": model,
                    "domain_a": a,
                    "domain_b": b,
                    "auc_a_to_b": auc_ab,
                    "auc_b_to_a": auc_ba,
                    "reciprocal_mean_auc": (auc_ab + auc_ba) / 2.0,
                    "reciprocal_min_auc": min(auc_ab, auc_ba),
                    "reciprocal_max_auc": max(auc_ab, auc_ba),
                    "absolute_directional_asymmetry": abs(auc_ab - auc_ba),
                    "train_size_difference_absolute": abs(
                        int(ab["train_n"]) - int(ba["train_n"])
                    ),
                    "response_rate_difference_absolute": ab[
                        "absolute_response_rate_difference"
                    ],
                    "same_study_family": ab["same_study_family"],
                    "same_country": ab["same_country"],
                    "same_treatment_scope_group": ab["same_treatment_scope_group"],
                    **{column: ab[column] for column in DISTANCE_COLUMNS},
                })
    return output


def spearman_stat(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.unique(x[valid]).size < 2:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def binary_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[valid], y[valid]
    if not np.any(xx == 1) or not np.any(xx == 0):
        return float("nan")
    return float(np.mean(yy[xx == 1]) - np.mean(yy[xx == 0]))


def qap_univariable(
    auc_matrix: np.ndarray,
    predictor_matrix: np.ndarray,
    stat_kind: str,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    n = auc_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y = auc_matrix[mask]
    x = predictor_matrix[mask]
    if stat_kind == "spearman":
        observed = spearman_stat(x, y)
    else:
        observed = binary_mean_difference(x, y)
    if not math.isfinite(observed) or iterations <= 0:
        return {
            "observed_statistic": observed,
            "qap_p_two_sided": float("nan"),
            "qap_p_prespecified_direction": float("nan"),
            "permutations": 0,
        }

    rng = np.random.default_rng(seed)
    permuted_values = []
    for _ in range(iterations):
        perm = rng.permutation(n)
        xp = predictor_matrix[np.ix_(perm, perm)][mask]
        stat = (
            spearman_stat(xp, y)
            if stat_kind == "spearman"
            else binary_mean_difference(xp, y)
        )
        if math.isfinite(stat):
            permuted_values.append(stat)
    values = np.asarray(permuted_values, dtype=float)
    p_two = float((1 + np.sum(np.abs(values) >= abs(observed))) / (1 + len(values)))
    # Distance hypotheses are negative; binary relation hypotheses are positive.
    if stat_kind == "spearman":
        p_direction = float((1 + np.sum(values <= observed)) / (1 + len(values)))
    else:
        p_direction = float((1 + np.sum(values >= observed)) / (1 + len(values)))
    return {
        "observed_statistic": observed,
        "qap_p_two_sided": p_two,
        "qap_p_prespecified_direction": p_direction,
        "permutations": len(values),
    }


def matrices_from_pair_rows(
    rows: Sequence[Dict[str, object]],
    scenario: str,
    model: str,
    domains: Sequence[str],
    value_column: str,
) -> np.ndarray:
    matrix = np.full((len(domains), len(domains)), np.nan, dtype=float)
    index = {d: i for i, d in enumerate(domains)}
    for row in rows:
        if row["scenario"] != scenario or row["model"] != model:
            continue
        i = index[text(row["train_domain"])]
        j = index[text(row["test_domain"])]
        matrix[i, j] = float(row[value_column])
    return matrix


def qap_association_rows(
    pair_rows: Sequence[Dict[str, object]],
    iterations: int,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    keys = sorted(set((text(r["scenario"]), text(r["model"])) for r in pair_rows))
    for key_index, (scenario, model) in enumerate(keys):
        subset = [r for r in pair_rows if r["scenario"] == scenario and r["model"] == model]
        domains = sorted(
            set(text(r["train_domain"]) for r in subset)
            | set(text(r["test_domain"]) for r in subset)
        )
        auc_matrix = matrices_from_pair_rows(
            pair_rows, scenario, model, domains, "roc_auc"
        )
        for variable in DISTANCE_COLUMNS:
            predictor = matrices_from_pair_rows(
                pair_rows, scenario, model, domains, variable
            )
            result = qap_univariable(
                auc_matrix, predictor, "spearman", iterations,
                RANDOM_SEED + key_index * 1000 + DISTANCE_COLUMNS.index(variable),
            )
            output.append({
                "scenario": scenario,
                "model": model,
                "predictor": variable,
                "statistic_type": "spearman_rho",
                "prespecified_direction": "negative",
                **result,
            })
        for variable in BINARY_RELATIONS:
            predictor = matrices_from_pair_rows(
                pair_rows, scenario, model, domains, variable
            )
            result = qap_univariable(
                auc_matrix, predictor, "binary", iterations,
                RANDOM_SEED + key_index * 1000 + 100 + BINARY_RELATIONS.index(variable),
            )
            output.append({
                "scenario": scenario,
                "model": model,
                "predictor": variable,
                "statistic_type": "mean_auc_relation_1_minus_0",
                "prespecified_direction": "positive",
                **result,
            })
    return output


def standardize_vector(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return np.zeros_like(x), mean, 1.0
    return (x - mean) / sd, mean, sd


def ols_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X) @ y


def primary_mrqap(
    pair_rows: Sequence[Dict[str, object]],
    iterations: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    subset = [
        r for r in pair_rows
        if r["scenario"] == "PRIMARY" and r["model"] == "elastic_net"
    ]
    domains = sorted(
        set(text(r["train_domain"]) for r in subset)
        | set(text(r["test_domain"]) for r in subset)
    )
    n = len(domains)
    mask = ~np.eye(n, dtype=bool)
    y_matrix = matrices_from_pair_rows(
        pair_rows, "PRIMARY", "elastic_net", domains, "roc_auc"
    )
    y = y_matrix[mask]

    predictor_matrices = {
        variable: matrices_from_pair_rows(
            pair_rows, "PRIMARY", "elastic_net", domains, variable
        )
        for variable in PRIMARY_MRQAP_PREDICTORS
    }
    standardized_columns = []
    scaling = {}
    for variable in PRIMARY_MRQAP_PREDICTORS:
        vector = predictor_matrices[variable][mask]
        z, mean, sd = standardize_vector(vector)
        standardized_columns.append(z)
        scaling[variable] = {"mean": mean, "sd": sd}
    X = np.column_stack([np.ones(len(y))] + standardized_columns)
    beta = ols_coefficients(X, y)
    condition = float(np.linalg.cond(X))

    rng = np.random.default_rng(RANDOM_SEED + 900000)
    permuted = np.empty((iterations, len(beta)), dtype=float)
    for iteration in range(iterations):
        perm = rng.permutation(n)
        columns = []
        for variable in PRIMARY_MRQAP_PREDICTORS:
            vector = predictor_matrices[variable][np.ix_(perm, perm)][mask]
            mean = scaling[variable]["mean"]
            sd = scaling[variable]["sd"]
            columns.append((vector - mean) / sd)
        Xp = np.column_stack([np.ones(len(y))] + columns)
        permuted[iteration] = ols_coefficients(Xp, y)

    names = ["intercept"] + PRIMARY_MRQAP_PREDICTORS
    rows = []
    for j, name in enumerate(names):
        p_two = float(
            (1 + np.sum(np.abs(permuted[:, j]) >= abs(beta[j])))
            / (1 + iterations)
        )
        if name == "aitchison_centroid_distance":
            p_direction = float(
                (1 + np.sum(permuted[:, j] <= beta[j])) / (1 + iterations)
            )
            direction = "negative"
        elif name in {"same_study_family", "same_country", "same_treatment_scope_group"}:
            p_direction = float(
                (1 + np.sum(permuted[:, j] >= beta[j])) / (1 + iterations)
            )
            direction = "positive"
        else:
            p_direction = float("nan")
            direction = "not_prespecified"
        rows.append({
            "term": name,
            "standardized_coefficient": float(beta[j]),
            "mrqap_p_two_sided": p_two,
            "mrqap_p_prespecified_direction": p_direction,
            "prespecified_direction": direction,
            "permutations": iterations,
        })

    fitted = X @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    diagnostics = {
        "n_domains": n,
        "n_directed_pairs": len(y),
        "r_squared": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "design_condition_number": condition,
        "predictors": PRIMARY_MRQAP_PREDICTORS,
        "interpretation": (
            "Exploratory multiple-regression QAP; only nine domains. "
            "Coefficients must not be treated as causal effects."
        ),
    }
    return rows, diagnostics


def matrix_to_rows(
    domains: Sequence[str],
    values: Dict[str, Dict[str, float]],
    metric: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    rows = []
    for a in domains:
        row: Dict[str, object] = {"domain_id": a}
        for b in domains:
            row[b] = values[f"{a}|||{b}"][metric]
        rows.append(row)
    return rows, ["domain_id"] + list(domains)


def create_grayscale_heatmap_svg(
    path: Path,
    matrix_rows: Sequence[Dict[str, object]],
    domains: Sequence[str],
    title: str,
    minimum: float,
    maximum: float,
    diagonal_blank: bool,
) -> None:
    cell = 64
    left = 190
    top = 130
    width = left + cell * len(domains) + 40
    height = top + cell * len(domains) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        '<text x="20" y="52" font-family="Arial" font-size="11">Rows: training domain; columns: testing domain</text>',
    ]
    for j, domain in enumerate(domains):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top-8}" text-anchor="end" transform="rotate(-45 {x},{top-8})" '
            f'font-family="Arial" font-size="10">{domain}</text>'
        )
    for i, row in enumerate(matrix_rows):
        y = top + i * cell + cell / 2 + 4
        parts.append(
            f'<text x="{left-8}" y="{y}" text-anchor="end" font-family="Arial" font-size="10">'
            f'{row["train_domain"]}</text>'
        )
        for j, domain in enumerate(domains):
            value_raw = row.get(domain, "")
            x0 = left + j * cell
            y0 = top + i * cell
            if diagonal_blank and text(value_raw) == "":
                shade = 245
                label = "—"
            else:
                try:
                    value = float(value_raw)
                    ratio = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
                    ratio = min(1.0, max(0.0, ratio))
                    shade = int(round(245 - ratio * 190))
                    label = f"{value:.2f}"
                except Exception:
                    shade = 245
                    label = ""
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{cell}" height="{cell}" '
                f'fill="rgb({shade},{shade},{shade})" stroke="white"/>'
            )
            text_color = "white" if shade < 120 else "black"
            parts.append(
                f'<text x="{x0+cell/2}" y="{y0+cell/2+4}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="{text_color}">{label}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def create_scatter_svg(
    path: Path,
    rows: Sequence[Dict[str, object]],
    x_column: str,
    y_column: str,
    title: str,
) -> None:
    valid = [
        (float(r[x_column]), float(r[y_column]), int(r["same_study_family"]))
        for r in rows
        if math.isfinite(float(r[x_column])) and math.isfinite(float(r[y_column]))
    ]
    width, height = 760, 540
    left, right, top, bottom = 90, 30, 55, 75
    x_values = np.array([v[0] for v in valid], dtype=float)
    y_values = np.array([v[1] for v in valid], dtype=float)
    xmin, xmax = float(x_values.min()), float(x_values.max())
    ymin, ymax = min(0.35, float(y_values.min()) - 0.02), max(0.70, float(y_values.max()) + 0.02)
    def xp(x): return left + (x - xmin) / max(1e-12, xmax - xmin) * (width-left-right)
    def yp(y): return top + (ymax - y) / max(1e-12, ymax-ymin) * (height-top-bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{yp(0.5)}" x2="{width-right}" y2="{yp(0.5)}" stroke="gray" stroke-dasharray="4,4"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">{x_column}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20,{height/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_column}</text>',
    ]
    for tick in np.linspace(xmin, xmax, 5):
        parts.append(f'<text x="{xp(tick)}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="10">{tick:.2f}</text>')
    for tick in np.linspace(ymin, ymax, 6):
        parts.append(f'<text x="{left-8}" y="{yp(tick)+4}" text-anchor="end" font-family="Arial" font-size="10">{tick:.2f}</text>')
    for x, y, same_family in valid:
        radius = 5 if same_family else 3
        fill = "black" if same_family else "gray"
        parts.append(f'<circle cx="{xp(x)}" cy="{yp(y)}" r="{radius}" fill="{fill}" opacity="0.75"/>')
    parts.append(f'<text x="{left+5}" y="{top+15}" font-family="Arial" font-size="10">Black: same study family; gray: different family</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--qap", type=int, default=10000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    raw_root = args.root.strip().strip('"').rstrip("\\/").rstrip('"').rstrip("\\/")
    root = Path(raw_root).resolve()
    output_dir = root / "02_results_step10_5B_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.bootstrap = 10
        args.qap = 20
        scenarios_to_run = ["PRIMARY"]
        models_to_run = ["elastic_net"]
    else:
        scenarios_to_run = list(SCENARIOS)
        models_to_run = list(MODEL_ORDER)

    start = time.time()
    log("[start] Step 10.5B directed domain transfer and ecological distance")
    data = load_data(root)
    metadata, metadata_path = read_domain_metadata(root)
    domain_stats_list, distances, _ = compute_domain_ecology(data, metadata)
    domain_stats = {text(row["domain_id"]): row for row in domain_stats_list}
    domains_primary = sorted(np.unique(data.domains))

    distance_pair_rows = []
    for train_domain in domains_primary:
        for test_domain in domains_primary:
            if train_domain == test_domain:
                continue
            distance_pair_rows.append(
                pair_metadata(
                    train_domain, test_domain, domain_stats, metadata, distances
                )
            )

    all_metrics: List[Dict[str, object]] = []
    all_predictions: List[Dict[str, object]] = []
    scenario_summary: List[Dict[str, object]] = []

    for scenario in scenarios_to_run:
        indices, y, manifest_path = load_scenario(root, data, scenario)
        scenario_domains = data.domains[indices]
        scenario_summary.append({
            "scenario": scenario,
            "manifest_file": manifest_path.name,
            "n": len(indices),
            "responders": int(y.sum()),
            "non_responders": int(len(y) - y.sum()),
            "domains": len(np.unique(scenario_domains)),
            "directed_domain_pairs": len(np.unique(scenario_domains)) * (len(np.unique(scenario_domains)) - 1),
        })
        for model_name in models_to_run:
            metrics, predictions = run_directed_transfer(
                scenario, model_name, data, indices, y, args.bootstrap
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)

    # Merge ecological and metadata covariates into every transfer result.
    pair_lookup = {
        (text(row["train_domain"]), text(row["test_domain"])): row
        for row in distance_pair_rows
    }
    enriched_metrics = []
    for row in all_metrics:
        pair = pair_lookup[(text(row["train_domain"]), text(row["test_domain"]))]
        enriched = dict(row)
        enriched.update(pair)
        # Scenario-specific sizes/rates overwrite primary pair descriptors where available.
        enriched["log_train_n"] = math.log(int(row["train_n"]))
        enriched["log_test_n"] = math.log(int(row["test_n"]))
        enriched["absolute_response_rate_difference"] = abs(
            int(row["train_responders"]) / int(row["train_n"])
            - int(row["test_responders"]) / int(row["test_n"])
        )
        enriched_metrics.append(enriched)

    reciprocal_rows = reciprocal_summary(enriched_metrics)
    qap_rows = qap_association_rows(enriched_metrics, args.qap)
    mrqap_rows, mrqap_diagnostics = primary_mrqap(enriched_metrics, args.qap)

    # Main tables.
    write_tsv(
        output_dir / "scenario_summary.tsv",
        scenario_summary,
        ["scenario", "manifest_file", "n", "responders", "non_responders", "domains", "directed_domain_pairs"],
    )
    write_tsv(
        output_dir / "domain_metadata_used.tsv",
        list(metadata.values()),
        [
            "domain_id", "study_family", "country", "macro_region",
            "treatment_scope_group", "known_combination_ici_fraction",
            "therapy_detail", "source_doi", "source_note",
            "metadata_confidence", "treatment_analysis_role",
        ],
    )
    write_tsv(
        output_dir / "domain_ecology_and_outcome_summary.tsv",
        domain_stats_list,
        [
            "domain_id", "n", "responders", "non_responders", "response_rate",
            "mean_detected_species_per_sample", "global_ecology_features",
            "global_ecology_pseudocount", "study_family", "country", "macro_region",
            "treatment_scope_group", "known_combination_ici_fraction",
            "therapy_detail", "source_doi", "source_note",
            "metadata_confidence", "treatment_analysis_role",
        ],
    )
    write_tsv(
        output_dir / "directed_domain_transfer_metrics.tsv",
        enriched_metrics,
        [
            "scenario", "model", "train_domain", "test_domain",
            "train_n", "test_n", "train_responders", "train_non_responders",
            "test_responders", "test_non_responders", "selected_features",
            "pseudocount", "locked_params", "roc_auc", "roc_auc_ci_low",
            "roc_auc_ci_high", "roc_auc_bootstrap_valid", "pr_auc", "brier",
            "balanced_accuracy", "sensitivity", "specificity", "log_loss",
            "train_study_family", "test_study_family", "same_study_family",
            "train_country", "test_country", "same_country",
            "train_macro_region", "test_macro_region", "same_macro_region",
            "train_treatment_scope_group", "test_treatment_scope_group",
            "same_treatment_scope_group",
            "known_combination_ici_fraction_difference",
            "absolute_response_rate_difference", "log_train_n", "log_test_n",
            *DISTANCE_COLUMNS,
        ],
    )
    write_tsv(
        output_dir / "all_directed_transfer_predictions.tsv",
        all_predictions,
        [
            "scenario", "model", "train_domain", "test_domain", "manifest_id",
            "patient_id", "true_label", "predicted_probability", "predicted_class",
        ],
    )
    write_tsv(
        output_dir / "reciprocal_transfer_asymmetry.tsv",
        reciprocal_rows,
        [
            "scenario", "model", "domain_a", "domain_b",
            "auc_a_to_b", "auc_b_to_a", "reciprocal_mean_auc",
            "reciprocal_min_auc", "reciprocal_max_auc",
            "absolute_directional_asymmetry", "train_size_difference_absolute",
            "response_rate_difference_absolute", "same_study_family",
            "same_country", "same_treatment_scope_group", *DISTANCE_COLUMNS,
        ],
    )
    write_tsv(
        output_dir / "univariable_QAP_associations.tsv",
        qap_rows,
        [
            "scenario", "model", "predictor", "statistic_type",
            "prespecified_direction", "observed_statistic",
            "qap_p_two_sided", "qap_p_prespecified_direction", "permutations",
        ],
    )
    write_tsv(
        output_dir / "primary_elastic_net_MRQAP.tsv",
        mrqap_rows,
        [
            "term", "standardized_coefficient", "mrqap_p_two_sided",
            "mrqap_p_prespecified_direction", "prespecified_direction", "permutations",
        ],
    )
    (output_dir / "primary_elastic_net_MRQAP_diagnostics.json").write_text(
        json.dumps(mrqap_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Ecological distance matrices.
    for metric in DISTANCE_COLUMNS:
        matrix_rows, fields = matrix_to_rows(domains_primary, distances, metric)
        write_tsv(output_dir / f"ecological_distance_matrix_{metric}.tsv", matrix_rows, fields)

    # Transfer matrices and primary figures.
    for scenario in scenarios_to_run:
        for model_name in models_to_run:
            rows, fields = create_matrix_rows(
                enriched_metrics, scenario, model_name, "roc_auc"
            )
            write_tsv(
                output_dir / f"directed_AUC_matrix_{scenario}_{model_name}.tsv",
                rows, fields,
            )
            if scenario == "PRIMARY" and model_name == "elastic_net":
                create_grayscale_heatmap_svg(
                    output_dir / "Figure10_5B_primary_elastic_net_directed_AUC_heatmap.svg",
                    rows, fields[1:],
                    "Primary elastic-net directed domain-to-domain ROC AUC",
                    0.30, 0.70, True,
                )

    distance_rows, distance_fields = matrix_to_rows(
        domains_primary, distances, "aitchison_centroid_distance"
    )
    # Adapt first field for the heatmap helper.
    heat_rows = [
        {"train_domain": row["domain_id"], **{k: v for k, v in row.items() if k != "domain_id"}}
        for row in distance_rows
    ]
    create_grayscale_heatmap_svg(
        output_dir / "Figure10_5B_Aitchison_domain_distance_heatmap.svg",
        heat_rows, domains_primary,
        "Aitchison distance between domain centroids",
        0.0,
        max(float(v) for d in distances.values() for k, v in d.items() if k == "aitchison_centroid_distance"),
        False,
    )
    primary_elastic_pairs = [
        row for row in enriched_metrics
        if row["scenario"] == "PRIMARY" and row["model"] == "elastic_net"
    ]
    create_scatter_svg(
        output_dir / "Figure10_5B_Aitchison_distance_vs_external_AUC.svg",
        primary_elastic_pairs,
        "aitchison_centroid_distance", "roc_auc",
        "Ecological distance versus directed external ROC AUC",
    )

    primary_qap = [
        row for row in qap_rows
        if row["scenario"] == "PRIMARY" and row["model"] == "elastic_net"
    ]
    status = {
        "step": "Step10.5B",
        "version": VERSION,
        "status": "PASS_STEP10_5B_COMPLETED",
        "input": "Step10_4B_results_v2_species_only.zip / assembled_species_matrix.tsv",
        "primary_n": 363,
        "species_features": 2205,
        "primary_domains": 9,
        "primary_directed_pairs": 72,
        "models": models_to_run,
        "scenarios": scenarios_to_run,
        "locked_hyperparameters": LOCKED_PARAMS,
        "ecological_distances": DISTANCE_COLUMNS,
        "qap_iterations": args.qap,
        "auc_bootstrap_iterations_per_pair": args.bootstrap,
        "primary_elastic_net_qap": primary_qap,
        "primary_elastic_net_mrqap_diagnostics": mrqap_diagnostics,
        "treatment_analysis": (
            "Exploratory, coarse domain-level treatment categories only; "
            "unknown combination proportions were not imputed."
        ),
        "leakage_controls": [
            "Every directed model was trained on one domain only.",
            "Feature filtering, pseudocount, CLR and scaling were learned only from the training domain.",
            "The test domain never contributed to preprocessing or model fitting.",
            "Ecological distances were response-blind and used only for post-model transferability analysis.",
            "QAP/MRQAP permutation respected dyadic dependence among domain pairs.",
        ],
        "runtime_seconds": round(time.time() - start, 2),
        "interpretation_gate": "REQUIRES_SCIENTIFIC_AUDIT_BEFORE_MANUSCRIPT_CLAIMS",
    }
    (output_dir / "Step10_5B_status_v1.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = [
        "Step 10.5B results v1",
        "",
        f"Primary domains: 9",
        f"Primary directed train-to-test pairs: 72",
        f"Scenarios: {', '.join(scenarios_to_run)}",
        f"Models: {', '.join(models_to_run)}",
        "",
        "Primary inference:",
        "- Elastic-net PRIMARY.",
        "- Univariable QAP associations.",
        "- Exploratory multivariable MRQAP.",
        "",
        "Treatment metadata are coarse domain-level descriptors.",
        "Unknown treatment proportions were not imputed.",
        "Do not interpret treatment coefficients causally.",
    ]
    (output_dir / "README_RESULTS_STEP10_5B_v1.txt").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )

    output_zip = root / (
        "Step10_5B_SMOKE_TEST_results_v1.zip"
        if args.smoke_test else "Step10_5B_results_v1.zip"
    )
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir))
    log(json.dumps(status, ensure_ascii=False, indent=2))
    log(f"[output] {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
