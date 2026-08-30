#!/usr/bin/env python
"""Prespecified DEBIAS-M secondary benchmark.

The frozen CLR elastic-net remains primary.
Standard mode is transductive.
Online mode performs deployment-time target-batch adaptation.
Target labels are joined only after probabilities are frozen.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

FULL_SEEDS = [20260802, 20260803, 20260804, 20260805, 20260806]
PILOT_SEEDS = [20260802]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kit-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pilot-only", action="store_true")
    return p.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    import pytorch_lightning as pl
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def read_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data(root: Path):
    matrix_path = root / "inputs/assembled_species_matrix.tsv"
    manifest_path = root / "inputs/Step10_2_manifest_R3_primary_lock_v8.tsv"

    manifest = [
        r for r in read_tsv(manifest_path)
        if str(r.get("r3_v8_primary_analysis_include", "")).upper() in {"YES", "TRUE", "1"}
    ]
    by_id = {r["manifest_id"]: r for r in manifest}

    with matrix_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    ids = np.asarray([r[0] for r in rows])
    if set(ids) != set(by_id):
        raise RuntimeError("Matrix/manifest IDs do not match")

    X = np.asarray([[float(v) for v in r[4:]] for r in rows], dtype=float)
    y = np.asarray([1 if r[3] == "Responder" else 0 for r in rows], dtype=int)
    domains = np.asarray([by_id[i]["domain_id"] for i in ids])
    families = np.asarray([by_id[i]["study_family"] for i in ids])

    if X.shape != (363, 2205):
        raise RuntimeError(f"Unexpected matrix shape {X.shape}")
    if not np.isfinite(X).all() or (X < 0).any():
        raise RuntimeError("Matrix must be finite and nonnegative")
    if not np.allclose(X.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Matrix rows do not sum to one")

    return ids, X, y, domains, families, matrix_path, manifest_path

def training_feature_indices(X_train, max_features=500):
    prevalence = (X_train > 0).mean(axis=0)
    selected = np.where(prevalence >= 0.10)[0]
    if len(selected) < 10:
        selected = np.where(prevalence >= 0.05)[0]
    if len(selected) < 5:
        selected = np.where(np.any(X_train > 0, axis=0))[0]

    positive = X_train[:, selected][X_train[:, selected] > 0]
    pc = float(np.clip((positive.min() / 2 if positive.size else 1e-6), 1e-8, 1e-4))
    Z = np.log(X_train[:, selected] + pc)
    Z -= Z.mean(axis=1, keepdims=True)
    variance = Z.var(axis=0)

    if len(selected) > max_features:
        selected = selected[np.argsort(variance)[::-1][:max_features]]
    return selected

def training_pseudocount(X_train):
    positive = X_train[X_train > 0]
    return float(np.clip((positive.min() / 2 if positive.size else 1e-6), 1e-8, 1e-4))

def clr(X, pseudocount):
    Z = np.log(X + pseudocount)
    return Z - Z.mean(axis=1, keepdims=True)

def fold_batch_ids(train_domains, target_domains):
    ordered_train = []
    for d in train_domains:
        if d not in ordered_train:
            ordered_train.append(d)
    ordered_target = []
    for d in target_domains:
        if d not in ordered_target:
            ordered_target.append(d)

    mapping = {d: i for i, d in enumerate(ordered_train)}
    next_id = len(mapping)
    for d in ordered_target:
        if d not in mapping:
            mapping[d] = next_id
            next_id += 1

    return (
        np.asarray([mapping[d] for d in train_domains], dtype=int),
        np.asarray([mapping[d] for d in target_domains], dtype=int),
        mapping,
    )

def correct(mode, X_train, X_target, train_batch, target_batch, y_train, seed):
    set_seed(seed)
    import torch
    from debiasm import DebiasMClassifier, OnlineDebiasMClassifier

    Xtr = np.column_stack([train_batch, X_train])
    Xte = np.column_stack([target_batch, X_target])

    if mode == "standard_transductive":
        model = DebiasMClassifier(x_val=Xte, random_state=seed, min_epochs=25)
        model.fit(Xtr, y_train)
    elif mode == "online_adaptive":
        model = OnlineDebiasMClassifier(random_state=seed, min_epochs=25)
        model.fit(Xtr, y_train)
    else:
        raise ValueError(mode)

    corrected_train = np.asarray(model.transform(Xtr), dtype=float)
    corrected_target = np.asarray(model.transform(Xte), dtype=float)

    if mode == "standard_transductive":
        embedded_prob = np.asarray(model.predict_proba(Xte), dtype=float)[:, 1]
    else:
        # Avoid repeating the expensive 10,000-step target adaptation.
        with torch.no_grad():
            embedded_prob = torch.softmax(
                model.model.linear(torch.tensor(corrected_target).float()), dim=1
            )[:, 1].cpu().numpy()

    return corrected_train, corrected_target, embedded_prob, model

def locked_elastic_net(corrected_train, corrected_target, y_train, seed):
    from sklearn.linear_model import LogisticRegression

    pc = training_pseudocount(corrected_train)
    Ztr = clr(corrected_train, pc)
    Zte = clr(corrected_target, pc)

    mean = Ztr.mean(axis=0)
    sd = Ztr.std(axis=0)
    sd[sd < 1e-10] = 1.0

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=0.2,
        l1_ratio=0.75,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    model.fit((Ztr - mean) / sd, y_train)
    p = model.predict_proba((Zte - mean) / sd)[:, 1]
    return p, pc, int(np.count_nonzero(model.coef_))

def main():
    args = parse_args()
    root = Path(args.kit_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    smoke = json.loads((root / "outputs/smoke_test_status.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise RuntimeError("Smoke test must pass before benchmark execution")

    ids, X, y, domains, families, matrix_path, manifest_path = load_data(root)
    seeds = PILOT_SEEDS if args.pilot_only else FULL_SEEDS
    modes = ["standard_transductive", "online_adaptive"]
    pilot = {("LODO", "Lee_Barcelona"), ("LOSFO", "LeeKA_2022")}

    predictions, audit, failures = [], [], []

    for scheme, groups in [("LODO", domains), ("LOSFO", families)]:
        held_groups = list(dict.fromkeys(groups.tolist()))
        for held in held_groups:
            if args.pilot_only and (scheme, held) not in pilot:
                continue

            test = groups == held
            train = ~test

            selected = training_feature_indices(X[train], max_features=500)
            Xtr = X[train][:, selected]
            Xte = X[test][:, selected]
            ytr = y[train]
            yte = y[test]
            target_ids = ids[test]
            target_domains = domains[test]

            train_batch, target_batch, batch_map = fold_batch_ids(domains[train], domains[test])

            for mode in modes:
                for seed in seeds:
                    started = time.time()
                    try:
                        corrected_train, corrected_target, embedded_prob, _ = correct(
                            mode, Xtr, Xte, train_batch, target_batch, ytr, seed
                        )
                        locked_prob, corrected_pc, nonzero = locked_elastic_net(
                            corrected_train, corrected_target, ytr, seed
                        )

                        # Probabilities are now frozen. Target labels are joined only here.
                        for i, mid in enumerate(target_ids):
                            predictions.append({
                                "scheme": scheme,
                                "held_out_group": held,
                                "mode": mode,
                                "seed": seed,
                                "manifest_id": mid,
                                "domain_id": target_domains[i],
                                "y": int(yte[i]),
                                "locked_downstream_probability": float(locked_prob[i]),
                                "embedded_DEBIASM_probability": float(embedded_prob[i]),
                            })

                        audit.append({
                            "scheme": scheme,
                            "held_out_group": held,
                            "mode": mode,
                            "seed": seed,
                            "status": "PASS",
                            "n_train": int(train.sum()),
                            "n_target": int(test.sum()),
                            "n_features": int(len(selected)),
                            "corrected_training_pseudocount": corrected_pc,
                            "locked_nonzero_coefficients": nonzero,
                            "batch_mapping_json": json.dumps(batch_map, sort_keys=True),
                            "runtime_seconds": time.time() - started,
                            "matrix_sha256": sha256(matrix_path),
                            "manifest_sha256": sha256(manifest_path),
                        })
                    except Exception as exc:
                        failures.append({
                            "scheme": scheme,
                            "held_out_group": held,
                            "mode": mode,
                            "seed": seed,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "runtime_seconds": time.time() - started,
                        })

    write_tsv(out / "predictions.tsv", predictions)
    write_tsv(out / "fit_audit.tsv", audit)
    write_tsv(out / "fit_failures.tsv", failures)

    expected = 4 if args.pilot_only else 140
    passed = len(audit)
    status = {
        "status": "PASS" if passed == expected and not failures else "INCOMPLETE_OR_FAILED",
        "pilot_only": args.pilot_only,
        "expected_fits": expected,
        "passed_fits": passed,
        "failed_fits": len(failures),
        "seeds": seeds,
        "modes": modes,
    }
    (out / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if status["status"] != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
