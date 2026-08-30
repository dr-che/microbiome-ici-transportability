#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

FAMILY_MAP = {
    "FrankelAE_2017": "Frankel_2017",
    "GopalakrishnanV_2018": "Gopalakrishnan_2018",
    "MatsonV_2018": "Matson_2018",
    "SpencerCN_2021": "Spencer_2021",
    "LeeKA_2022": "Lee_2022_family",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kit-root", required=True)
    p.add_argument("--benchmark-dir", required=True)
    return p.parse_args()

def read_tsv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def calibration(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])

def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    prevalence = float(y.mean())
    intercept, slope = calibration(y, p)
    return {
        "n": int(len(y)),
        "responders": int(y.sum()),
        "non_responders": int(len(y) - y.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        "pr_auc": float(average_precision_score(y, p)),
        "pr_lift": float(average_precision_score(y, p) / prevalence) if prevalence > 0 else math.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }

def main():
    args = parse_args()
    root = Path(args.kit_root).resolve()
    bdir = Path(args.benchmark_dir).resolve()
    preds = read_tsv(bdir / "predictions.tsv")

    pred_types = [
        ("locked_downstream", "locked_downstream_probability"),
        ("embedded_DEBIASM", "embedded_DEBIASM_probability"),
    ]

    fold_rows = []
    grouped = defaultdict(list)
    for r in preds:
        for ptype, col in pred_types:
            grouped[(r["scheme"], r["held_out_group"], r["mode"], int(r["seed"]), ptype)].append(
                (int(r["y"]), float(r[col]))
            )

    for key, vals in grouped.items():
        scheme, held, mode, seed, ptype = key
        m = metrics([v[0] for v in vals], [v[1] for v in vals])
        fold_rows.append({
            "scheme": scheme,
            "held_out_group": held,
            "mode": mode,
            "seed": seed,
            "prediction_type": ptype,
            **m,
        })

    # Pooled metrics by scheme/mode/seed/prediction type.
    pooled_groups = defaultdict(list)
    for r in preds:
        for ptype, col in pred_types:
            pooled_groups[(r["scheme"], r["mode"], int(r["seed"]), ptype)].append(
                (int(r["y"]), float(r[col]))
            )

    pooled_rows = []
    for key, vals in pooled_groups.items():
        scheme, mode, seed, ptype = key
        m = metrics([v[0] for v in vals], [v[1] for v in vals])
        matching_folds = [
            r for r in fold_rows
            if r["scheme"] == scheme and r["mode"] == mode
            and r["seed"] == seed and r["prediction_type"] == ptype
        ]
        pooled_rows.append({
            "scheme": scheme,
            "mode": mode,
            "seed": seed,
            "prediction_type": ptype,
            **m,
            "macro_roc_auc": float(np.nanmean([r["roc_auc"] for r in matching_folds])),
            "macro_pr_auc": float(np.nanmean([r["pr_auc"] for r in matching_folds])),
            "folds": len(matching_folds),
        })

    # Frozen baseline lookup.
    baseline_folds = read_tsv(root / "inputs/frozen_baseline_fold_metrics.tsv")
    baseline_lookup = {}
    for r in baseline_folds:
        if r["scenario"] != "PRIMARY" or r["model"] != "elastic_net":
            continue
        if r["validation_scheme"] == "LEAVE_ONE_DOMAIN_OUT":
            baseline_lookup[("LODO", r["test_group"])] = r
        elif r["validation_scheme"] == "LEAVE_ONE_STUDY_FAMILY_OUT":
            rev = {v: k for k, v in FAMILY_MAP.items()}
            if r["test_group"] in rev:
                baseline_lookup[("LOSFO", rev[r["test_group"]])] = r

    comparisons = []
    for r in fold_rows:
        base = baseline_lookup.get((r["scheme"], r["held_out_group"]))
        if not base:
            continue
        comparisons.append({
            "scheme": r["scheme"],
            "held_out_group": r["held_out_group"],
            "mode": r["mode"],
            "seed": r["seed"],
            "prediction_type": r["prediction_type"],
            "debiasm_roc_auc": r["roc_auc"],
            "frozen_baseline_roc_auc": float(base["roc_auc"]),
            "delta_roc_auc": r["roc_auc"] - float(base["roc_auc"]),
            "debiasm_pr_auc": r["pr_auc"],
            "frozen_baseline_pr_auc": float(base["pr_auc"]),
            "delta_pr_auc": r["pr_auc"] - float(base["pr_auc"]),
            "debiasm_brier": r["brier"],
            "frozen_baseline_brier": float(base["brier"]),
            "delta_brier": r["brier"] - float(base["brier"]),
        })

    write_tsv(bdir / "fold_metrics.tsv", fold_rows)
    write_tsv(bdir / "pooled_metrics.tsv", pooled_rows)
    write_tsv(bdir / "fold_vs_frozen_baseline.tsv", comparisons)

    status = json.loads((bdir / "benchmark_status.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS" if status.get("status") == "PASS" else "INCOMPLETE",
        "fold_metric_rows": len(fold_rows),
        "pooled_metric_rows": len(pooled_rows),
        "comparison_rows": len(comparisons),
        "interpretation": (
            "Secondary adaptation benchmark only. Standard mode is transductive; "
            "online mode uses an unlabeled target batch at deployment."
        ),
    }
    (bdir / "evaluation_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
