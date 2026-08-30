#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t as student_t

VERSION = "v1"
MIN_GROUP_N_PER_CLASS = 3
MIN_PREVALENCE_ABUNDANCE = 0.10
MIN_NONZERO_TOTAL = 4
MIN_ABSENT_TOTAL_PREVALENCE = 4
MIN_META_GROUPS = 3
ROBUST_MIN_DOMAINS = 5

SCENARIOS = {
    "PRIMARY": "Step10_2_manifest_R3_primary_lock_v8.tsv",
    "S1_EXCLUDE_BCN12": "Step10_2_manifest_R3_sensitivity_S1_exclude_BCN12_v8.tsv",
    "S2_BCN12_AS_NR": "Step10_2_manifest_R3_sensitivity_S2_BCN12_as_NR_v8.tsv",
    "S3_EXCLUDE_LEE_SITES": "Step10_2_manifest_R3_sensitivity_S3_exclude_Lee_sites_v8.tsv",
}

DOMAIN_ORDER = [
    "Frankel_US", "Gopalakrishnan_US", "Matson_US", "Spencer_US",
    "Lee_PRIMM_UK", "Lee_PRIMM_NL", "Lee_Manchester", "Lee_Leeds", "Lee_Barcelona",
]
FAMILY_ORDER = [
    "Frankel_2017", "Gopalakrishnan_2018", "Matson_2018", "Spencer_2021", "Lee_2022_family"
]
FAMILY_CANONICAL = {
    "FrankelAE_2017": "Frankel_2017",
    "GopalakrishnanV_2018": "Gopalakrishnan_2018",
    "MatsonV_2018": "Matson_2018",
    "SpencerCN_2021": "Spencer_2021",
    "LeeKA_2022": "Lee_2022_family",
}


def log(message: str) -> None:
    print(message, flush=True)


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def yes(value: object) -> bool:
    return text(value).lower() in {"yes", "y", "true", "1"}


def clean_root(raw: str) -> Path:
    value = raw.strip().strip('"').rstrip("\\/").rstrip('"').rstrip("\\/")
    return Path(value).resolve()


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text(row.get(field, "")) for field in fields})


def read_tsv_path(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [{k: text(v) for k, v in row.items()} for row in reader]
    return rows, fields


def locate_file(root: Path, filename: str) -> Path:
    for candidate in [root / filename, root / "01_downloaded" / filename, root / "02_results_step10_4B_v2_species_only" / filename]:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    matches = [p for p in root.rglob(filename) if p.is_file() and p.stat().st_size > 0 and "02_results_step10_5C" not in str(p)]
    if not matches:
        raise FileNotFoundError(filename)
    return sorted(matches, key=lambda p: (len(p.parts), str(p).lower()))[0]


def load_matrix(root: Path) -> Tuple[List[str], List[str], np.ndarray, List[str], List[str], List[str]]:
    extracted = root / "02_results_step10_4B_v2_species_only" / "assembled_species_matrix.tsv"
    zip_path = root / "Step10_4B_results_v2_species_only.zip"
    if extracted.is_file():
        source = extracted
        handle = source.open("r", encoding="utf-8-sig", newline="")
        close_handle = True
        source_name = str(source)
    elif zip_path.is_file():
        archive = zipfile.ZipFile(zip_path)
        member = next((n for n in archive.namelist() if n.endswith("assembled_species_matrix.tsv")), None)
        if member is None:
            archive.close()
            raise FileNotFoundError("assembled_species_matrix.tsv in Step10_4B_results_v2_species_only.zip")
        handle = io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="")
        close_handle = True
        source_name = f"{zip_path.name}/{member}"
    else:
        raise FileNotFoundError("Step10_4B_results_v2_species_only.zip or extracted assembled_species_matrix.tsv")

    try:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:4] != ["manifest_id", "patient_id", "domain_id", "response_harmonized"]:
            raise RuntimeError("Unexpected assembled matrix header")
        features = header[4:]
        manifest_ids, patient_ids, domains, labels, values = [], [], [], [], []
        for row in reader:
            manifest_ids.append(row[0])
            patient_ids.append(row[1])
            domains.append(row[2])
            labels.append(row[3])
            values.append([float(x) for x in row[4:]])
    finally:
        if close_handle:
            handle.close()
        if 'archive' in locals():
            archive.close()

    X = np.asarray(values, dtype=float)
    if X.shape != (363, 2205):
        raise RuntimeError(f"Expected 363x2205 matrix, observed {X.shape}")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise RuntimeError("Duplicate manifest IDs in assembled matrix")
    log(f"[data] source={source_name}; n={X.shape[0]}; species={X.shape[1]}")
    return manifest_ids, patient_ids, X, domains, labels, features


def scenario_selection(root: Path, manifest_ids: List[str], scenario: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    rows, _ = read_tsv_path(locate_file(root, SCENARIOS[scenario]))
    by_id = {r["manifest_id"]: r for r in rows}
    idx, y, domains = [], [], []
    family_map: Dict[str, str] = {}
    for i, mid in enumerate(manifest_ids):
        row = by_id.get(mid)
        if row is None:
            continue
        include = yes(row.get("strict_primary_target")) and row.get("profile_mapping_status") == "MATCHED" and yes(row.get("r3_v8_primary_analysis_include"))
        if not include:
            continue
        label = row.get("response_harmonized")
        if label not in {"Responder", "Non-responder"}:
            raise RuntimeError(f"Invalid label for {mid}: {label}")
        idx.append(i)
        y.append(1 if label == "Responder" else 0)
        domains.append(row["domain_id"])
        raw_family = row.get("study_family") or row["domain_id"]
        family_map[row["domain_id"]] = FAMILY_CANONICAL.get(raw_family, raw_family)
    return np.asarray(idx, dtype=int), np.asarray(y, dtype=int), np.asarray(domains, dtype=object), family_map


def hedges_g_vector(clr: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = clr[y == 1]
    x0 = clr[y == 0]
    n1, n0 = x1.shape[0], x0.shape[0]
    mean1 = x1.mean(axis=0)
    mean0 = x0.mean(axis=0)
    var1 = x1.var(axis=0, ddof=1)
    var0 = x0.var(axis=0, ddof=1)
    df = n1 + n0 - 2
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / max(df, 1)
    pooled_sd = np.sqrt(np.maximum(pooled_var, 0))
    d = np.divide(mean1 - mean0, pooled_sd, out=np.full_like(mean1, np.nan), where=pooled_sd > 1e-12)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    g = correction * d
    var_g = (n1 + n0) / (n1 * n0) + np.square(g) / (2.0 * max(df, 1))
    return g, var_g, mean1, mean0, var1, var0


def group_effects(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    features: List[str],
    scenario: str,
    analysis_level: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    order = DOMAIN_ORDER if analysis_level == "domain" else FAMILY_ORDER
    for group in sorted(np.unique(groups), key=lambda x: order.index(x) if x in order else 999):
        mask = groups == group
        Xg = X[mask]
        yg = y[mask]
        n1, n0 = int(yg.sum()), int(len(yg) - yg.sum())
        if n1 < MIN_GROUP_N_PER_CLASS or n0 < MIN_GROUP_N_PER_CLASS:
            log(f"[skip group] {scenario}/{analysis_level}/{group}: R={n1}, NR={n0}")
            continue
        positive = Xg[Xg > 0]
        pseudocount = float(np.min(positive) / 2.0) if positive.size else 1e-6
        pseudocount = min(max(pseudocount, 1e-8), 1e-4)
        logX = np.log(Xg + pseudocount)
        clr = logX - logX.mean(axis=1, keepdims=True)
        g, var_g, mean1, mean0, var1, var0 = hedges_g_vector(clr, yg)
        present = Xg > 0
        prevalence = present.mean(axis=0)
        nz_total = present.sum(axis=0)
        absent_total = len(yg) - nz_total
        a = present[yg == 1].sum(axis=0).astype(float)
        b = n1 - a
        c = present[yg == 0].sum(axis=0).astype(float)
        d = n0 - c
        correction_mask = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        aa, bb, cc, dd = a.copy(), b.copy(), c.copy(), d.copy()
        aa[correction_mask] += 0.5
        bb[correction_mask] += 0.5
        cc[correction_mask] += 0.5
        dd[correction_mask] += 0.5
        log_or = np.log((aa * dd) / (bb * cc))
        var_log_or = 1/aa + 1/bb + 1/cc + 1/dd
        abundance_eligible = (
            (prevalence >= MIN_PREVALENCE_ABUNDANCE)
            & (nz_total >= MIN_NONZERO_TOTAL)
            & np.isfinite(g) & np.isfinite(var_g) & (var_g > 0)
        )
        prevalence_eligible = (
            (nz_total >= MIN_NONZERO_TOTAL)
            & (absent_total >= MIN_ABSENT_TOTAL_PREVALENCE)
            & np.isfinite(log_or) & np.isfinite(var_log_or) & (var_log_or > 0)
        )
        for j, species in enumerate(features):
            common = {
                "scenario": scenario,
                "analysis_level": analysis_level,
                "group": group,
                "species": species,
                "n": len(yg),
                "responders": n1,
                "non_responders": n0,
                "prevalence": float(prevalence[j]),
                "responder_prevalence": float(a[j] / n1),
                "non_responder_prevalence": float(c[j] / n0),
                "pseudocount": pseudocount,
            }
            if abundance_eligible[j]:
                se = math.sqrt(float(var_g[j]))
                rows.append({
                    **common,
                    "effect_type": "CLR_HEDGES_G",
                    "effect": float(g[j]),
                    "variance": float(var_g[j]),
                    "se": se,
                    "ci_low": float(g[j] - 1.96 * se),
                    "ci_high": float(g[j] + 1.96 * se),
                    "responder_mean": float(mean1[j]),
                    "non_responder_mean": float(mean0[j]),
                })
            if prevalence_eligible[j]:
                se = math.sqrt(float(var_log_or[j]))
                rows.append({
                    **common,
                    "effect_type": "PRESENCE_LOG_OR",
                    "effect": float(log_or[j]),
                    "variance": float(var_log_or[j]),
                    "se": se,
                    "ci_low": float(log_or[j] - 1.96 * se),
                    "ci_high": float(log_or[j] + 1.96 * se),
                    "responder_mean": "",
                    "non_responder_mean": "",
                })
        log(f"[effects] {scenario}/{analysis_level}/{group}: n={len(yg)}, R={n1}, NR={n0}, abundance={int(abundance_eligible.sum())}, prevalence={int(prevalence_eligible.sum())}")
    return rows


def reml_meta(effects: Sequence[float], variances: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[mask], v[mask]
    k = len(y)
    if k < 2:
        return {}
    max_tau = max(float(np.var(y, ddof=1)) * 10.0, float(np.max(v)) * 10.0, 1.0)
    def objective(tau2: float) -> float:
        w = 1.0 / (v + tau2)
        mu = float(np.sum(w * y) / np.sum(w))
        q = float(np.sum(w * (y - mu) ** 2))
        return 0.5 * (float(np.sum(np.log(v + tau2))) + math.log(float(np.sum(w))) + q)
    result = minimize_scalar(objective, bounds=(0.0, max_tau), method="bounded", options={"xatol": 1e-10})
    tau2 = max(0.0, float(result.x)) if result.success else 0.0
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    q_star = float(np.sum(w * (y - mu) ** 2) / max(k - 1, 1))
    q_star_modified = max(q_star, 1.0)
    se = math.sqrt(q_star_modified / float(np.sum(w)))
    df = k - 1
    crit = float(student_t.ppf(0.975, df)) if df > 0 else 1.96
    ci_low, ci_high = mu - crit * se, mu + crit * se
    t_stat = mu / se if se > 0 else float("nan")
    p = float(2 * student_t.sf(abs(t_stat), df)) if df > 0 and math.isfinite(t_stat) else float("nan")
    w_fixed = 1.0 / v
    mu_fixed = float(np.sum(w_fixed * y) / np.sum(w_fixed))
    Q = float(np.sum(w_fixed * (y - mu_fixed) ** 2))
    Q_p = float(chi2.sf(Q, k - 1)) if k > 1 else float("nan")
    I2 = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 and k > 1 else 0.0
    if k >= 3:
        pred_crit = float(student_t.ppf(0.975, k - 2))
        pred_se = math.sqrt(max(tau2, 0.0) + se * se)
        pred_low, pred_high = mu - pred_crit * pred_se, mu + pred_crit * pred_se
    else:
        pred_low = pred_high = float("nan")
    pos = int(np.sum(y > 0)); neg = int(np.sum(y < 0)); zero = k - pos - neg
    sign_consistency = max(pos, neg) / k
    return {
        "k": k, "pooled_effect": mu, "se": se, "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p, "tau2": tau2, "Q": Q, "Q_p": Q_p, "I2": I2,
        "prediction_low": pred_low, "prediction_high": pred_high,
        "positive_groups": pos, "negative_groups": neg, "zero_groups": zero,
        "sign_consistency": sign_consistency,
        "minimum_effect": float(np.min(y)), "maximum_effect": float(np.max(y)),
        "median_effect": float(np.median(y)),
    }


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.empty(m)
    previous = 1.0
    for rank_index in range(m - 1, -1, -1):
        rank = rank_index + 1
        value = p[order[rank_index]] * m / rank
        previous = min(previous, value)
        adjusted[rank_index] = min(previous, 1.0)
    for i, idx in enumerate(order):
        q[idx] = adjusted[i]
    return q.tolist()


def meta_analyze(effect_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in effect_rows:
        grouped[(text(row["scenario"]), text(row["analysis_level"]), text(row["effect_type"]), text(row["species"]))].append(row)
    output: List[Dict[str, object]] = []
    for (scenario, level, effect_type, species), rows in grouped.items():
        if len(rows) < MIN_META_GROUPS:
            continue
        result = reml_meta([float(r["effect"]) for r in rows], [float(r["variance"]) for r in rows])
        if not result:
            continue
        significant_positive = sum(float(r["ci_low"]) > 0 for r in rows)
        significant_negative = sum(float(r["ci_high"]) < 0 for r in rows)
        output.append({
            "scenario": scenario, "analysis_level": level, "effect_type": effect_type,
            "species": species, **result,
            "significant_positive_groups": significant_positive,
            "significant_negative_groups": significant_negative,
            "group_list": ";".join(sorted(text(r["group"]) for r in rows)),
        })
    strata: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i, row in enumerate(output):
        strata[(text(row["scenario"]), text(row["analysis_level"]), text(row["effect_type"]))].append(i)
    for indices in strata.values():
        qvals = bh_adjust([float(output[i]["p_value"]) for i in indices])
        for i, q in zip(indices, qvals):
            output[i]["fdr_q"] = q
    return output


def influence_summary(effect_rows: List[Dict[str, object]], meta_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in effect_rows:
        if row["analysis_level"] == "domain" and row["effect_type"] == "CLR_HEDGES_G":
            by_key[(text(row["scenario"]), text(row["species"]))].append(row)
    meta_map = {(text(r["scenario"]), text(r["species"])): r for r in meta_rows if r["analysis_level"] == "domain" and r["effect_type"] == "CLR_HEDGES_G"}
    output = []
    for key, rows in by_key.items():
        meta = meta_map.get(key)
        if meta is None or len(rows) < 4:
            continue
        # Restrict detailed influence to potentially interpretable or heterogeneous species.
        if not (float(meta.get("p_value", 1)) < 0.20 or float(meta.get("I2", 0)) >= 50 or float(meta.get("sign_consistency", 0)) >= 0.75):
            continue
        full = float(meta["pooled_effect"])
        abs_effects = np.asarray([abs(float(r["effect"])) for r in rows])
        dominant_share = float(abs_effects.max() / abs_effects.sum()) if abs_effects.sum() > 0 else float("nan")
        best = None
        sign_flip = False
        for omitted in rows:
            kept = [r for r in rows if r is not omitted]
            loo = reml_meta([float(r["effect"]) for r in kept], [float(r["variance"]) for r in kept])
            if not loo:
                continue
            delta = float(loo["pooled_effect"] - full)
            flip = full != 0 and float(loo["pooled_effect"]) * full < 0
            sign_flip = sign_flip or flip
            candidate = (abs(delta), text(omitted["group"]), float(loo["pooled_effect"]), delta)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            continue
        max_effect_row = rows[int(np.argmax(abs_effects))]
        output.append({
            "scenario": key[0], "species": key[1], "full_pooled_effect": full,
            "full_I2": meta["I2"], "full_fdr_q": meta.get("fdr_q", ""),
            "influential_omitted_domain": best[1], "max_abs_loo_delta": best[0],
            "pooled_effect_after_influential_omission": best[2], "signed_loo_delta": best[3],
            "any_leave_one_domain_sign_flip": int(sign_flip),
            "largest_absolute_effect_domain": max_effect_row["group"],
            "largest_absolute_domain_effect": max_effect_row["effect"],
            "dominant_domain_absolute_effect_share": dominant_share,
        })
    return output


def classify_species(
    meta_rows: List[Dict[str, object]],
    influence_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    domain = {(text(r["scenario"]), text(r["species"])): r for r in meta_rows if r["analysis_level"] == "domain" and r["effect_type"] == "CLR_HEDGES_G"}
    family = {(text(r["scenario"]), text(r["species"])): r for r in meta_rows if r["analysis_level"] == "family" and r["effect_type"] == "CLR_HEDGES_G"}
    prevalence = {(text(r["scenario"]), text(r["species"])): r for r in meta_rows if r["analysis_level"] == "domain" and r["effect_type"] == "PRESENCE_LOG_OR"}
    influence = {(text(r["scenario"]), text(r["species"])): r for r in influence_rows}
    output = []
    for key, d in domain.items():
        f = family.get(key, {})
        p = prevalence.get(key, {})
        inf = influence.get(key, {})
        k = int(d["k"]); pooled = float(d["pooled_effect"]); q = float(d.get("fdr_q", 1)); I2 = float(d["I2"])
        consistency = float(d["sign_consistency"])
        sig_pos = int(d["significant_positive_groups"]); sig_neg = int(d["significant_negative_groups"])
        family_same_sign = bool(f) and float(f.get("pooled_effect", 0)) * pooled > 0
        family_support = family_same_sign and float(f.get("p_value", 1)) < 0.10 and float(f.get("sign_consistency", 0)) >= 0.60
        prevalence_same_sign = bool(p) and float(p.get("pooled_effect", 0)) * pooled > 0
        significant_domain_count = sig_pos + sig_neg
        dominant_share = float(inf.get("dominant_domain_absolute_effect_share", float("nan"))) if inf else float("nan")
        loo_flip = int(inf.get("any_leave_one_domain_sign_flip", 0)) if inf else 0
        if k >= ROBUST_MIN_DOMAINS and q < 0.10 and consistency >= 0.75 and I2 < 50 and family_support:
            classification = "ROBUST_CONSISTENT_POSITIVE" if pooled > 0 else "ROBUST_CONSISTENT_NEGATIVE"
        elif k >= ROBUST_MIN_DOMAINS and sig_pos >= 1 and sig_neg >= 1:
            classification = "STRONG_DIRECTION_REVERSAL"
        elif k >= ROBUST_MIN_DOMAINS and int(d["positive_groups"]) >= 2 and int(d["negative_groups"]) >= 2 and consistency <= 0.67 and I2 >= 75:
            classification = "PROBABLE_DIRECTION_REVERSAL"
        elif k >= ROBUST_MIN_DOMAINS and q >= 0.10 and significant_domain_count == 1 and I2 >= 50 and ((math.isfinite(dominant_share) and dominant_share >= 0.30) or loo_flip):
            classification = "DOMAIN_SPECIFIC"
        elif k >= ROBUST_MIN_DOMAINS and I2 >= 75:
            classification = "HIGH_HETEROGENEITY"
        elif k >= ROBUST_MIN_DOMAINS and consistency >= 0.75 and I2 < 50 and family_same_sign:
            classification = "CONSISTENT_NONSIGNIFICANT"
        else:
            classification = "OTHER_OR_INSUFFICIENT"
        output.append({
            "scenario": key[0], "species": key[1], "classification": classification,
            "domain_k": k, "domain_pooled_effect": pooled, "domain_ci_low": d["ci_low"], "domain_ci_high": d["ci_high"],
            "domain_p_value": d["p_value"], "domain_fdr_q": d.get("fdr_q", ""), "domain_I2": I2,
            "domain_prediction_low": d["prediction_low"], "domain_prediction_high": d["prediction_high"],
            "domain_sign_consistency": consistency, "positive_domains": d["positive_groups"], "negative_domains": d["negative_groups"],
            "significant_positive_domains": sig_pos, "significant_negative_domains": sig_neg,
            "family_k": f.get("k", ""), "family_pooled_effect": f.get("pooled_effect", ""), "family_p_value": f.get("p_value", ""),
            "family_fdr_q": f.get("fdr_q", ""), "family_I2": f.get("I2", ""), "family_sign_consistency": f.get("sign_consistency", ""),
            "family_same_direction": int(family_same_sign), "family_support_p_lt_0_10": int(family_support),
            "prevalence_meta_effect": p.get("pooled_effect", ""), "prevalence_meta_p": p.get("p_value", ""),
            "prevalence_meta_fdr_q": p.get("fdr_q", ""), "prevalence_same_direction": int(prevalence_same_sign),
            "dominant_domain_share": inf.get("dominant_domain_absolute_effect_share", ""),
            "influential_domain": inf.get("influential_omitted_domain", ""), "loo_sign_flip": loo_flip,
        })
    return output


def sensitivity_concordance(class_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_species: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in class_rows:
        by_species[text(row["species"])][text(row["scenario"])] = row
    output = []
    for species, rows in by_species.items():
        if "PRIMARY" not in rows:
            continue
        effects = [float(rows[s]["domain_pooled_effect"]) for s in SCENARIOS if s in rows]
        signs = [1 if e > 0 else -1 if e < 0 else 0 for e in effects]
        classes = [text(rows[s]["classification"]) for s in SCENARIOS if s in rows]
        primary = rows["PRIMARY"]
        output.append({
            "species": species,
            "available_scenarios": len(effects),
            "same_pooled_direction_all_scenarios": int(len(set(signs)) == 1),
            "pooled_effect_range": max(effects) - min(effects) if effects else "",
            "same_classification_all_scenarios": int(len(set(classes)) == 1),
            "primary_classification": primary["classification"],
            "primary_pooled_effect": primary["domain_pooled_effect"],
            "primary_fdr_q": primary["domain_fdr_q"],
            "primary_I2": primary["domain_I2"],
            "S1_effect": rows.get("S1_EXCLUDE_BCN12", {}).get("domain_pooled_effect", ""),
            "S2_effect": rows.get("S2_BCN12_AS_NR", {}).get("domain_pooled_effect", ""),
            "S3_effect": rows.get("S3_EXCLUDE_LEE_SITES", {}).get("domain_pooled_effect", ""),
            "S1_classification": rows.get("S1_EXCLUDE_BCN12", {}).get("classification", ""),
            "S2_classification": rows.get("S2_BCN12_AS_NR", {}).get("classification", ""),
            "S3_classification": rows.get("S3_EXCLUDE_LEE_SITES", {}).get("classification", ""),
        })
    return output


def escape_xml(value: object) -> str:
    return text(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def color_for_effect(value: float, max_abs: float = 1.5) -> str:
    value = max(-max_abs, min(max_abs, value)) / max_abs
    if value >= 0:
        r = 245; g = int(245 - 150 * value); b = int(245 - 180 * value)
    else:
        v = -value; r = int(245 - 180 * v); g = int(245 - 110 * v); b = 245
    return f"rgb({r},{g},{b})"


def make_heatmap(path: Path, class_rows: List[Dict[str, object]], effect_rows: List[Dict[str, object]]) -> None:
    primary_classes = [r for r in class_rows if r["scenario"] == "PRIMARY"]
    priority = {
        "ROBUST_CONSISTENT_POSITIVE": 0, "ROBUST_CONSISTENT_NEGATIVE": 0,
        "STRONG_DIRECTION_REVERSAL": 1, "PROBABLE_DIRECTION_REVERSAL": 2,
        "DOMAIN_SPECIFIC": 3, "HIGH_HETEROGENEITY": 4,
        "CONSISTENT_NONSIGNIFICANT": 5, "OTHER_OR_INSUFFICIENT": 9,
    }
    primary_classes.sort(key=lambda r: (priority.get(text(r["classification"]), 9), float(r["domain_fdr_q"]) if text(r["domain_fdr_q"]) else 1, -abs(float(r["domain_pooled_effect"]))))
    selected = primary_classes[:40]
    species_order = [text(r["species"]) for r in selected]
    effect_map = {(text(r["species"]), text(r["group"])): float(r["effect"]) for r in effect_rows if r["scenario"] == "PRIMARY" and r["analysis_level"] == "domain" and r["effect_type"] == "CLR_HEDGES_G"}
    domains = [d for d in DOMAIN_ORDER if any((s, d) in effect_map for s in species_order)]
    cell_w, cell_h, left, top = 92, 22, 360, 150
    width = left + cell_w * len(domains) + 40
    height = top + cell_h * len(species_order) + 80
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append('<text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold">Step 10.5C domain effect direction map</text>')
    svg.append('<text x="20" y="55" font-family="Arial" font-size="12">Red: higher CLR abundance in responders; blue: lower in responders</text>')
    for j, domain in enumerate(domains):
        x = left + j * cell_w + cell_w/2
        svg.append(f'<text x="{x}" y="{top-10}" font-family="Arial" font-size="10" text-anchor="end" transform="rotate(-55 {x} {top-10})">{escape_xml(domain)}</text>')
    class_map = {text(r["species"]): text(r["classification"]) for r in selected}
    for i, species in enumerate(species_order):
        y = top + i * cell_h
        label = species.replace("s__", "")
        svg.append(f'<text x="{left-8}" y="{y+15}" font-family="Arial" font-size="10" text-anchor="end">{escape_xml(label)}</text>')
        svg.append(f'<text x="10" y="{y+15}" font-family="Arial" font-size="9">{escape_xml(class_map[species])}</text>')
        for j, domain in enumerate(domains):
            x = left + j * cell_w
            value = effect_map.get((species, domain))
            fill = "#eeeeee" if value is None else color_for_effect(value)
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}" stroke="white"/>')
            if value is not None:
                svg.append(f'<text x="{x+cell_w/2}" y="{y+14}" font-family="Arial" font-size="9" text-anchor="middle">{value:.2f}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def make_volcano(path: Path, class_rows: List[Dict[str, object]]) -> None:
    rows = [r for r in class_rows if r["scenario"] == "PRIMARY" and text(r["domain_p_value"])]
    width, height, left, right, top, bottom = 1000, 700, 90, 40, 70, 80
    xvals = np.asarray([float(r["domain_pooled_effect"]) for r in rows])
    yvals = np.asarray([-math.log10(max(float(r["domain_p_value"]), 1e-300)) for r in rows])
    xmax = max(0.5, float(np.max(np.abs(xvals))) * 1.1)
    ymax = max(2.0, float(np.max(yvals)) * 1.1)
    def sx(x): return left + (x + xmax) / (2*xmax) * (width-left-right)
    def sy(y): return top + (ymax-y)/ymax * (height-top-bottom)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append('<text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold">Random-effects meta-analysis of domain-level CLR effects</text>')
    svg.append(f'<line x1="{sx(0)}" y1="{top}" x2="{sx(0)}" y2="{height-bottom}" stroke="#999"/>')
    svg.append(f'<line x1="{left}" y1="{sy(-math.log10(0.05))}" x2="{width-right}" y2="{sy(-math.log10(0.05))}" stroke="#999" stroke-dasharray="5,5"/>')
    colors = {"STRONG_DIRECTION_REVERSAL":"#7b3294", "PROBABLE_DIRECTION_REVERSAL":"#c2a5cf", "DOMAIN_SPECIFIC":"#fdae61", "HIGH_HETEROGENEITY":"#fee08b", "ROBUST_CONSISTENT_POSITIVE":"#d73027", "ROBUST_CONSISTENT_NEGATIVE":"#4575b4", "CONSISTENT_NONSIGNIFICANT":"#66bd63", "OTHER_OR_INSUFFICIENT":"#888888"}
    for r, x, y in zip(rows, xvals, yvals):
        c = colors.get(text(r["classification"]), "#888888")
        radius = 2.5 + min(4.0, float(r["domain_I2"])/25.0)
        svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{radius:.2f}" fill="{c}" fill-opacity="0.65"/>')
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>')
    svg.append(f'<text x="{width/2}" y="{height-25}" font-family="Arial" font-size="14" text-anchor="middle">Pooled Hedges g (Responder minus Non-responder)</text>')
    svg.append(f'<text x="20" y="{height/2}" font-family="Arial" font-size="14" text-anchor="middle" transform="rotate(-90 20 {height/2})">-log10(P)</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def make_classification_chart(path: Path, class_rows: List[Dict[str, object]]) -> None:
    rows = [r for r in class_rows if r["scenario"] == "PRIMARY"]
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[text(r["classification"])] += 1
    classes = sorted(counts, key=lambda c: counts[c], reverse=True)
    width, height, left, top, bottom = 1000, 520, 310, 60, 50
    max_count = max(counts.values()) if counts else 1
    bar_h = max(24, int((height-top-bottom)/max(len(classes),1)))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold">Primary species classification counts</text>']
    for i, cls in enumerate(classes):
        y = top + i*bar_h
        w = (width-left-80)*counts[cls]/max_count
        svg.append(f'<text x="{left-10}" y="{y+bar_h*0.7}" font-family="Arial" font-size="12" text-anchor="end">{escape_xml(cls)}</text>')
        svg.append(f'<rect x="{left}" y="{y+3}" width="{w}" height="{bar_h-6}" fill="#777"/>')
        svg.append(f'<text x="{left+w+8}" y="{y+bar_h*0.7}" font-family="Arial" font-size="12">{counts[cls]}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = clean_root(args.root)
    result_dir = root / ("02_results_step10_5C_SMOKE_v1" if args.smoke_test else "02_results_step10_5C_v1")
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    manifest_ids, patient_ids, X_all, matrix_domains, matrix_labels, features_all = load_matrix(root)
    if args.smoke_test:
        features = features_all[:80]
        X_all = X_all[:, :80]
        scenarios = ["PRIMARY"]
    else:
        features = features_all
        scenarios = list(SCENARIOS)

    all_effects: List[Dict[str, object]] = []
    scenario_rows = []
    for scenario in scenarios:
        idx, y, domains, family_map = scenario_selection(root, manifest_ids, scenario)
        X = X_all[idx]
        families = np.asarray([family_map[d] for d in domains], dtype=object)
        scenario_rows.append({
            "scenario": scenario, "n": len(y), "responders": int(y.sum()), "non_responders": int(len(y)-y.sum()),
            "domains": len(np.unique(domains)), "families": len(np.unique(families)),
        })
        log(f"[scenario] {scenario}: n={len(y)}, R={int(y.sum())}, NR={int(len(y)-y.sum())}, domains={len(np.unique(domains))}, families={len(np.unique(families))}")
        all_effects.extend(group_effects(X, y, domains, features, scenario, "domain"))
        all_effects.extend(group_effects(X, y, families, features, scenario, "family"))

    log(f"[meta] total eligible group-level effects={len(all_effects)}")
    meta_rows = meta_analyze(all_effects)
    log(f"[meta] random-effects summaries={len(meta_rows)}")
    influence_rows = influence_summary(all_effects, meta_rows)
    class_rows = classify_species(meta_rows, influence_rows)
    sensitivity_rows = sensitivity_concordance(class_rows)

    effect_fields = [
        "scenario","analysis_level","group","species","effect_type","n","responders","non_responders",
        "prevalence","responder_prevalence","non_responder_prevalence","pseudocount","effect","variance","se",
        "ci_low","ci_high","responder_mean","non_responder_mean"
    ]
    meta_fields = [
        "scenario","analysis_level","effect_type","species","k","pooled_effect","se","ci_low","ci_high","p_value","fdr_q",
        "tau2","Q","Q_p","I2","prediction_low","prediction_high","positive_groups","negative_groups","zero_groups",
        "sign_consistency","significant_positive_groups","significant_negative_groups","minimum_effect","maximum_effect","median_effect","group_list"
    ]
    influence_fields = [
        "scenario","species","full_pooled_effect","full_I2","full_fdr_q","influential_omitted_domain","max_abs_loo_delta",
        "pooled_effect_after_influential_omission","signed_loo_delta","any_leave_one_domain_sign_flip",
        "largest_absolute_effect_domain","largest_absolute_domain_effect","dominant_domain_absolute_effect_share"
    ]
    class_fields = [
        "scenario","species","classification","domain_k","domain_pooled_effect","domain_ci_low","domain_ci_high","domain_p_value","domain_fdr_q","domain_I2",
        "domain_prediction_low","domain_prediction_high","domain_sign_consistency","positive_domains","negative_domains",
        "significant_positive_domains","significant_negative_domains","family_k","family_pooled_effect","family_p_value","family_fdr_q","family_I2","family_sign_consistency",
        "family_same_direction","family_support_p_lt_0_10","prevalence_meta_effect","prevalence_meta_p","prevalence_meta_fdr_q","prevalence_same_direction",
        "dominant_domain_share","influential_domain","loo_sign_flip"
    ]
    sensitivity_fields = [
        "species","available_scenarios","same_pooled_direction_all_scenarios","pooled_effect_range","same_classification_all_scenarios",
        "primary_classification","primary_pooled_effect","primary_fdr_q","primary_I2","S1_effect","S2_effect","S3_effect",
        "S1_classification","S2_classification","S3_classification"
    ]
    write_tsv(result_dir/"scenario_summary.tsv", scenario_rows, ["scenario","n","responders","non_responders","domains","families"])
    write_tsv(result_dir/"domain_and_family_species_effects.tsv", all_effects, effect_fields)
    write_tsv(result_dir/"species_random_effects_meta_analysis.tsv", meta_rows, meta_fields)
    write_tsv(result_dir/"leave_one_domain_out_influence_summary.tsv", influence_rows, influence_fields)
    write_tsv(result_dir/"species_directionality_and_specificity_classification.tsv", class_rows, class_fields)
    write_tsv(result_dir/"sensitivity_scenario_concordance.tsv", sensitivity_rows, sensitivity_fields)

    primary_classes = [r for r in class_rows if r["scenario"] == "PRIMARY"]
    consistent = [r for r in primary_classes if text(r["classification"]).startswith("ROBUST_CONSISTENT") or r["classification"] == "CONSISTENT_NONSIGNIFICANT"]
    reversals = [r for r in primary_classes if "DIRECTION_REVERSAL" in text(r["classification"])]
    domain_specific = [r for r in primary_classes if r["classification"] == "DOMAIN_SPECIFIC"]
    heterogeneous = [r for r in primary_classes if r["classification"] == "HIGH_HETEROGENEITY"]
    write_tsv(result_dir/"cross_domain_consistent_candidates.tsv", sorted(consistent, key=lambda r: (float(r["domain_fdr_q"]) if text(r["domain_fdr_q"]) else 1, -float(r["domain_sign_consistency"]), -abs(float(r["domain_pooled_effect"])))), class_fields)
    write_tsv(result_dir/"direction_reversal_candidates.tsv", sorted(reversals, key=lambda r: (-float(r["domain_I2"]), -int(r["significant_positive_domains"])-int(r["significant_negative_domains"]))), class_fields)
    write_tsv(result_dir/"domain_specific_candidates.tsv", sorted(domain_specific, key=lambda r: (-float(r["domain_I2"]), -float(r["dominant_domain_share"]) if text(r["dominant_domain_share"]) else 0)), class_fields)

    make_heatmap(result_dir/"Figure10_5C_domain_effect_direction_heatmap.svg", class_rows, all_effects)
    make_volcano(result_dir/"Figure10_5C_meta_effect_heterogeneity_map.svg", class_rows)
    make_classification_chart(result_dir/"Figure10_5C_species_classification_counts.svg", class_rows)

    primary_domain_abundance = [r for r in meta_rows if r["scenario"] == "PRIMARY" and r["analysis_level"] == "domain" and r["effect_type"] == "CLR_HEDGES_G"]
    primary_family_abundance = [r for r in meta_rows if r["scenario"] == "PRIMARY" and r["analysis_level"] == "family" and r["effect_type"] == "CLR_HEDGES_G"]
    class_counts: Dict[str, int] = defaultdict(int)
    for r in primary_classes:
        class_counts[text(r["classification"])] += 1
    status = {
        "step": "Step10.5C", "version": VERSION, "status": "PASS_STEP10_5C_COMPLETED" if not args.smoke_test else "PASS_SMOKE_TEST",
        "input_n": 363, "species_features": len(features), "scenarios": scenarios,
        "primary_effect": "Domain-level Hedges g of CLR abundance (Responder minus Non-responder)",
        "secondary_effect": "Domain-level log odds ratio of species presence",
        "meta_model": "REML random effects with modified Hartung-Knapp confidence intervals",
        "study_family_sensitivity": "Patient-level effects recalculated within five study families, followed by random-effects meta-analysis",
        "primary_species_meta_analyzed": len(primary_domain_abundance),
        "primary_family_meta_analyzed": len(primary_family_abundance),
        "primary_domain_fdr_q_lt_0_05": int(sum(float(r.get("fdr_q", 1)) < 0.05 for r in primary_domain_abundance)),
        "primary_domain_fdr_q_lt_0_10": int(sum(float(r.get("fdr_q", 1)) < 0.10 for r in primary_domain_abundance)),
        "primary_classification_counts": dict(class_counts),
        "robust_consistent_candidates": int(sum(text(r["classification"]).startswith("ROBUST_CONSISTENT") for r in primary_classes)),
        "direction_reversal_candidates": len(reversals),
        "domain_specific_candidates": len(domain_specific),
        "high_heterogeneity_candidates": len(heterogeneous),
        "interpretation_gate": "REQUIRES_FINAL_SCIENTIFIC_REVIEW",
        "runtime_seconds": round(time.time()-start, 2),
    }
    (result_dir/"Step10_5C_status_v1.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir/"README_RESULTS_STEP10_5C_v1.txt").write_text(
        "Step 10.5C results\n\nPrimary effect: CLR Hedges g, responder minus non-responder.\n"
        "Positive pooled effects indicate higher relative CLR abundance in responders.\n"
        "Candidate classifications are prespecified screening categories and require final scientific review.\n"
        "No species may be described as a validated biomarker solely from this step.\n",
        encoding="utf-8"
    )

    output_zip = root / ("Step10_5C_SMOKE_TEST_results_v1.zip" if args.smoke_test else "Step10_5C_results_v1.zip")
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    log(json.dumps(status, ensure_ascii=False, indent=2))
    log(f"[output] {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
