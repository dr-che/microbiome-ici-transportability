#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = Path('/mnt/data/melanoma_transportability_review/repo_v1/melanoma_microbiome_transportability')
OUT = Path('/mnt/data/melanoma_transportability_review/analysis3_outputs')
OUT.mkdir(parents=True, exist_ok=True)
MATRIX = ROOT / 'data/derived/assembled_species_matrix.tsv'
FROZEN_METRICS = ROOT / 'results/step10_5B/directed_domain_transfer_metrics.tsv'

FEATURE_CAPS = [10, 25, 50, 500]
MIN_PREVALENCE = 0.10
FALLBACK_PREVALENCE = 0.05
C_VALUE = 0.2
L1_RATIO = 0.75
RANDOM_SEED = 20260720
DOWNSAMPLE_N = 25
DOWNSAMPLE_REPEATS = 100


def txt(x):
    return '' if x is None else str(x)


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: txt(r.get(k, '')) for k in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Data:
    manifest_ids: List[str]
    patient_ids: List[str]
    domains: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    X: np.ndarray


def load_data() -> Data:
    with MATRIX.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if header[:4] != ['manifest_id', 'patient_id', 'domain_id', 'response_harmonized']:
            raise RuntimeError(f'Unexpected header: {header[:4]}')
        mids, pids, domains, y, vals = [], [], [], [], []
        for row in reader:
            mids.append(row[0]); pids.append(row[1]); domains.append(row[2])
            y.append(1 if row[3] == 'Responder' else 0)
            vals.append([float(v) for v in row[4:]])
    X = np.asarray(vals, dtype=float)
    X[X < 0] = 0
    sums = X.sum(axis=1)
    if np.any(sums <= 0):
        raise RuntimeError('Zero-sum sample')
    X = X / sums[:, None]
    return Data(mids, pids, np.asarray(domains, dtype=object), np.asarray(y, dtype=int), header[4:], X)


@dataclass
class Prep:
    selected: np.ndarray
    pseudocount: float
    mean: np.ndarray
    scale: np.ndarray


def fit_prep(X_train: np.ndarray, cap: int) -> Prep:
    prevalence = np.mean(X_train > 0, axis=0)
    selected = np.where(prevalence >= MIN_PREVALENCE)[0]
    if len(selected) < 10:
        selected = np.where(prevalence >= FALLBACK_PREVALENCE)[0]
    if len(selected) < 5:
        selected = np.where(np.any(X_train > 0, axis=0))[0]
    if len(selected) == 0:
        raise RuntimeError('No features after prevalence filtering')
    nonzero = X_train[:, selected][X_train[:, selected] > 0]
    pc = float(np.min(nonzero) / 2.0) if nonzero.size else 1e-6
    pc = min(max(pc, 1e-8), 1e-4)
    clr = np.log(X_train[:, selected] + pc)
    clr -= clr.mean(axis=1, keepdims=True)
    variances = np.var(clr, axis=0)
    if len(selected) > cap:
        order = np.argsort(variances)[::-1][:cap]
        selected = selected[order]
        clr = np.log(X_train[:, selected] + pc)
        clr -= clr.mean(axis=1, keepdims=True)
    mean = clr.mean(axis=0)
    scale = clr.std(axis=0, ddof=0)
    scale[scale < 1e-10] = 1.0
    return Prep(selected, pc, mean, scale)


def apply_prep(X: np.ndarray, prep: Prep) -> np.ndarray:
    clr = np.log(X[:, prep.selected] + prep.pseudocount)
    clr -= clr.mean(axis=1, keepdims=True)
    return (clr - prep.mean) / prep.scale


def build_model(seed: int):
    return LogisticRegression(
        penalty='elasticnet', solver='saga', C=C_VALUE, l1_ratio=L1_RATIO,
        class_weight='balanced', max_iter=5000, tol=1e-3,
        random_state=seed, n_jobs=1,
    )


def metric(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
    }


def summarize_directed(rows: List[Dict[str, object]], cap: int, analysis_set: str) -> Dict[str, object]:
    aucs = np.array([float(r['roc_auc']) for r in rows])
    prs = np.array([float(r['pr_auc']) for r in rows])
    briers = np.array([float(r['brier']) for r in rows])
    return {
        'feature_cap': cap,
        'analysis_set': analysis_set,
        'directed_pairs': len(rows),
        'mean_roc_auc': float(np.mean(aucs)),
        'median_roc_auc': float(np.median(aucs)),
        'sd_roc_auc': float(np.std(aucs, ddof=1)),
        'mean_pr_auc': float(np.mean(prs)),
        'mean_brier': float(np.mean(briers)),
    }


def reciprocal_rows(rows: List[Dict[str, object]], eligible_domains: set | None = None) -> List[Dict[str, object]]:
    lookup = {(r['train_domain'], r['test_domain']): float(r['roc_auc']) for r in rows}
    domains = sorted(set(r['train_domain'] for r in rows) | set(r['test_domain'] for r in rows))
    if eligible_domains is not None:
        domains = [d for d in domains if d in eligible_domains]
    out = []
    for a, b in combinations(domains, 2):
        if (a, b) not in lookup or (b, a) not in lookup:
            continue
        ab, ba = lookup[(a, b)], lookup[(b, a)]
        out.append({
            'domain_a': a, 'domain_b': b,
            'auc_a_to_b': ab, 'auc_b_to_a': ba,
            'absolute_delta_auc': abs(ab - ba),
        })
    return out


def summarize_asym(pair_rows: List[Dict[str, object]], cap: int, analysis_set: str) -> Dict[str, object]:
    vals = np.array([float(r['absolute_delta_auc']) for r in pair_rows])
    return {
        'feature_cap': cap,
        'analysis_set': analysis_set,
        'reciprocal_pairs': len(vals),
        'mean_abs_delta_auc': float(np.mean(vals)),
        'median_abs_delta_auc': float(np.median(vals)),
        'pairs_ge_0_10': int(np.sum(vals >= 0.10)),
        'pairs_ge_0_20': int(np.sum(vals >= 0.20)),
        'max_abs_delta_auc': float(np.max(vals)),
    }


def run_full_cap(data: Data, cap: int) -> List[Dict[str, object]]:
    rows = []
    domains = sorted(np.unique(data.domains))
    for si, source in enumerate(domains):
        train_idx = np.where(data.domains == source)[0]
        prep = fit_prep(data.X[train_idx], cap)
        Xtr = apply_prep(data.X[train_idx], prep)
        model = build_model(RANDOM_SEED + si * 1000)
        model.fit(Xtr, data.y[train_idx])
        for target in domains:
            if target == source:
                continue
            test_idx = np.where(data.domains == target)[0]
            p = model.predict_proba(apply_prep(data.X[test_idx], prep))[:, 1]
            rows.append({
                'feature_cap': cap,
                'train_domain': source,
                'test_domain': target,
                'train_n': len(train_idx),
                'test_n': len(test_idx),
                'train_responders': int(data.y[train_idx].sum()),
                'test_responders': int(data.y[test_idx].sum()),
                'selected_features': len(prep.selected),
                'pseudocount': prep.pseudocount,
                **metric(data.y[test_idx], p),
            })
    return rows


def stratified_sample_25(indices: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    if len(indices) == DOWNSAMPLE_N:
        return indices.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=DOWNSAMPLE_N, random_state=seed)
    local = np.arange(len(indices))
    sampled_local, _ = next(splitter.split(local, y[indices]))
    return indices[sampled_local]


def run_downsample_repeat(data: Data, cap: int, repeat: int, eligible_sources: List[str]) -> List[Dict[str, object]]:
    rows = []
    all_domains = sorted(np.unique(data.domains))
    source_order = {d: i for i, d in enumerate(all_domains)}
    for source in eligible_sources:
        full_idx = np.where(data.domains == source)[0]
        sample_seed = RANDOM_SEED + 10_000_000 + repeat * 100_000 + source_order[source] * 1000
        train_idx = stratified_sample_25(full_idx, data.y, sample_seed)
        prep = fit_prep(data.X[train_idx], cap)
        Xtr = apply_prep(data.X[train_idx], prep)
        model = build_model(sample_seed + 17)
        model.fit(Xtr, data.y[train_idx])
        for target in all_domains:
            if target == source:
                continue
            test_idx = np.where(data.domains == target)[0]
            p = model.predict_proba(apply_prep(data.X[test_idx], prep))[:, 1]
            rows.append({
                'feature_cap': cap,
                'repeat': repeat,
                'train_domain': source,
                'test_domain': target,
                'train_n_original': len(full_idx),
                'train_n_sampled': len(train_idx),
                'train_responders_sampled': int(data.y[train_idx].sum()),
                'selected_features': len(prep.selected),
                **metric(data.y[test_idx], p),
            })
    return rows


def quantiles(x: np.ndarray) -> Tuple[float, float]:
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def main():
    t0 = time.time()
    data = load_data()
    domains = sorted(np.unique(data.domains))
    counts = {d: int(np.sum(data.domains == d)) for d in domains}
    eligible = sorted([d for d, n in counts.items() if n >= DOWNSAMPLE_N])
    eligible_set = set(eligible)

    full_rows_all = []
    pair_rows_all = []
    summary_rows = []
    matrix_by_cap = {}
    asym_by_cap = {}

    for cap in FEATURE_CAPS:
        rows = run_full_cap(data, cap)
        full_rows_all.extend(rows)
        pairs = reciprocal_rows(rows)
        for r in pairs:
            r['feature_cap'] = cap
        pair_rows_all.extend(pairs)
        summary = summarize_directed(rows, cap, 'all_9_sources_all_targets')
        summary.update(summarize_asym(pairs, cap, 'all_9_sources_all_targets'))
        summary_rows.append(summary)

        ge25_rows = [r for r in rows if int(r['train_n']) >= DOWNSAMPLE_N]
        ge25_pairs = reciprocal_rows(ge25_rows, eligible_set)
        summary2 = summarize_directed(ge25_rows, cap, 'source_n_ge_25_all_targets')
        summary2.update(summarize_asym(ge25_pairs, cap, 'source_n_ge_25_all_targets'))
        summary_rows.append(summary2)

        matrix_by_cap[cap] = {(r['train_domain'], r['test_domain']): float(r['roc_auc']) for r in rows}
        asym_by_cap[cap] = {(r['domain_a'], r['domain_b']): float(r['absolute_delta_auc']) for r in pairs}

    # Stability vs cap 500
    stability_rows = []
    base_keys = sorted(matrix_by_cap[500])
    pair_keys = sorted(asym_by_cap[500])
    for cap in FEATURE_CAPS:
        x = np.array([matrix_by_cap[500][k] for k in base_keys])
        y = np.array([matrix_by_cap[cap][k] for k in base_keys])
        px = np.array([asym_by_cap[500][k] for k in pair_keys])
        py = np.array([asym_by_cap[cap][k] for k in pair_keys])
        stability_rows.append({
            'feature_cap': cap,
            'directed_auc_spearman_vs_cap500': float(spearmanr(x, y).statistic),
            'directed_auc_mean_absolute_difference_vs_cap500': float(np.mean(np.abs(x-y))),
            'reciprocal_asymmetry_spearman_vs_cap500': float(spearmanr(px, py).statistic),
            'reciprocal_asymmetry_mean_absolute_difference_vs_cap500': float(np.mean(np.abs(px-py))),
        })

    # Reproduction check against frozen 500-cap results
    frozen = []
    with FROZEN_METRICS.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['scenario'] == 'PRIMARY' and r['model'] == 'elastic_net':
                frozen.append(r)
    frozen_map = {(r['train_domain'], r['test_domain']): float(r['roc_auc']) for r in frozen}
    current_map = matrix_by_cap[500]
    reproduction_diffs = np.array([abs(current_map[k] - frozen_map[k]) for k in frozen_map])

    # Downsample all caps, 100 repeats
    downsample_repeat_summary = []
    downsample_rows_all = []
    for cap in FEATURE_CAPS:
        for repeat in range(1, DOWNSAMPLE_REPEATS + 1):
            rows = run_downsample_repeat(data, cap, repeat, eligible)
            downsample_rows_all.extend(rows)
            pairs = reciprocal_rows(rows, eligible_set)
            d = summarize_directed(rows, cap, 'downsample_sources_to_25_all_targets')
            d.update(summarize_asym(pairs, cap, 'downsample_sources_to_25_all_targets'))
            d['repeat'] = repeat
            downsample_repeat_summary.append(d)
        print(f'[complete] cap={cap}, downsample repeats={DOWNSAMPLE_REPEATS}', flush=True)

    # Aggregate downsample repetitions
    aggregate_rows = []
    metrics = ['mean_roc_auc','median_roc_auc','mean_pr_auc','mean_brier','mean_abs_delta_auc','median_abs_delta_auc','pairs_ge_0_10','pairs_ge_0_20','max_abs_delta_auc']
    for cap in FEATURE_CAPS:
        subset = [r for r in downsample_repeat_summary if int(r['feature_cap']) == cap]
        row = {'feature_cap': cap, 'repeats': len(subset), 'eligible_source_domains': len(eligible), 'directed_pairs_per_repeat': int(subset[0]['directed_pairs']), 'reciprocal_pairs_per_repeat': int(subset[0]['reciprocal_pairs'])}
        for m in metrics:
            vals = np.array([float(r[m]) for r in subset])
            lo, hi = quantiles(vals)
            row[f'{m}_mean'] = float(np.mean(vals))
            row[f'{m}_sd'] = float(np.std(vals, ddof=1))
            row[f'{m}_q025'] = lo
            row[f'{m}_q975'] = hi
        aggregate_rows.append(row)

    # Domain counts
    domain_rows = []
    for d in domains:
        idx = np.where(data.domains == d)[0]
        domain_rows.append({'domain': d, 'n': len(idx), 'responders': int(data.y[idx].sum()), 'non_responders': int(len(idx)-data.y[idx].sum()), 'eligible_source_n_ge_25': int(d in eligible_set)})

    # Write outputs
    write_tsv(OUT/'Analysis3_domain_counts.tsv', domain_rows, list(domain_rows[0].keys()))
    write_tsv(OUT/'Analysis3_feature_cap_directed_metrics.tsv', full_rows_all, list(full_rows_all[0].keys()))
    write_tsv(OUT/'Analysis3_feature_cap_reciprocal_pairs.tsv', pair_rows_all, list(pair_rows_all[0].keys()))
    write_tsv(OUT/'Analysis3_feature_cap_and_source_size_summary.tsv', summary_rows, list(summary_rows[0].keys()))
    write_tsv(OUT/'Analysis3_feature_cap_stability_vs_500.tsv', stability_rows, list(stability_rows[0].keys()))
    write_tsv(OUT/'Analysis3_downsample_repeat_summary.tsv', downsample_repeat_summary, list(downsample_repeat_summary[0].keys()))
    write_tsv(OUT/'Analysis3_downsample_aggregate_summary.tsv', aggregate_rows, list(aggregate_rows[0].keys()))
    # Full downsample directed rows are useful but sizable; retain for reproducibility.
    write_tsv(OUT/'Analysis3_downsample_directed_metrics.tsv', downsample_rows_all, list(downsample_rows_all[0].keys()))

    status = {
        'analysis': 'Analysis 3: feature-cap, source-n>=25, and source-size-matched sensitivity',
        'status': 'PASS_COMPLETED',
        'matrix': str(MATRIX),
        'matrix_sha256': sha256(MATRIX),
        'n_patients': len(data.y),
        'n_domains': len(domains),
        'domain_counts': counts,
        'eligible_source_domains_n_ge_25': eligible,
        'feature_caps': FEATURE_CAPS,
        'downsample_n': DOWNSAMPLE_N,
        'downsample_repeats': DOWNSAMPLE_REPEATS,
        'model': {'type':'elastic_net_logistic_regression','C':C_VALUE,'l1_ratio':L1_RATIO,'class_weight':'balanced'},
        'reproduction_check_cap500': {
            'n_pairs': len(reproduction_diffs),
            'max_absolute_auc_difference_vs_frozen': float(np.max(reproduction_diffs)),
            'mean_absolute_auc_difference_vs_frozen': float(np.mean(reproduction_diffs)),
        },
        'runtime_seconds': time.time()-t0,
    }
    (OUT/'Analysis3_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2), flush=True)

if __name__ == '__main__':
    main()
