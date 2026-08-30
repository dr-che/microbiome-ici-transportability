#!/usr/bin/env python3
import sys, importlib.util, csv, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
MODULE_PATH=Path('/mnt/data/melanoma_transportability_review/analysis3_run.py')
spec=importlib.util.spec_from_file_location('a3',MODULE_PATH);a3=importlib.util.module_from_spec(spec);sys.modules['a3']=a3;spec.loader.exec_module(a3)
OUT=a3.OUT; data=a3.load_data(); domains=sorted(np.unique(data.domains)); counts={d:int(np.sum(data.domains==d)) for d in domains}; eligible=sorted([d for d,n in counts.items() if n>=a3.DOWNSAMPLE_N]);es=set(eligible)
full=[];pairs_all=[];summ=[];mat={};asym={}
for cap in a3.FEATURE_CAPS:
 rows=a3.run_full_cap(data,cap); full+=rows; pairs=a3.reciprocal_rows(rows)
 for r in pairs:r['feature_cap']=cap
 pairs_all+=pairs
 s=a3.summarize_directed(rows,cap,'all_9_sources_all_targets');s.update(a3.summarize_asym(pairs,cap,'all_9_sources_all_targets'));summ.append(s)
 ge=[r for r in rows if int(r['train_n'])>=a3.DOWNSAMPLE_N];gp=a3.reciprocal_rows(ge,es)
 s2=a3.summarize_directed(ge,cap,'source_n_ge_25_all_targets');s2.update(a3.summarize_asym(gp,cap,'source_n_ge_25_all_targets'));summ.append(s2)
 mat[cap]={(r['train_domain'],r['test_domain']):float(r['roc_auc']) for r in rows};asym[cap]={(r['domain_a'],r['domain_b']):float(r['absolute_delta_auc']) for r in pairs}
stability=[];keys=sorted(mat[500]);pkeys=sorted(asym[500])
for cap in a3.FEATURE_CAPS:
 x=np.array([mat[500][k] for k in keys]);y=np.array([mat[cap][k] for k in keys]);px=np.array([asym[500][k] for k in pkeys]);py=np.array([asym[cap][k] for k in pkeys])
 stability.append({'feature_cap':cap,'directed_auc_spearman_vs_cap500':float(spearmanr(x,y).statistic),'directed_auc_mean_absolute_difference_vs_cap500':float(np.mean(np.abs(x-y))),'reciprocal_asymmetry_spearman_vs_cap500':float(spearmanr(px,py).statistic),'reciprocal_asymmetry_mean_absolute_difference_vs_cap500':float(np.mean(np.abs(px-py)))})
# reproduce frozen
f=[]
with a3.FROZEN_METRICS.open('r',encoding='utf-8-sig',newline='') as fh:
 for r in csv.DictReader(fh,delimiter='\t'):
  if r['scenario']=='PRIMARY' and r['model']=='elastic_net':f.append(r)
fm={(r['train_domain'],r['test_domain']):float(r['roc_auc']) for r in f};diff=np.array([abs(mat[500][k]-fm[k]) for k in fm])
# read downsample checkpoints
rep=[];drows=[]
for cap in a3.FEATURE_CAPS:
 with (OUT/f'Analysis3_downsample_repeat_summary_cap{cap}.tsv').open('r',encoding='utf-8-sig',newline='') as fh:rep += list(csv.DictReader(fh,delimiter='\t'))
 with (OUT/f'Analysis3_downsample_directed_metrics_cap{cap}.tsv').open('r',encoding='utf-8-sig',newline='') as fh:drows += list(csv.DictReader(fh,delimiter='\t'))
agg=[];mets=['mean_roc_auc','median_roc_auc','mean_pr_auc','mean_brier','mean_abs_delta_auc','median_abs_delta_auc','pairs_ge_0_10','pairs_ge_0_20','max_abs_delta_auc']
for cap in a3.FEATURE_CAPS:
 sub=[r for r in rep if int(r['feature_cap'])==cap];row={'feature_cap':cap,'repeats':len(sub),'eligible_source_domains':len(eligible),'directed_pairs_per_repeat':int(float(sub[0]['directed_pairs'])),'reciprocal_pairs_per_repeat':int(float(sub[0]['reciprocal_pairs']))}
 for m in mets:
  v=np.array([float(r[m]) for r in sub]);lo,hi=a3.quantiles(v);row[f'{m}_mean']=float(np.mean(v));row[f'{m}_sd']=float(np.std(v,ddof=1));row[f'{m}_q025']=lo;row[f'{m}_q975']=hi
 agg.append(row)
dom=[]
for d in domains:
 idx=np.where(data.domains==d)[0];dom.append({'domain':d,'n':len(idx),'responders':int(data.y[idx].sum()),'non_responders':int(len(idx)-data.y[idx].sum()),'eligible_source_n_ge_25':int(d in es)})
a3.write_tsv(OUT/'Analysis3_domain_counts.tsv',dom,list(dom[0].keys()));a3.write_tsv(OUT/'Analysis3_feature_cap_directed_metrics.tsv',full,list(full[0].keys()));a3.write_tsv(OUT/'Analysis3_feature_cap_reciprocal_pairs.tsv',pairs_all,list(pairs_all[0].keys()));a3.write_tsv(OUT/'Analysis3_feature_cap_and_source_size_summary.tsv',summ,list(summ[0].keys()));a3.write_tsv(OUT/'Analysis3_feature_cap_stability_vs_500.tsv',stability,list(stability[0].keys()));a3.write_tsv(OUT/'Analysis3_downsample_repeat_summary.tsv',rep,list(rep[0].keys()));a3.write_tsv(OUT/'Analysis3_downsample_aggregate_summary.tsv',agg,list(agg[0].keys()));a3.write_tsv(OUT/'Analysis3_downsample_directed_metrics.tsv',drows,list(drows[0].keys()))
status={'analysis':'Analysis 3: feature-cap, source-n>=25, and source-size-matched sensitivity','status':'PASS_COMPLETED','matrix':str(a3.MATRIX),'matrix_sha256':a3.sha256(a3.MATRIX),'n_patients':len(data.y),'n_domains':len(domains),'domain_counts':counts,'eligible_source_domains_n_ge_25':eligible,'feature_caps':a3.FEATURE_CAPS,'downsample_n':a3.DOWNSAMPLE_N,'downsample_repeats':a3.DOWNSAMPLE_REPEATS,'model':{'type':'elastic_net_logistic_regression','C':a3.C_VALUE,'l1_ratio':a3.L1_RATIO,'class_weight':'balanced'},'reproduction_check_cap500':{'n_pairs':len(diff),'max_absolute_auc_difference_vs_frozen':float(np.max(diff)),'mean_absolute_auc_difference_vs_frozen':float(np.mean(diff))}}
(OUT/'Analysis3_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2));print('SUMMARY');
for r in summ:print(r)
print('AGG');
for r in agg:print(r)
print('STABILITY');
for r in stability:print(r)
