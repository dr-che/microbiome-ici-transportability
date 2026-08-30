#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, hashlib, time
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path('/mnt/data/melanoma_transportability_review/work/melanoma_microbiome_transportability')
OUT = Path('/mnt/data/melanoma_transportability_review/analysis4_outputs')
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260801
BOOT_REPS = 1000
QAP_REPS = 10000

METRICS_FILE = ROOT / 'results/step10_5B/directed_domain_transfer_metrics.tsv'
PRED_FILE = ROOT / 'results/step10_5B/all_directed_transfer_predictions.tsv'
ORIG_QAP_FILE = ROOT / 'results/step10_5B/univariable_QAP_associations.tsv'
ORIG_MRQAP_FILE = ROOT / 'results/step10_5B/primary_elastic_net_MRQAP.tsv'

DISTANCE_COLUMNS = [
    'aitchison_centroid_distance',
    'bray_curtis_centroid_distance',
    'jensen_shannon_distance',
    'prevalence_l1_distance',
]
BINARY_RELATIONS = [
    'same_study_family',
    'same_country',
    'same_macro_region',
    'same_treatment_scope_group',
]
MRQAP_PREDICTORS = [
    'aitchison_centroid_distance',
    'same_study_family',
    'same_country',
    'same_treatment_scope_group',
    'absolute_response_rate_difference',
    'log_train_n',
    'log_test_n',
]

def read_tsv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def write_tsv(path: Path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def fnum(x):
    try:
        v=float(x); return v if math.isfinite(v) else np.nan
    except Exception: return np.nan

def build_matrix(rows, domains, col):
    idx={d:i for i,d in enumerate(domains)}
    M=np.full((len(domains),len(domains)),np.nan)
    for r in rows:
        a,b=r['train_domain'],r['test_domain']
        if a in idx and b in idx:
            M[idx[a],idx[b]]=fnum(r[col])
    return M

def reciprocal_stats(auc_matrix):
    n=auc_matrix.shape[0]
    vals=[]
    for i in range(n):
        for j in range(i+1,n):
            vals.append(abs(auc_matrix[i,j]-auc_matrix[j,i]))
    v=np.asarray(vals,float)
    return {
        'mean_abs_asymmetry': float(v.mean()),
        'median_abs_asymmetry': float(np.median(v)),
        'pairs_ge_0_10': int((v>=0.10).sum()),
        'pairs_ge_0_20': int((v>=0.20).sum()),
        'max_abs_asymmetry': float(v.max()),
        'reciprocal_pair_n': int(len(v)),
    }

def summarize_network(rows, domains):
    auc=build_matrix(rows,domains,'roc_auc')
    mask=~np.eye(len(domains),dtype=bool)
    y=auc[mask]
    out={
        'node_n': len(domains),
        'directed_pair_n': int(mask.sum()),
        'mean_directed_auc': float(np.mean(y)),
        'median_directed_auc': float(np.median(y)),
        'directed_auc_sd': float(np.std(y,ddof=1)),
    }
    out.update(reciprocal_stats(auc))
    return out

def pearson_corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a-a.mean(); b=b-b.mean()
    den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/den) if den>0 else np.nan

def qap_for_subset(rows, domains, reps, seed):
    n=len(domains); mask=~np.eye(n,dtype=bool)
    auc=build_matrix(rows,domains,'roc_auc'); y=auc[mask]
    y_rank=rankdata(y,method='average')
    rng=np.random.default_rng(seed)
    perms=np.asarray([rng.permutation(n) for _ in range(reps)],dtype=int)
    output=[]
    for k,col in enumerate(DISTANCE_COLUMNS+BINARY_RELATIONS):
        X=build_matrix(rows,domains,col)
        x=X[mask]
        if col in DISTANCE_COLUMNS:
            xr=rankdata(x,method='average')
            obs=pearson_corr(xr,y_rank)
            null=np.empty(reps,float)
            XR=np.full_like(X,np.nan)
            XR[mask]=xr
            for b,p in enumerate(perms):
                null[b]=pearson_corr(XR[np.ix_(p,p)][mask],y_rank)
            p2=(1+np.sum(np.abs(null)>=abs(obs)))/(1+reps)
            pd=(1+np.sum(null<=obs))/(1+reps)
            stype='spearman_rho'; direction='negative'
        else:
            obs=float(y[x==1].mean()-y[x==0].mean()) if np.any(x==1) and np.any(x==0) else np.nan
            null=np.empty(reps,float)
            for b,p in enumerate(perms):
                xp=X[np.ix_(p,p)][mask]
                null[b]=float(y[xp==1].mean()-y[xp==0].mean()) if np.any(xp==1) and np.any(xp==0) else np.nan
            valid=np.isfinite(null); nv=null[valid]
            p2=(1+np.sum(np.abs(nv)>=abs(obs)))/(1+len(nv)) if math.isfinite(obs) else np.nan
            pd=(1+np.sum(nv>=obs))/(1+len(nv)) if math.isfinite(obs) else np.nan
            stype='mean_auc_relation_1_minus_0'; direction='positive'
        output.append({
            'predictor':col,'statistic_type':stype,'prespecified_direction':direction,
            'observed_statistic':obs,'qap_p_two_sided':p2,
            'qap_p_prespecified_direction':pd,'permutations':reps,
        })
    return output

def standardize(x):
    x=np.asarray(x,float); m=float(np.mean(x)); s=float(np.std(x))
    if s<1e-12: s=1.0
    return (x-m)/s,m,s

def ols_beta_r2(X,y):
    beta=np.linalg.pinv(X)@y
    fitted=X@beta
    ssr=float(np.sum((y-fitted)**2)); sst=float(np.sum((y-y.mean())**2))
    r2=1-ssr/sst if sst>0 else np.nan
    return beta,r2,float(np.linalg.cond(X))

def mrqap_for_subset(rows,domains,reps,seed):
    n=len(domains); mask=~np.eye(n,dtype=bool)
    Y=build_matrix(rows,domains,'roc_auc'); y=Y[mask]
    mats={c:build_matrix(rows,domains,c) for c in MRQAP_PREDICTORS}
    cols=[]; scaling={}
    for c in MRQAP_PREDICTORS:
        z,m,s=standardize(mats[c][mask]); cols.append(z); scaling[c]=(m,s)
    X=np.column_stack([np.ones(len(y))]+cols)
    beta,r2,cond=ols_beta_r2(X,y)
    rng=np.random.default_rng(seed)
    null=np.empty((reps,len(beta)),float); null_r2=np.empty(reps,float)
    for b in range(reps):
        p=rng.permutation(n); pcols=[]
        for c in MRQAP_PREDICTORS:
            m,s=scaling[c]
            pcols.append((mats[c][np.ix_(p,p)][mask]-m)/s)
        Xp=np.column_stack([np.ones(len(y))]+pcols)
        bb,rr,_=ols_beta_r2(Xp,y); null[b]=bb; null_r2[b]=rr
    names=['intercept']+MRQAP_PREDICTORS
    rowsout=[]
    for j,name in enumerate(names):
        p2=(1+np.sum(np.abs(null[:,j])>=abs(beta[j])))/(1+reps)
        if name in DISTANCE_COLUMNS:
            pd=(1+np.sum(null[:,j]<=beta[j]))/(1+reps); direction='negative'
        elif name in BINARY_RELATIONS:
            pd=(1+np.sum(null[:,j]>=beta[j]))/(1+reps); direction='positive'
        else:
            pd=np.nan; direction='not_prespecified'
        rowsout.append({'term':name,'standardized_coefficient':beta[j],
                        'mrqap_p_two_sided':p2,'mrqap_p_prespecified_direction':pd,
                        'prespecified_direction':direction,'permutations':reps})
    diag={'r_squared':r2,'condition_number':cond,
          'r2_permutation_p':(1+np.sum(null_r2>=r2))/(1+reps),
          'node_n':n,'directed_pair_n':len(y)}
    return rowsout,diag

start=time.time()
metrics=[r for r in read_tsv(METRICS_FILE) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
preds=[r for r in read_tsv(PRED_FILE) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
domains=sorted(set(r['train_domain'] for r in metrics)|set(r['test_domain'] for r in metrics))
assert len(metrics)==72 and len(domains)==9

# Effective unit audit
unit_rows=[
 {'analysis_component':'Primary nested LODO prediction','reported_observation_n':'363 patients','effective_inference_unit':'9 clinical domains','dependency_structure':'Patients are nested within held-out domains','appropriate_uncertainty':'Domain-cluster bootstrap plus domain-specific metrics','permitted_claim':'Performance across the nine observed domains','prohibited_overinterpretation':'363 independent external validations'},
 {'analysis_component':'Fixed-complexity LODO structured null','reported_observation_n':'363 predictions','effective_inference_unit':'9 clinical domains','dependency_structure':'Predictions arise from nine overlapping training sets','appropriate_uncertainty':'Within-domain label permutation','permitted_claim':'Observed pooled discrimination did not exceed the domain-preserving null','prohibited_overinterpretation':'Patient-level independence of cross-validation predictions'},
 {'analysis_component':'Complete publication-family holdout','reported_observation_n':'363 predictions','effective_inference_unit':'5 publication families','dependency_structure':'Four families contain one domain; only Lee contains five domains','appropriate_uncertainty':'Family-level raw estimates and descriptive/jackknife sensitivity','permitted_claim':'Joint Lee-family holdout changed the performance estimate','prohibited_overinterpretation':'A general publication-family penalty estimated from five equivalent contrasts'},
 {'analysis_component':'Directed source-to-target transfer','reported_observation_n':'72 directed transfers','effective_inference_unit':'9 domain nodes','dependency_structure':'Each domain appears repeatedly as source and target','appropriate_uncertainty':'QAP/MRQAP and node-deletion sensitivity','permitted_claim':'Observed transfer matrix was heterogeneous and weak on average','prohibited_overinterpretation':'72 independent external validation experiments'},
 {'analysis_component':'Reciprocal asymmetry','reported_observation_n':'36 reciprocal pairs','effective_inference_unit':'9 domain nodes','dependency_structure':'Pairs share domain nodes and fitted source models','appropriate_uncertainty':'Domain-preserving permutation and source-size/feature-cap sensitivity','permitted_claim':'Substantial descriptive reciprocal differences were observed','prohibited_overinterpretation':'A statistically established direction-dependent biological effect'},
 {'analysis_component':'Domain-level QAP/MRQAP','reported_observation_n':'72 directed dyads','effective_inference_unit':'9-node network','dependency_structure':'Dyadic outcomes are node-dependent','appropriate_uncertainty':'Joint node-label permutations and leave-one-node-out stability','permitted_claim':'Measured domain characteristics did not robustly explain transfer variation','prohibited_overinterpretation':'Conventional regression degrees of freedom based on 72 rows'},
 {'analysis_component':'Species random-effects meta-analysis','reported_observation_n':'363 patients','effective_inference_unit':'9 domains; 5 families in sensitivity','dependency_structure':'Patient effects are summarized within domains before pooling','appropriate_uncertainty':'Random-effects meta-analysis and family sensitivity','permitted_claim':'No FDR-supported cross-domain species association','prohibited_overinterpretation':'363 independent studies or universal absence of an effect'},
 {'analysis_component':'V1 pathway sensitivity analysis','reported_observation_n':'276 patients','effective_inference_unit':'5 studies','dependency_structure':'Patients are nested within five studies and a separate processing system','appropriate_uncertainty':'Five-study random-effects sensitivity analysis','permitted_claim':'No more stable functional layer was identified in this sensitivity dataset','prohibited_overinterpretation':'Primary V2 multi-omics integration or universal functional null'},
]
write_tsv(OUT/'Analysis4_effective_independence_audit.tsv',unit_rows)

# Full-network and domain jackknife summaries, QAP and MRQAP
full_summary=summarize_network(metrics,domains); full_summary.update({'analysis':'FULL_9_DOMAIN_NETWORK','removed_domain':''})
jack_summ=[]; jack_qap=[]; jack_mrqap=[]; jack_diag=[]
for di,removed in enumerate(domains):
    keep=[d for d in domains if d!=removed]
    sub=[r for r in metrics if r['train_domain'] in keep and r['test_domain'] in keep]
    sm=summarize_network(sub,keep); sm.update({'analysis':'LEAVE_ONE_DOMAIN_OUT_NETWORK','removed_domain':removed}); jack_summ.append(sm)
    qrows=qap_for_subset(sub,keep,QAP_REPS,SEED+1000+di*100)
    for r in qrows: r.update({'removed_domain':removed,'node_n':len(keep)})
    jack_qap.extend(qrows)
    mrows,diag=mrqap_for_subset(sub,keep,QAP_REPS,SEED+5000+di*100)
    for r in mrows: r.update({'removed_domain':removed,'node_n':len(keep)})
    jack_mrqap.extend(mrows)
    diag.update({'removed_domain':removed}); jack_diag.append(diag)
write_tsv(OUT/'Analysis4_domain_jackknife_summary.tsv',[full_summary]+jack_summ)
write_tsv(OUT/'Analysis4_domain_jackknife_QAP.tsv',jack_qap)
write_tsv(OUT/'Analysis4_domain_jackknife_MRQAP.tsv',jack_mrqap)
write_tsv(OUT/'Analysis4_domain_jackknife_MRQAP_diagnostics.tsv',jack_diag)

# Aggregate jackknife stability
orig_qap=[r for r in read_tsv(ORIG_QAP_FILE) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
orig_qap_map={r['predictor']:r for r in orig_qap}
qap_stability=[]
for pred in DISTANCE_COLUMNS+BINARY_RELATIONS:
    vals=[r for r in jack_qap if r['predictor']==pred]
    stats=np.array([float(r['observed_statistic']) for r in vals])
    p2=np.array([float(r['qap_p_two_sided']) for r in vals])
    pd=np.array([float(r['qap_p_prespecified_direction']) for r in vals])
    orig=float(orig_qap_map[pred]['observed_statistic'])
    qap_stability.append({
      'predictor':pred,'full_network_statistic':orig,'jackknife_min_statistic':stats.min(),
      'jackknife_max_statistic':stats.max(),'jackknife_median_statistic':np.median(stats),
      'sign_concordant_with_full_n':int(np.sum(np.sign(stats)==np.sign(orig))),
      'jackknife_n':len(stats),'min_two_sided_p':p2.min(),'max_two_sided_p':p2.max(),
      'min_directional_p':pd.min(),'max_directional_p':pd.max(),
      'any_two_sided_p_lt_0_05':bool(np.any(p2<0.05)),
      'any_directional_p_lt_0_05':bool(np.any(pd<0.05)),
    })
write_tsv(OUT/'Analysis4_QAP_jackknife_stability_summary.tsv',qap_stability)

orig_mrqap=read_tsv(ORIG_MRQAP_FILE); orig_m_map={r['term']:r for r in orig_mrqap}
mr_stability=[]
for term in ['intercept']+MRQAP_PREDICTORS:
    vals=[r for r in jack_mrqap if r['term']==term]
    co=np.array([float(r['standardized_coefficient']) for r in vals]); p2=np.array([float(r['mrqap_p_two_sided']) for r in vals])
    orig=float(orig_m_map[term]['standardized_coefficient'])
    mr_stability.append({
      'term':term,'full_network_coefficient':orig,'jackknife_min_coefficient':co.min(),
      'jackknife_max_coefficient':co.max(),'jackknife_median_coefficient':np.median(co),
      'sign_concordant_with_full_n':int(np.sum(np.sign(co)==np.sign(orig))),
      'jackknife_n':len(co),'min_two_sided_p':p2.min(),'max_two_sided_p':p2.max(),
      'any_two_sided_p_lt_0_05':bool(np.any(p2<0.05)),
    })
write_tsv(OUT/'Analysis4_MRQAP_jackknife_stability_summary.tsv',mr_stability)

# Prepare patient prediction arrays keyed by target and source
pred_lookup={}
for r in preds:
    key=(r['test_domain'],r['train_domain'])
    pred_lookup.setdefault(key,[]).append(r)
# Canonical patient order per target
patient_by_target={}
for target in domains:
    base_rows=pred_lookup[(target,next(s for s in domains if s!=target))]
    canon=sorted([(r['patient_id'],int(r['true_label'])) for r in base_rows])
    patient_by_target[target]=canon
    for source in domains:
        if source==target: continue
        rows=sorted(pred_lookup[(target,source)], key=lambda r:r['patient_id'])
        assert [(r['patient_id'],int(r['true_label'])) for r in rows]==canon

prob_by_pair={}
for (target,source),rows in pred_lookup.items():
    rows=sorted(rows,key=lambda r:r['patient_id'])
    prob_by_pair[(source,target)]=np.array([float(r['predicted_probability']) for r in rows])
y_by_target={t:np.array([y for _,y in patient_by_target[t]],int) for t in domains}

# Predictor vectors for descriptive bootstrap associations
metric_lookup={(r['train_domain'],r['test_domain']):r for r in metrics}
pair_order=[(a,b) for a in domains for b in domains if a!=b]
aitch=np.array([float(metric_lookup[p]['aitchison_centroid_distance']) for p in pair_order])
samefam=np.array([float(metric_lookup[p]['same_study_family']) for p in pair_order])
# MRQAP fixed X for descriptive target-sampling bootstrap
xcols=[]
for c in MRQAP_PREDICTORS:
    vv=np.array([float(metric_lookup[p][c]) for p in pair_order]); z,_,_=standardize(vv); xcols.append(z)
Xmr=np.column_stack([np.ones(len(pair_order))]+xcols)

rng=np.random.default_rng(SEED+90000)
boot=[]
for b in range(BOOT_REPS):
    sample_idx={}
    for t in domains:
        y=y_by_target[t]; inds=[]
        for cls in [0,1]:
            ix=np.where(y==cls)[0]
            inds.extend(rng.choice(ix,size=len(ix),replace=True).tolist())
        sample_idx[t]=np.array(inds,int)
    aucs={}
    for source,target in pair_order:
        ix=sample_idx[target]; y=y_by_target[target][ix]; p=prob_by_pair[(source,target)][ix]
        aucs[(source,target)]=roc_auc_score(y,p)
    yauc=np.array([aucs[p] for p in pair_order])
    A=np.full((len(domains),len(domains)),np.nan); idx={d:i for i,d in enumerate(domains)}
    for (s,t),v in aucs.items(): A[idx[s],idx[t]]=v
    rs=reciprocal_stats(A)
    beta,r2,_=ols_beta_r2(Xmr,yauc)
    br={
      'bootstrap_id':b+1,'mean_directed_auc':float(yauc.mean()),'median_directed_auc':float(np.median(yauc)),
      **rs,
      'aitchison_spearman_rho':pearson_corr(rankdata(aitch),rankdata(yauc)),
      'same_family_mean_auc_difference':float(yauc[samefam==1].mean()-yauc[samefam==0].mean()),
      'mrqap_r_squared':r2,
    }
    boot.append(br)
write_tsv(OUT/'Analysis4_target_patient_bootstrap_distribution.tsv',boot)

# Bootstrap summary with observed fixed-prediction statistic
obs_auc=np.array([float(metric_lookup[p]['roc_auc']) for p in pair_order])
obsA=build_matrix(metrics,domains,'roc_auc'); obsrs=reciprocal_stats(obsA)
obs_beta,obs_r2,_=ols_beta_r2(Xmr,obs_auc)
obsmap={
 'mean_directed_auc':float(obs_auc.mean()),'median_directed_auc':float(np.median(obs_auc)),
 **obsrs,
 'aitchison_spearman_rho':pearson_corr(rankdata(aitch),rankdata(obs_auc)),
 'same_family_mean_auc_difference':float(obs_auc[samefam==1].mean()-obs_auc[samefam==0].mean()),
 'mrqap_r_squared':obs_r2,
}
boot_summary=[]
for key in ['mean_directed_auc','median_directed_auc','mean_abs_asymmetry','median_abs_asymmetry','pairs_ge_0_10','pairs_ge_0_20','max_abs_asymmetry','aitchison_spearman_rho','same_family_mean_auc_difference','mrqap_r_squared']:
    v=np.array([float(r[key]) for r in boot])
    boot_summary.append({'metric':key,'observed':obsmap[key],'bootstrap_mean':v.mean(),'bootstrap_median':np.median(v),
                         'bootstrap_sd':v.std(ddof=1),'percentile_2_5':np.quantile(v,.025),'percentile_97_5':np.quantile(v,.975),
                         'bootstrap_repetitions':BOOT_REPS,'uncertainty_scope':'Target-patient sampling uncertainty with frozen source models'})
write_tsv(OUT/'Analysis4_target_patient_bootstrap_summary.tsv',boot_summary)

# Claim audit and reviewer response
claim_rows=[
 {'manuscript_location':'Title/Abstract','current_or_prior_claim':'Direction-dependent gut microbiome associations limit transportability','evidence_after_analyses_1_to_4':'Reciprocal asymmetry P = 0.066 under the domain-preserving null and was sensitive to feature cap and source size','risk':'High','required_revision':'Remove direction-dependent from the title; describe reciprocal differences as substantial but descriptive and specification-sensitive'},
 {'manuscript_location':'Abstract','current_or_prior_claim':'Publication provenance and transfer direction are key constraints','evidence_after_analyses_1_to_4':'Only the five-site Lee family changes between LODO and family holdout; four other families have identical partitions','risk':'High','required_revision':'State that estimates were sensitive to joint versus separate withholding of related Lee sites; do not infer a general publication-family effect'},
 {'manuscript_location':'Results—directed transfer','current_or_prior_claim':'Transportability was directionally asymmetric','evidence_after_analyses_1_to_4':'Observed mean |ΔAUC| = 0.134, but null P = 0.066 and source-size matching reduced it to 0.094','risk':'Moderate','required_revision':'Use descriptive asymmetry and report the structured-null and feature/source-size sensitivities'},
 {'manuscript_location':'Results—QAP/MRQAP','current_or_prior_claim':'Measured domain features did not explain transferability','evidence_after_analyses_1_to_4':'No full-network predictor was significant; node-deletion estimates are unstable because the network contains nine nodes','risk':'Moderate','required_revision':'State that no robust association was detected and that power and node-level precision were limited'},
 {'manuscript_location':'Methods/Statistics','current_or_prior_claim':'363-patient analysis','evidence_after_analyses_1_to_4':'Patients support prediction estimation, while transportability inference is constrained by nine domains and five publication families','risk':'Moderate','required_revision':'Explicitly distinguish patient-level sample size from domain- and family-level inferential units'},
 {'manuscript_location':'Discussion','current_or_prior_claim':'The biological signal changes with clinical and data-generation context','evidence_after_analyses_1_to_4':'Observed estimates may combine biological heterogeneity, technical variation, and high-dimensional model instability','risk':'Moderate','required_revision':'Replace biological signal changes with estimated associations varied across contexts'},
 {'manuscript_location':'Conclusion','current_or_prior_claim':'No universal biomarker exists','evidence_after_analyses_1_to_4':'No transferable signal was supported in the evaluated datasets and methods; universal absence cannot be proven','risk':'Moderate','required_revision':'Limit the conclusion to the evaluated species profiles, cohorts, and validation framework'},
]
write_tsv(OUT/'Analysis4_manuscript_claim_audit.tsv',claim_rows)
review_rows=[
 {'anticipated_reviewer_question':'Why is n = 363 not the effective sample size for transportability inference?','evidence_based_response':'The 363 patients contribute probability estimates, but external transportability is evaluated across nine non-exchangeable domain nodes and five publication families. The manuscript now separates these levels explicitly.','action_in_manuscript':'Add effective-unit paragraph and Supplementary audit table.'},
 {'anticipated_reviewer_question':'Are 72 directed transfers independent observations?','evidence_based_response':'No. Each domain appears repeatedly as source and target. QAP/MRQAP, domain-preserving permutations, and leave-one-node-out analyses are used to respect this dependency.','action_in_manuscript':'Avoid conventional regression language based on 72 degrees of freedom.'},
 {'anticipated_reviewer_question':'Does the study prove a publication-family effect?','evidence_based_response':'No. Four families contain one domain and produce identical LODO and family-out partitions. The difference is concentrated in the five-site Lee family.','action_in_manuscript':'Describe a Lee-family joint-withholding sensitivity rather than a general effect.'},
 {'anticipated_reviewer_question':'Is reciprocal asymmetry a biological property?','evidence_based_response':'Not established. The observed mean asymmetry did not exceed the domain-preserving null at P < 0.05 and decreased after source-size matching.','action_in_manuscript':'Use descriptive, specification-sensitive asymmetry wording.'},
 {'anticipated_reviewer_question':'What uncertainty does the new bootstrap measure?','evidence_based_response':'It resamples target patients within response strata while holding fitted source models fixed. It quantifies evaluation-sample uncertainty, not source-training or between-domain uncertainty.','action_in_manuscript':'Label the interval as target-patient sampling uncertainty and retain domain-level inference limits.'},
]
write_tsv(OUT/'Analysis4_reviewer_response_table.tsv',review_rows)

# Status
status={
 'analysis':'Analysis4_effective_independence_uncertainty_inference_boundary',
 'created_utc':'2026-08-01T15:00:00Z','seed':SEED,'bootstrap_repetitions':BOOT_REPS,'qap_repetitions':QAP_REPS,
 'patient_n':363,'domain_n':9,'publication_family_n':5,'directed_pair_n':72,'reciprocal_pair_n':36,
 'target_bootstrap_scope':'Target-patient sampling uncertainty with source models frozen',
 'metrics_file':str(METRICS_FILE),'prediction_file':str(PRED_FILE),
 'metrics_sha256':hashlib.sha256(METRICS_FILE.read_bytes()).hexdigest(),
 'predictions_sha256':hashlib.sha256(PRED_FILE.read_bytes()).hexdigest(),
 'runtime_seconds':time.time()-start,
}
(OUT/'Analysis4_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
