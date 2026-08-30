#!/usr/bin/env python3
import csv,json,math,hashlib,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT=Path('/mnt/data/melanoma_transportability_review/work/melanoma_microbiome_transportability')
OUT=Path('/mnt/data/melanoma_transportability_review/analysis4_outputs'); OUT.mkdir(exist_ok=True)
METRICS_FILE=ROOT/'results/step10_5B/directed_domain_transfer_metrics.tsv'
PRED_FILE=ROOT/'results/step10_5B/all_directed_transfer_predictions.tsv'
SEED=20260801; BOOT_REPS=1000; QAP_REPS=10000
MRQAP_PREDICTORS=['aitchison_centroid_distance','same_study_family','same_country','same_treatment_scope_group','absolute_response_rate_difference','log_train_n','log_test_n']

def read_tsv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write_tsv(p,rows,fields=None):
    rows=list(rows); fields=fields or list(rows[0].keys())
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def standardize(x):
    x=np.asarray(x,float);m=x.mean();s=x.std();s=s if s>1e-12 else 1.;return (x-m)/s

def pearson(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean();b=b-b.mean();d=np.sqrt((a*a).sum()*(b*b).sum());return float((a*b).sum()/d) if d else np.nan
def beta_r2(X,y):
    beta=np.linalg.pinv(X)@y;fit=X@beta;sst=((y-y.mean())**2).sum();return beta,float(1-((y-fit)**2).sum()/sst)
def fast_auc(y,p):
    pos=p[y==1];neg=p[y==0]
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))
def reciprocal_stats(A):
    v=np.array([abs(A[i,j]-A[j,i]) for i in range(A.shape[0]) for j in range(i+1,A.shape[0])])
    return dict(mean_abs_asymmetry=v.mean(),median_abs_asymmetry=np.median(v),pairs_ge_0_10=int((v>=.1).sum()),pairs_ge_0_20=int((v>=.2).sum()),max_abs_asymmetry=v.max(),reciprocal_pair_n=len(v))

start=time.time()
metrics=[r for r in read_tsv(METRICS_FILE) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
preds=[r for r in read_tsv(PRED_FILE) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
domains=sorted(set(r['train_domain'] for r in metrics)); di={d:i for i,d in enumerate(domains)}
pair_order=[(s,t) for s in domains for t in domains if s!=t]
metric_lookup={(r['train_domain'],r['test_domain']):r for r in metrics}
# predictions canonical order
by={}
for r in preds:by.setdefault((r['test_domain'],r['train_domain']),[]).append(r)
y_by={};prob={}
for t in domains:
    source=next(s for s in domains if s!=t)
    canon=sorted(by[(t,source)],key=lambda r:r['patient_id'])
    y_by[t]=np.array([int(r['true_label']) for r in canon])
    ids=[r['patient_id'] for r in canon]
    for s in domains:
        if s==t:continue
        rr=sorted(by[(t,s)],key=lambda r:r['patient_id']);assert [r['patient_id'] for r in rr]==ids
        prob[(s,t)]=np.array([float(r['predicted_probability']) for r in rr])
aitch=np.array([float(metric_lookup[p]['aitchison_centroid_distance']) for p in pair_order]); ar=rankdata(aitch)
samefam=np.array([float(metric_lookup[p]['same_study_family']) for p in pair_order])
xcols=[]
for c in MRQAP_PREDICTORS:xcols.append(standardize(np.array([float(metric_lookup[p][c]) for p in pair_order])))
Xmr=np.column_stack([np.ones(len(pair_order))]+xcols)

rng=np.random.default_rng(SEED+90000);boot=[]
for b in range(BOOT_REPS):
    ixmap={}
    for t in domains:
        y=y_by[t]; a=[]
        for c in (0,1):
            ix=np.where(y==c)[0];a.extend(rng.choice(ix,len(ix),replace=True))
        ixmap[t]=np.array(a,dtype=int)
    aucs=[];A=np.full((9,9),np.nan)
    for s,t in pair_order:
        ix=ixmap[t];v=fast_auc(y_by[t][ix],prob[(s,t)][ix]);aucs.append(v);A[di[s],di[t]]=v
    aucs=np.array(aucs);rs=reciprocal_stats(A);_,r2=beta_r2(Xmr,aucs)
    boot.append({'bootstrap_id':b+1,'mean_directed_auc':aucs.mean(),'median_directed_auc':np.median(aucs),**rs,
      'aitchison_spearman_rho':pearson(ar,rankdata(aucs)),
      'same_family_mean_auc_difference':aucs[samefam==1].mean()-aucs[samefam==0].mean(),'mrqap_r_squared':r2})
write_tsv(OUT/'Analysis4_target_patient_bootstrap_distribution.tsv',boot)
obs=np.array([float(metric_lookup[p]['roc_auc']) for p in pair_order]);A=np.full((9,9),np.nan)
for (s,t),v in zip(pair_order,obs):A[di[s],di[t]]=v
rs=reciprocal_stats(A);_,r2=beta_r2(Xmr,obs)
obsmap={'mean_directed_auc':obs.mean(),'median_directed_auc':np.median(obs),**rs,'aitchison_spearman_rho':pearson(ar,rankdata(obs)),'same_family_mean_auc_difference':obs[samefam==1].mean()-obs[samefam==0].mean(),'mrqap_r_squared':r2}
summary=[]
for k in ['mean_directed_auc','median_directed_auc','mean_abs_asymmetry','median_abs_asymmetry','pairs_ge_0_10','pairs_ge_0_20','max_abs_asymmetry','aitchison_spearman_rho','same_family_mean_auc_difference','mrqap_r_squared']:
    v=np.array([float(r[k]) for r in boot]);summary.append({'metric':k,'observed':obsmap[k],'bootstrap_mean':v.mean(),'bootstrap_median':np.median(v),'bootstrap_sd':v.std(ddof=1),'percentile_2_5':np.quantile(v,.025),'percentile_97_5':np.quantile(v,.975),'bootstrap_repetitions':BOOT_REPS,'uncertainty_scope':'Target-patient sampling uncertainty with frozen source models'})
write_tsv(OUT/'Analysis4_target_patient_bootstrap_summary.tsv',summary)
# Claim and response tables
claim_rows=[
 {'manuscript_location':'Title/Abstract','current_or_prior_claim':'Direction-dependent gut microbiome associations limit transportability','evidence_after_analyses_1_to_4':'Reciprocal asymmetry P = 0.066 under the domain-preserving null and was sensitive to feature cap and source size','risk':'High','required_revision':'Remove direction-dependent from the title; describe reciprocal differences as substantial but descriptive and specification-sensitive'},
 {'manuscript_location':'Abstract','current_or_prior_claim':'Publication provenance and transfer direction are key constraints','evidence_after_analyses_1_to_4':'Only the five-site Lee family changes between LODO and family holdout; four other families have identical partitions','risk':'High','required_revision':'State that estimates were sensitive to joint versus separate withholding of related Lee sites; do not infer a general publication-family effect'},
 {'manuscript_location':'Results—directed transfer','current_or_prior_claim':'Transportability was directionally asymmetric','evidence_after_analyses_1_to_4':'Observed mean |ΔAUC| = 0.134, but null P = 0.066 and source-size matching reduced it to 0.094','risk':'Moderate','required_revision':'Use descriptive asymmetry and report the structured-null and feature/source-size sensitivities'},
 {'manuscript_location':'Results—QAP/MRQAP','current_or_prior_claim':'Measured domain features did not explain transferability','evidence_after_analyses_1_to_4':'No full-network predictor was significant; node-deletion estimates are unstable because the network contains nine nodes','risk':'Moderate','required_revision':'State that no robust association was detected and that power and node-level precision were limited'},
 {'manuscript_location':'Methods/Statistics','current_or_prior_claim':'363-patient analysis','evidence_after_analyses_1_to_4':'Patients support prediction estimation, while transportability inference is constrained by nine domains and five publication families','risk':'Moderate','required_revision':'Explicitly distinguish patient-level sample size from domain- and family-level inferential units'},
 {'manuscript_location':'Discussion','current_or_prior_claim':'The biological signal changes with clinical and data-generation context','evidence_after_analyses_1_to_4':'Observed estimates may combine biological heterogeneity, technical variation, and high-dimensional model instability','risk':'Moderate','required_revision':'Replace biological signal changes with estimated associations varied across contexts'},
 {'manuscript_location':'Conclusion','current_or_prior_claim':'No universal biomarker exists','evidence_after_analyses_1_to_4':'No transferable signal was supported in the evaluated datasets and methods; universal absence cannot be proven','risk':'Moderate','required_revision':'Limit the conclusion to the evaluated species profiles, cohorts, and validation framework'},]
write_tsv(OUT/'Analysis4_manuscript_claim_audit.tsv',claim_rows)
response=[
 {'anticipated_reviewer_question':'Why is n = 363 not the effective sample size for transportability inference?','evidence_based_response':'The 363 patients contribute probability estimates, but external transportability is evaluated across nine non-exchangeable domain nodes and five publication families.','action_in_manuscript':'Add effective-unit paragraph and Supplementary audit table.'},
 {'anticipated_reviewer_question':'Are 72 directed transfers independent observations?','evidence_based_response':'No. Each domain appears repeatedly as source and target. QAP/MRQAP, domain-preserving permutations, and leave-one-node-out analyses respect this dependency.','action_in_manuscript':'Avoid conventional regression language based on 72 degrees of freedom.'},
 {'anticipated_reviewer_question':'Does the study prove a publication-family effect?','evidence_based_response':'No. Four families contain one domain and produce identical LODO and family-out partitions. The difference is concentrated in the five-site Lee family.','action_in_manuscript':'Describe a Lee-family joint-withholding sensitivity rather than a general effect.'},
 {'anticipated_reviewer_question':'Is reciprocal asymmetry a biological property?','evidence_based_response':'Not established. Mean asymmetry did not exceed the domain-preserving null at P < 0.05 and decreased after source-size matching.','action_in_manuscript':'Use descriptive, specification-sensitive asymmetry wording.'},
 {'anticipated_reviewer_question':'What uncertainty does the new bootstrap measure?','evidence_based_response':'It resamples target patients within response strata while holding fitted source models fixed. It quantifies evaluation-sample uncertainty, not source-training or between-domain uncertainty.','action_in_manuscript':'Label intervals as target-patient sampling uncertainty.'}]
write_tsv(OUT/'Analysis4_reviewer_response_table.tsv',response)
status={'analysis':'Analysis4_effective_independence_uncertainty_inference_boundary','created_utc':'2026-08-01T15:00:00Z','seed':SEED,'bootstrap_repetitions':BOOT_REPS,'qap_repetitions':QAP_REPS,'patient_n':363,'domain_n':9,'publication_family_n':5,'directed_pair_n':72,'reciprocal_pair_n':36,'target_bootstrap_scope':'Target-patient sampling uncertainty with source models frozen','metrics_sha256':hashlib.sha256(METRICS_FILE.read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256(PRED_FILE.read_bytes()).hexdigest(),'continuation_runtime_seconds':time.time()-start}
(OUT/'Analysis4_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
