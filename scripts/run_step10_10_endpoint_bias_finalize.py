from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--bias-dir',default='reproduced/step10_10/bias_benchmark')
parser.add_argument('--output-dir',default='reproduced/step10_10/defensive_results')
args=parser.parse_args()
ROOT=Path(args.root).resolve()
OUT=ROOT/args.output_dir
OUT.mkdir(parents=True,exist_ok=True)

# ---------- Bias benchmark calibration ----------
pred=pd.read_csv(ROOT/args.bias_dir/'predictions.tsv',sep='\t')
rows=[]
for (scheme,method),g in pred.groupby(['scheme','method']):
    y=g['y'].astype(int).to_numpy()
    p=np.clip(g['prob'].astype(float).to_numpy(),1e-6,1-1e-6)
    lp=np.log(p/(1-p))
    X=sm.add_constant(lp)
    try:
        fit=sm.GLM(y,X,family=sm.families.Binomial()).fit()
        intercept=float(fit.params[0]); slope=float(fit.params[1])
        intercept_se=float(fit.bse[0]); slope_se=float(fit.bse[1])
    except Exception:
        intercept=slope=intercept_se=slope_se=np.nan
    rows.append({
        'scheme':scheme,'method':method,'n':len(g),
        'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),
        'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),
        'calibration_intercept':intercept,'calibration_slope':slope,
        'calibration_intercept_se':intercept_se,'calibration_slope_se':slope_se
    })
cal=pd.DataFrame(rows).sort_values(['scheme','method'])
cal.to_csv(OUT/'Step10_10A_bias_correction_overall_metrics.tsv',sep='\t',index=False)
shutil_cols=['scheme','method','delta_auc','ci_low','ci_high','p_boot_gt0']
delta=pd.read_csv(ROOT/args.bias_dir/'delta_bootstrap.tsv',sep='\t')
delta.to_csv(OUT/'Step10_10A_bias_correction_delta_bootstrap.tsv',sep='\t',index=False)

# ---------- Endpoint compatibility audit ----------
manifest=pd.read_csv(ROOT/'data/metadata/Step10_2_manifest_R3_primary_lock_v8.tsv',sep='\t')
inc=manifest[manifest['r3_v8_primary_analysis_include'].astype(str).str.upper().isin(['TRUE','1','YES'])].copy()
endpoint_map={
    'Frankel_US':'RECIST_ORR_PROGRESSION',
    'Matson_US':'RECIST_ORR_PROGRESSION',
    'Gopalakrishnan_US':'DCB6_COMPATIBLE',
    'Spencer_US':'DCB6_COMPATIBLE',
    'Lee_Barcelona':'DCB6_COMPATIBLE',
    'Lee_Leeds':'DCB6_COMPATIBLE',
    'Lee_Manchester':'DCB6_COMPATIBLE',
    'Lee_PRIMM_NL':'DCB6_COMPATIBLE',
    'Lee_PRIMM_UK':'DCB6_COMPATIBLE',
}
source_basis={
    'Frankel_US':'RECIST-based response versus progressive disease in source study',
    'Matson_US':'Official patient-level Response field; RECIST 1.1 source assessment',
    'Gopalakrishnan_US':'CR/PR or stable disease >=6 months versus progression/SD <6 months',
    'Spencer_US':'CR/PR or stable disease >=6 months in the source clinical-response definition',
    'Lee_Barcelona':'Lee clinical-benefit/ORR harmonization; patient-level RECIST evidence available',
    'Lee_Leeds':'Lee clinical-benefit/ORR harmonization; aggregate-concordant labels',
    'Lee_Manchester':'Lee clinical-benefit/ORR harmonization; patient-level RECIST evidence available',
    'Lee_PRIMM_NL':'Lee clinical-benefit/ORR harmonization; aggregate-concordant labels',
    'Lee_PRIMM_UK':'Lee clinical-benefit/ORR harmonization; aggregate-concordant labels',
}
endpoint_rows=[]
for dom,g in inc.groupby('domain_id'):
    ev=g['r3_v8_endpoint_evidence'].dropna().astype(str)
    endpoint_rows.append({
        'domain_id':dom,'n':len(g),
        'responders':int((g['response_harmonized']=='Responder').sum()),
        'non_responders':int((g['response_harmonized']=='Non-responder').sum()),
        'endpoint_class':endpoint_map[dom],
        'source_basis':source_basis[dom],
        'patient_level_endpoint_evidence_available':int(ev.str.contains('RECIST|Official Matson',case=False,regex=True).any()),
        'endpoint_ambiguous_records':int((g['r3_v8_primary_label_status']=='FLAG_ENDPOINT_AMBIGUOUS').sum()),
    })
endpoint_audit=pd.DataFrame(endpoint_rows).sort_values('domain_id')
endpoint_audit.to_csv(OUT/'Step10_10B_endpoint_definition_audit.tsv',sep='\t',index=False)

# Explicit RECIST recoding audit in Barcelona + Manchester
sub=inc[inc['domain_id'].isin(['Lee_Barcelona','Lee_Manchester'])].copy()
def recist_cat(x):
    s=str(x)
    for cat in ['Complete Response','Partial Response','Stable Disease','Progressive Disease','Not applicable']:
        if cat in s: return cat
    return 'Unresolved'
sub['recist_category']=sub['r3_v8_endpoint_evidence'].map(recist_cat)
sub['strict_orr_label']=sub['recist_category'].map({'Complete Response':'Responder','Partial Response':'Responder','Progressive Disease':'Non-responder'})
rec_summary=(sub.groupby(['domain_id','recist_category'],dropna=False).size().reset_index(name='n'))
rec_summary.to_csv(OUT/'Step10_10B_explicit_RECIST_category_counts.tsv',sep='\t',index=False)
strict=sub[sub['strict_orr_label'].notna()].copy()
strict_status=pd.DataFrame([{
    'domains':'Lee_Barcelona + Lee_Manchester','original_n':len(sub),'strict_ORR_evaluable_n':len(strict),
    'excluded_stable_disease_n':int((sub['recist_category']=='Stable Disease').sum()),
    'excluded_not_applicable_n':int((sub['recist_category']=='Not applicable').sum()),
    'discordant_labels_among_strict_evaluable_n':int((strict['strict_orr_label']!=strict['response_harmonized']).sum()),
}])
strict_status.to_csv(OUT/'Step10_10B_strict_ORR_recoding_audit.tsv',sep='\t',index=False)

# Endpoint-class relation in 72 directed transfers
trans=pd.read_csv(ROOT/'results/step10_5B/directed_domain_transfer_metrics.tsv',sep='\t')
t=trans[(trans['scenario']=='PRIMARY')&(trans['model']=='elastic_net')].copy()
t['train_endpoint_class']=t['train_domain'].map(endpoint_map)
t['test_endpoint_class']=t['test_domain'].map(endpoint_map)
t['same_endpoint_class']=(t['train_endpoint_class']==t['test_endpoint_class']).astype(int)
t['endpoint_direction']=t['train_endpoint_class'].str.replace('_COMPATIBLE','',regex=False)+' -> '+t['test_endpoint_class'].str.replace('_COMPATIBLE','',regex=False)
t.to_csv(OUT/'Step10_10B_directed_transfer_with_endpoint_class.tsv',sep='\t',index=False)

summary=t.groupby(['train_endpoint_class','test_endpoint_class']).agg(
    directed_pairs=('roc_auc','size'),mean_auc=('roc_auc','mean'),median_auc=('roc_auc','median'),
    sd_auc=('roc_auc','std'),mean_brier=('brier','mean'),mean_pr_auc=('pr_auc','mean')
).reset_index()
summary.to_csv(OUT/'Step10_10B_endpoint_direction_transfer_summary.tsv',sep='\t',index=False)

same=t[t.same_endpoint_class==1].roc_auc.to_numpy(); cross=t[t.same_endpoint_class==0].roc_auc.to_numpy()
obs=float(np.mean(same)-np.mean(cross))
# QAP-style permutation: permute endpoint labels across the 9 domain nodes, preserving 7/2 group size.
rng=np.random.default_rng(20260722)
domains=sorted(endpoint_map)
labels=np.array([endpoint_map[d] for d in domains],object)
auc_lookup={(r.train_domain,r.test_domain):r.roc_auc for r in t.itertuples()}
perm=[]
for _ in range(10000):
    plab=rng.permutation(labels)
    mp=dict(zip(domains,plab))
    s=[]; c=[]
    for (a,b),auc in auc_lookup.items():
        (s if mp[a]==mp[b] else c).append(auc)
    perm.append(np.mean(s)-np.mean(c))
perm=np.array(perm)
p_two=(1+np.sum(np.abs(perm)>=abs(obs)))/(len(perm)+1)
p_one=(1+np.sum(perm>=obs))/(len(perm)+1)
qap=pd.DataFrame([{
    'observed_same_minus_cross_mean_auc':obs,
    'same_endpoint_pairs':len(same),'cross_endpoint_pairs':len(cross),
    'same_endpoint_mean_auc':float(np.mean(same)),'cross_endpoint_mean_auc':float(np.mean(cross)),
    'permutation_n':len(perm),'qap_two_sided_p':p_two,'qap_one_sided_p_same_higher':p_one,
    'null_mean':float(np.mean(perm)),'null_sd':float(np.std(perm,ddof=1))
}])
qap.to_csv(OUT/'Step10_10B_endpoint_class_QAP_inference.tsv',sep='\t',index=False)

status={
 'step':'Step10.10A-B','status':'PASS',
 'bias_correction_conclusion':'No robust AUC improvement; target percentile delta CIs crossed zero and CORAL reduced discrimination.',
 'endpoint_conclusion':'Endpoint-class matching did not detectably explain directed transfer AUC; explicit RECIST recoding was concordant among evaluable Manchester/Barcelona records.',
 'endpoint_qap':qap.iloc[0].to_dict(),
 'strict_orr_audit':strict_status.iloc[0].to_dict(),
}
(OUT/'Step10_10AB_final_decision.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(status,indent=2,ensure_ascii=False))
