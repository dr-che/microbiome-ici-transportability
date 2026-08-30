import csv
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/melanoma_transportability_review/work/melanoma_microbiome_transportability')
OUT=Path('/mnt/data/melanoma_transportability_review/analysis4_outputs')
PRED=ROOT/'results/step10_5B/all_directed_transfer_predictions.tsv'
MET=ROOT/'results/step10_5B/directed_domain_transfer_metrics.tsv'
SEED=20260801; B=1000

def read(p):
 with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 fields=list(rows[0].keys())
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def auc(y,p):
 pos=p[y==1];neg=p[y==0];return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
metrics=[r for r in read(MET) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
preds=[r for r in read(PRED) if r['scenario']=='PRIMARY' and r['model']=='elastic_net']
doms=sorted(set(r['train_domain'] for r in metrics)); pairs=[(a,b) for i,a in enumerate(doms) for b in doms[i+1:]]
by={}
for r in preds:by.setdefault((r['test_domain'],r['train_domain']),[]).append(r)
yby={};prob={}
for t in doms:
 s=next(x for x in doms if x!=t); rr=sorted(by[(t,s)],key=lambda r:r['patient_id']); ids=[r['patient_id'] for r in rr]; yby[t]=np.array([int(r['true_label']) for r in rr])
 for ss in doms:
  if ss==t:continue
  z=sorted(by[(t,ss)],key=lambda r:r['patient_id']); assert [r['patient_id'] for r in z]==ids; prob[(ss,t)]=np.array([float(r['predicted_probability']) for r in z])
lookup={(r['train_domain'],r['test_domain']):float(r['roc_auc']) for r in metrics}
rng=np.random.default_rng(SEED+120000); arr=np.empty((B,len(pairs)))
for b in range(B):
 ixmap={}
 for t in doms:
  y=yby[t]; idx=[]
  for c in (0,1):
   ii=np.where(y==c)[0];idx.extend(rng.choice(ii,len(ii),replace=True))
  ixmap[t]=np.array(idx)
 for j,(a,bb) in enumerate(pairs):
  ixb=ixmap[bb]; ixa=ixmap[a]
  ab=auc(yby[bb][ixb],prob[(a,bb)][ixb]); ba=auc(yby[a][ixa],prob[(bb,a)][ixa]); arr[b,j]=ab-ba
rows=[]
for j,(a,b) in enumerate(pairs):
 v=arr[:,j]; obs=lookup[(a,b)]-lookup[(b,a)]; lo,hi=np.quantile(v,[.025,.975]); pp=(v>0).mean(); pn=(v<0).mean()
 rows.append({'domain_a':a,'domain_b':b,'observed_auc_a_to_b':lookup[(a,b)],'observed_auc_b_to_a':lookup[(b,a)],'observed_signed_delta_a_to_b_minus_b_to_a':obs,'observed_absolute_delta':abs(obs),'bootstrap_mean_signed_delta':v.mean(),'bootstrap_median_signed_delta':np.median(v),'percentile_2_5':lo,'percentile_97_5':hi,'bootstrap_probability_delta_positive':pp,'bootstrap_probability_delta_negative':pn,'direction_stability_probability':max(pp,pn),'percentile_interval_excludes_zero':bool(lo>0 or hi<0),'bootstrap_repetitions':B})
write(OUT/'Analysis4_pair_specific_target_bootstrap.tsv',rows)
# distribution long optional
long=[]
for b in range(B):
 for j,(a,c) in enumerate(pairs):long.append({'bootstrap_id':b+1,'domain_a':a,'domain_b':c,'signed_delta':arr[b,j]})
write(OUT/'Analysis4_pair_specific_target_bootstrap_distribution.tsv',long)
print('interval excludes zero',sum(r['percentile_interval_excludes_zero'] for r in rows),'of',len(rows))
print('stability >=.975',sum(float(r['direction_stability_probability'])>=.975 for r in rows))
