from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

BASE_SEED=20260720
MIN_PREV=0.10
FALLBACK_PREV=0.05
MAX_FEATURES=500
C=0.2
L1_RATIO=0.75


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def load_matrix(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        r=csv.reader(f, delimiter='\t')
        hdr=next(r)
        assert hdr[:4]==['manifest_id','patient_id','domain_id','response_harmonized']
        ids=[]; pids=[]; domains=[]; y=[]; vals=[]
        for row in r:
            ids.append(row[0]); pids.append(row[1]); domains.append(row[2]); y.append(1 if row[3]=='Responder' else 0); vals.append([float(x) for x in row[4:]])
    X=np.asarray(vals,float); X[X<0]=0; X/=X.sum(axis=1,keepdims=True)
    return ids,pids,np.asarray(domains,object),np.asarray(y,int),hdr[4:],X

def fit_preprocessor(Xtr):
    prev=np.mean(Xtr>0,axis=0)
    sel=np.where(prev>=MIN_PREV)[0]
    if len(sel)<10: sel=np.where(prev>=FALLBACK_PREV)[0]
    if len(sel)<5: sel=np.where(np.any(Xtr>0,axis=0))[0]
    nz=Xtr[:,sel][Xtr[:,sel]>0]
    pc=float(np.min(nz)/2) if nz.size else 1e-6
    pc=min(max(pc,1e-8),1e-4)
    clr=np.log(Xtr[:,sel]+pc); clr-=clr.mean(axis=1,keepdims=True)
    vars=np.var(clr,axis=0)
    if len(sel)>MAX_FEATURES:
        order=np.argsort(vars)[::-1][:MAX_FEATURES]; sel=sel[order]
        clr=np.log(Xtr[:,sel]+pc); clr-=clr.mean(axis=1,keepdims=True)
    mean=clr.mean(axis=0); scale=clr.std(axis=0,ddof=0); scale[scale<1e-10]=1
    return sel,pc,mean,scale

def apply(X,state):
    sel,pc,mean,scale=state
    clr=np.log(X[:,sel]+pc); clr-=clr.mean(axis=1,keepdims=True)
    return (clr-mean)/scale

def build_model(seed):
    return LogisticRegression(penalty='elasticnet',solver='saga',C=C,l1_ratio=L1_RATIO,class_weight='balanced',max_iter=5000,tol=1e-3,random_state=seed,n_jobs=1)

def prepare(X,domains):
    uds=sorted(np.unique(domains).tolist())
    caches={}
    for i,d in enumerate(uds):
        tr=np.where(domains==d)[0]
        st=fit_preprocessor(X[tr])
        caches[d]={'train_idx':tr,'X_train':apply(X[tr],st),'state':st,'source_index':i}
        for t in uds:
            if t!=d:
                te=np.where(domains==t)[0]
                caches[d].setdefault('targets',{})[t]=(te,apply(X[te],st))
    return uds,caches

def permute_y(y,domains,index):
    rng=np.random.default_rng(BASE_SEED+index*7919)
    yp=y.copy()
    for d in np.unique(domains):
        idx=np.where(domains==d)[0]
        yp[idx]=rng.permutation(yp[idx])
    return yp

def run_one(index,y,domains,uds,caches,observed=False):
    yp=y if observed else permute_y(y,domains,index)
    aucs={}
    for d in uds:
        c=caches[d]
        seed=(BASE_SEED+c['source_index']*1000) if observed else (BASE_SEED+index*100000+c['source_index']*1000)
        model=build_model(seed)
        model.fit(c['X_train'],yp[c['train_idx']])
        for t,(te,Xte) in c['targets'].items():
            p=model.predict_proba(Xte)[:,1]
            aucs[(d,t)]=float(roc_auc_score(yp[te],p))
    pair=[]
    for i,a in enumerate(uds):
        for b in uds[i+1:]:
            ab=aucs[(a,b)]; ba=aucs[(b,a)]; diff=abs(ab-ba)
            pair.append((a,b,ab,ba,diff))
    diffs=np.array([x[4] for x in pair])
    return {
        'index':index,
        'mean_abs_asymmetry':float(np.mean(diffs)),
        'median_abs_asymmetry':float(np.median(diffs)),
        'pairs_ge_0_10':int(np.sum(diffs>=0.10)),
        'pairs_ge_0_20':int(np.sum(diffs>=0.20)),
        'max_abs_asymmetry':float(np.max(diffs)),
        'pair':pair,
    }

def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',required=True); ap.add_argument('--out',required=True); ap.add_argument('--permutations',type=int,default=1000); ap.add_argument('--jobs',type=int,default=1)
    args=ap.parse_args(); matrix=Path(args.matrix); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    t0=time.time(); ids,pids,domains,y,features,X=load_matrix(matrix); uds,caches=prepare(X,domains)
    obs=run_one(0,y,domains,uds,caches,observed=True)
    print('observed', {k:v for k,v in obs.items() if k!='pair'}, flush=True)
    results=Parallel(n_jobs=args.jobs,verbose=10,batch_size=1)(delayed(run_one)(i,y,domains,uds,caches,False) for i in range(1,args.permutations+1))
    metrics=['mean_abs_asymmetry','median_abs_asymmetry','pairs_ge_0_10','pairs_ge_0_20','max_abs_asymmetry']
    nullrows=[{k:r[k] for k in ['index']+metrics} for r in results]
    write_tsv(out/'Analysis2_null_distribution.tsv',nullrows,['index']+metrics)
    obspairs=[]
    for a,b,ab,ba,d in obs['pair']:
        obspairs.append({'domain_a':a,'domain_b':b,'auc_a_to_b':ab,'auc_b_to_a':ba,'absolute_asymmetry':d})
    write_tsv(out/'Analysis2_observed_reciprocal_pairs.tsv',obspairs,list(obspairs[0]))
    # pair-specific null rows / summaries
    pair_null={ (a,b):[] for a,b,_,_,_ in obs['pair'] }
    for r in results:
        for a,b,ab,ba,d in r['pair']:
            pair_null[(a,b)].append(d)
    pair_summ=[]
    for row in obspairs:
        key=(row['domain_a'],row['domain_b']); vals=np.array(pair_null[key]); o=row['absolute_asymmetry']
        pair_summ.append({**row,'null_mean':float(vals.mean()),'null_sd':float(vals.std(ddof=1)),'null_95th':float(np.quantile(vals,.95)),'empirical_one_sided_p':float((1+np.sum(vals>=o))/(len(vals)+1))})
    write_tsv(out/'Analysis2_pair_specific_null_summary.tsv',pair_summ,list(pair_summ[0]))
    summary=[]
    for m in metrics:
        vals=np.array([r[m] for r in results],float); o=float(obs[m])
        p=(1+np.sum(vals>=o))/(len(vals)+1)
        mc=math.sqrt(p*(1-p)/(len(vals)+1))
        summary.append({'metric':m,'observed':o,'null_mean':float(vals.mean()),'null_sd':float(vals.std(ddof=1)),'null_median':float(np.median(vals)),'null_025':float(np.quantile(vals,.025)),'null_95':float(np.quantile(vals,.95)),'null_975':float(np.quantile(vals,.975)),'empirical_one_sided_p':float(p),'monte_carlo_se':float(mc),'permutations':len(vals)})
    write_tsv(out/'Analysis2_global_null_inference.tsv',summary,list(summary[0]))
    status={'analysis':'Analysis 2 reciprocal directional asymmetry domain-preserving null','matrix':str(matrix),'matrix_sha256':sha256(matrix),'n':len(y),'responders':int(y.sum()),'domains':uds,'features':len(features),'permutations':args.permutations,'jobs':args.jobs,'model':{'type':'elastic_net','C':C,'l1_ratio':L1_RATIO,'class_weight':'balanced','max_features':MAX_FEATURES},'permutation':'response labels shuffled separately within each domain; prevalence preserved','preprocessing':'source-only prevalence filtering, pseudocount, CLR, variance ranking, centering and scaling; label-independent preprocessing cached exactly across permutations','observed':{k:v for k,v in obs.items() if k!='pair'},'runtime_seconds':time.time()-t0}
    (out/'Analysis2_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__': main()
