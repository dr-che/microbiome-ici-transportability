from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SEED=20260722

def parse_args():
    p=argparse.ArgumentParser(description='Reproduce explicit distribution-correction benchmark.')
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output-dir',default='reproduced/step10_10/bias_benchmark')
    p.add_argument('--bootstrap',type=int,default=500)
    return p.parse_args()

def fit_state(Xtr,max_features=500):
    prev=(Xtr>0).mean(0); sel=np.where(prev>=.10)[0]
    if len(sel)<10: sel=np.where(prev>=.05)[0]
    if len(sel)<5: sel=np.where(np.any(Xtr>0,axis=0))[0]
    nz=Xtr[:,sel][Xtr[:,sel]>0]; pc=float(np.clip((nz.min()/2 if nz.size else 1e-6),1e-8,1e-4))
    Z=np.log(Xtr[:,sel]+pc); Z-=Z.mean(1,keepdims=True); var=Z.var(0)
    if len(sel)>max_features: sel=sel[np.argsort(var)[::-1][:max_features]]
    Z=np.log(Xtr[:,sel]+pc); Z-=Z.mean(1,keepdims=True); mu=Z.mean(0); sd=Z.std(0); sd[sd<1e-10]=1
    return sel,pc,mu,sd

def clr(X,pc):
    Z=np.log(X+pc); return Z-Z.mean(1,keepdims=True)

def percentile_domain(A,domains):
    out=np.empty_like(A,float)
    for d in pd.unique(domains):
        idx=np.where(domains==d)[0]
        for j in range(A.shape[1]): out[idx,j]=(pd.Series(A[idx,j]).rank(method='average').to_numpy()-.5)/len(idx)
    return out

def sqrtm(C,inverse=False,eps=1e-4):
    v,U=np.linalg.eigh(C); v=np.maximum(v,eps); p=-.5 if inverse else .5; return (U*(v**p))@U.T

def coral_target_to_source(Ztr,Zte):
    if len(Zte)<3:return Zte
    Cs=np.cov(Ztr,rowvar=False)+np.eye(Ztr.shape[1])*1e-3; Ct=np.cov(Zte,rowvar=False)+np.eye(Zte.shape[1])*1e-3
    return (Zte-Zte.mean(0))@sqrtm(Ct,True)@sqrtm(Cs,False)+Ztr.mean(0)

def model():
    return LogisticRegression(penalty='elasticnet',solver='saga',C=.2,l1_ratio=.75,class_weight='balanced',max_iter=5000,tol=1e-3,random_state=SEED,n_jobs=1)

def metrics(y,p):
    return dict(roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]))

def main():
    args=parse_args(); root=Path(args.root).resolve(); out=root/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    mat=pd.read_csv(root/'data/derived/assembled_species_matrix.tsv',sep='\t')
    man=pd.read_csv(root/'data/metadata/Step10_2_manifest_R3_primary_lock_v8.tsv',sep='\t')
    inc=man['r3_v8_primary_analysis_include'].astype(str).str.upper().isin(['TRUE','1','YES'])
    man=man.loc[inc,['manifest_id','study_family']]
    df=mat.merge(man,on='manifest_id',validate='one_to_one')
    features=[c for c in mat.columns if c.startswith('s__')]
    X=df[features].to_numpy(float); y=(df.response_harmonized=='Responder').astype(int).to_numpy(); dom=df.domain_id.to_numpy(); fam=df.study_family.to_numpy()
    predrows=[]; foldrows=[]
    for scheme,groups in [('LODO',dom),('LOSFO',fam)]:
      for held in pd.unique(groups):
        te=groups==held; tr=~te; sel,pc,mu,sd=fit_state(X[tr]); Xtr=X[tr][:,sel]; Xte=X[te][:,sel]; ytr=y[tr]; yte=y[te]; dtr=dom[tr]; dte=dom[te]
        Ztr=(clr(Xtr,pc)-mu)/sd; Zte=(clr(Xte,pc)-mu)/sd
        mb=model().fit(Ztr,ytr); outputs={'baseline_CLR':mb.predict_proba(Zte)[:,1]}
        Ptr=percentile_domain(Xtr,dtr); Pte=percentile_domain(Xte,dte); pmu=Ptr.mean(0); psd=Ptr.std(0); psd[psd<1e-10]=1
        mp=model().fit((Ptr-pmu)/psd,ytr); outputs['target_percentile']=mp.predict_proba((Pte-pmu)/psd)[:,1]
        rawtr=clr(Xtr,pc); rawte=clr(Xte,pc); cte=np.zeros_like(rawte)
        for td in pd.unique(dte):
            ix=np.where(dte==td)[0]; cte[ix]=coral_target_to_source(rawtr,rawte[ix])
        cmu=rawtr.mean(0); csd=rawtr.std(0); csd[csd<1e-10]=1
        mc=model().fit((rawtr-cmu)/csd,ytr); outputs['CORAL_target_to_source']=mc.predict_proba((cte-cmu)/csd)[:,1]
        for method,p in outputs.items():
            foldrows.append({'scheme':scheme,'held_out_group':held,'method':method,'n_test':int(te.sum()),'n_features':len(sel),**metrics(yte,p)})
            for idx,prob in zip(np.where(te)[0],p): predrows.append({'scheme':scheme,'held_out_group':held,'manifest_id':df.iloc[idx].manifest_id,'domain_id':dom[idx],'study_family':fam[idx],'y':int(y[idx]),'prob':float(prob),'method':method})
    pred=pd.DataFrame(predrows); folds=pd.DataFrame(foldrows); pred.to_csv(out/'predictions.tsv',sep='\t',index=False); folds.to_csv(out/'fold_metrics.tsv',sep='\t',index=False)
    summary=[]
    for (s,m),g in pred.groupby(['scheme','method']): summary.append({'scheme':s,'method':m,'n':len(g),**metrics(g.y,g.prob)})
    pd.DataFrame(summary).to_csv(out/'overall_metrics.tsv',sep='\t',index=False)
    rng=np.random.default_rng(SEED); deltas=[]
    for s in ['LODO','LOSFO']:
        w=pred[pred.scheme==s].pivot_table(index=['manifest_id','held_out_group','y'],columns='method',values='prob').reset_index(); units=w.held_out_group.unique(); idxmap={u:w.index[w.held_out_group==u].to_numpy() for u in units}
        for m in ['target_percentile','CORAL_target_to_source']:
            obs=roc_auc_score(w.y,w[m])-roc_auc_score(w.y,w.baseline_CLR); bs=[]
            for _ in range(args.bootstrap):
                sample=rng.choice(units,len(units),replace=True); ix=np.concatenate([idxmap[u] for u in sample]); yy=w.y.to_numpy()[ix]
                if len(np.unique(yy))<2:continue
                bs.append(roc_auc_score(yy,w[m].to_numpy()[ix])-roc_auc_score(yy,w.baseline_CLR.to_numpy()[ix]))
            deltas.append({'scheme':s,'method':m,'delta_auc':obs,'ci_low':np.quantile(bs,.025),'ci_high':np.quantile(bs,.975),'p_boot_gt0':np.mean(np.array(bs)>0)})
    pd.DataFrame(deltas).to_csv(out/'delta_bootstrap.tsv',sep='\t',index=False)
    (out/'status.json').write_text(json.dumps({'status':'PASS','bootstrap':args.bootstrap},indent=2),encoding='utf-8')
    print(out)
if __name__=='__main__':main()
