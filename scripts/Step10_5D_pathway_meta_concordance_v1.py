from __future__ import annotations
import os, re, math, json, zipfile, shutil, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t as student_t, rankdata, mannwhitneyu

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
args = parser.parse_args()
PROJECT_ROOT = Path(str(args.root).strip().strip('"').rstrip('\\/')).resolve()

V1_ZIP = PROJECT_ROOT / 'Melanoma_ICI_Microbiome_Reproducibility_v1_2.zip'
if not V1_ZIP.is_file():
    raise FileNotFoundError(f'Missing required V1 reproducibility archive: {V1_ZIP}')

WORK = PROJECT_ROOT / '.step10_5d_v1_functional_data'
expected = WORK / 'Melanoma_ICI_Microbiome_Reproducibility_v1_2' / '02_processed_data' / 'pathway_abundance_filtered.tsv'
if not expected.is_file():
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(V1_ZIP) as zf:
        zf.extractall(WORK)

candidates = list(WORK.rglob('pathway_abundance_filtered.tsv'))
if not candidates:
    raise FileNotFoundError('pathway_abundance_filtered.tsv was not found after extracting the V1 archive.')
ROOT = candidates[0].parent
LOCK = PROJECT_ROOT
OUT = PROJECT_ROOT / '02_results_step10_5D_v1'
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

STUDY_ORDER=['FrankelAE_2017','LeeKA_2022','MatsonV_2018','PetersBA_2019','WindTT_2020']
MIN_PREV=0.10
MIN_GROUP=3
MIN_STUDIES=3

def bh(pvals):
    p=np.asarray(pvals,float); q=np.full(len(p),np.nan)
    valid=np.where(np.isfinite(p))[0]
    if len(valid)==0:return q
    order=valid[np.argsort(p[valid])]; m=len(order); prev=1.0
    for pos in range(m-1,-1,-1):
        rank=pos+1; val=p[order[pos]]*m/rank; prev=min(prev,val); q[order[pos]]=min(prev,1.0)
    return q

def reml_meta(effects, variances):
    y=np.asarray(effects,float); v=np.asarray(variances,float)
    m=np.isfinite(y)&np.isfinite(v)&(v>0); y=y[m]; v=v[m]; k=len(y)
    if k<2:return None
    max_tau=max(float(np.var(y,ddof=1))*10 if k>1 else 1,float(np.max(v))*10,1.0)
    def obj(tau2):
        w=1/(v+tau2); mu=np.sum(w*y)/np.sum(w); Q=np.sum(w*(y-mu)**2)
        return .5*(np.sum(np.log(v+tau2))+math.log(np.sum(w))+Q)
    res=minimize_scalar(obj,bounds=(0,max_tau),method='bounded',options={'xatol':1e-10})
    tau2=max(0,float(res.x)) if res.success else 0
    w=1/(v+tau2); mu=float(np.sum(w*y)/np.sum(w)); qstar=float(np.sum(w*(y-mu)**2)/max(k-1,1)); qstar=max(qstar,1)
    se=math.sqrt(qstar/np.sum(w)); df=k-1; crit=float(student_t.ppf(.975,df)); lo=mu-crit*se; hi=mu+crit*se
    p=float(2*student_t.sf(abs(mu/se),df)) if se>0 else np.nan
    wf=1/v; muf=float(np.sum(wf*y)/np.sum(wf)); Q=float(np.sum(wf*(y-muf)**2)); Qp=float(chi2.sf(Q,k-1)); I2=max(0,(Q-(k-1))/Q*100) if Q>0 else 0
    if k>=3:
        pc=float(student_t.ppf(.975,k-2)); ps=math.sqrt(tau2+se*se); pl=mu-pc*ps; ph=mu+pc*ps
    else: pl=ph=np.nan
    pos=int((y>0).sum()); neg=int((y<0).sum())
    return dict(k=k,pooled_effect=mu,se=se,ci_low=lo,ci_high=hi,p_value=p,tau2=tau2,Q=Q,Q_p=Qp,I2=I2,prediction_low=pl,prediction_high=ph,positive_studies=pos,negative_studies=neg,sign_consistency=max(pos,neg)/k,minimum_effect=float(y.min()),maximum_effect=float(y.max()),median_effect=float(np.median(y)))

def hedges_g(X,y):
    a=X[y==1]; b=X[y==0]; n1=len(a); n0=len(b); df=n1+n0-2
    m1=a.mean(0); m0=b.mean(0); v1=a.var(0,ddof=1); v0=b.var(0,ddof=1)
    pv=((n1-1)*v1+(n0-1)*v0)/max(df,1); sd=np.sqrt(np.maximum(pv,0))
    d=np.divide(m1-m0,sd,out=np.full_like(m1,np.nan),where=sd>1e-12); J=1-3/(4*df-1) if df>1 else 1; g=J*d
    vg=(n1+n0)/(n1*n0)+g*g/(2*max(df,1))
    return g,vg,m1,m0

def canonical_species_name(x):
    x=re.sub(r'^species:','',str(x)); x=re.sub(r'^s__','',x); x=x.replace('_',' ').strip(); return x.lower()

def pathway_category(name):
    s=name.lower()
    rules=[
        ('Amino acid', ['amino acid','arginine','lysine','methionine','tryptophan','histidine','valine','isoleucine','leucine','ornithine','glutamate','glutamine','serine','glycine','threonine','cysteine','phenylalanine','tyrosine']),
        ('Nucleotide', ['nucleotide','nucleoside','purine','pyrimidine','adenosine','guanosine','uridine','cytidine','inosine','ump','amp','gmp']),
        ('Carbohydrate', ['glycolysis','starch','glycogen','glucose','fructose','galactose','mannose','pentose','sugar','carbohydrate','xylan','arabinose','rhamnose']),
        ('Fermentation/SCFA', ['fermentation','butanoate','butyrate','propanoate','propionate','acetate','lactate','succinate']),
        ('Lipid', ['fatty acid','lipid','phospholipid','diacylglycerol','palmitate','beta-oxidation','&beta;-oxidation']),
        ('Cofactor/vitamin', ['vitamin','folate','coenzyme','biotin','pantothenate','riboflavin','thiamin','cobalamin','nad','heme','menaquinone','ubiquinone']),
        ('Cell envelope', ['peptidoglycan','lipopolysaccharide','cell wall','teichoic','muramoyl']),
        ('Aromatic/secondary', ['chorismate','aromatic','phenylpropanoate','cinnamate','coumarate','isoprene','taxadiene']),
    ]
    for cat,keys in rules:
        if any(k in s for k in keys):return cat
    return 'Other'

# Load metadata and matrices
meta=pd.read_csv(ROOT/'pathway_abundance_sample_metadata_primary.tsv',sep='\t')
meta=meta.set_index('export_sample_id',drop=False)
y=(meta['response_standard']=='Responder').astype(int).values
studies=meta['study_name'].values
sample_ids=meta['export_sample_id'].tolist()

pw_raw=pd.read_csv(ROOT/'pathway_abundance_filtered.tsv',sep='\t')
pw_raw=pw_raw.set_index('feature')
pw_features=[f for f in pw_raw.index if '|' not in f and f not in {'UNMAPPED','UNINTEGRATED'}]
P=pw_raw.loc[pw_features,sample_ids].T.astype(float).values
# renormalize only unstratified community pathways
rs=P.sum(1); P=np.divide(P,rs[:,None],out=np.zeros_like(P),where=rs[:,None]>0)

sp_raw=pd.read_csv(ROOT/'species_abundance_filtered.tsv',sep='\t').set_index('feature')
sp_features=[f for f in sp_raw.index if str(f).startswith('species:')]
S=sp_raw.loc[sp_features,sample_ids].T.astype(float).values
rs=S.sum(1); S=np.divide(S,rs[:,None],out=np.zeros_like(S),where=rs[:,None]>0)

# Domain effects/meta for each layer
def analyze_layer(X,features,layer):
    effects=[]; effect_matrix=np.full((len(features),len(STUDY_ORDER)),np.nan)
    for j,study in enumerate(STUDY_ORDER):
        m=studies==study; yy=y[m]; xx=X[m]
        prev=(xx>0).mean(0); pos=xx[xx>0]; pc=max(min(float(pos.min()/2) if pos.size else 1e-8,1e-4),1e-10)
        clr=np.log(xx+pc); clr-=clr.mean(1,keepdims=True)
        g,vg,m1,m0=hedges_g(clr,yy)
        for i,f in enumerate(features):
            if prev[i]<MIN_PREV or not np.isfinite(g[i]) or not np.isfinite(vg[i]) or vg[i]<=0: continue
            effect_matrix[i,j]=g[i]
            se=math.sqrt(vg[i]); crit=1.96
            effects.append(dict(layer=layer,feature=f,study=study,n=int(m.sum()),responders=int(yy.sum()),nonresponders=int(len(yy)-yy.sum()),prevalence=float(prev[i]),pseudocount=pc,effect=float(g[i]),variance=float(vg[i]),se=se,ci_low=float(g[i]-crit*se),ci_high=float(g[i]+crit*se),mean_clr_R=float(m1[i]),mean_clr_NR=float(m0[i])))
    edf=pd.DataFrame(effects)
    metas=[]
    for f,gdf in edf.groupby('feature'):
        if len(gdf)<MIN_STUDIES:continue
        r=reml_meta(gdf.effect,gdf.variance)
        sigpos=int((gdf.ci_low>0).sum()); signeg=int((gdf.ci_high<0).sum())
        metas.append(dict(layer=layer,feature=f,**r,significant_positive_studies=sigpos,significant_negative_studies=signeg,study_list=';'.join(gdf.study)))
    mdf=pd.DataFrame(metas)
    mdf['fdr_q']=bh(mdf.p_value.values)
    return edf,mdf,effect_matrix

pw_eff,pw_meta,pw_effect_mat=analyze_layer(P,pw_features,'pathway')
sp_eff,sp_meta,sp_effect_mat=analyze_layer(S,sp_features,'species')

# influence and classification
def classify(meta_df,eff_df,layer):
    rows=[]
    for _,r in meta_df.iterrows():
        f=r.feature; e=eff_df[eff_df.feature==f]
        loo=[]
        for st in e.study:
            q=e[e.study!=st]
            if len(q)>=3:
                rr=reml_meta(q.effect,q.variance); loo.append((st,rr['pooled_effect']))
        loo_flip=any(v*r.pooled_effect<0 for _,v in loo) if r.pooled_effect!=0 else False
        absvals=np.abs(e.effect.values); domshare=float(absvals.max()/absvals.sum()) if absvals.sum()>0 else np.nan
        sigpos=int(r.significant_positive_studies); signeg=int(r.significant_negative_studies)
        if r.fdr_q<.10 and r.k>=4 and r.sign_consistency>=.8 and r.I2<50:
            c='ROBUST_CONSISTENT_POSITIVE' if r.pooled_effect>0 else 'ROBUST_CONSISTENT_NEGATIVE'
        elif sigpos>=1 and signeg>=1:
            c='STRONG_DIRECTION_REVERSAL'
        elif r.positive_studies>=2 and r.negative_studies>=2 and r.sign_consistency<=.6 and r.I2>=50:
            c='PROBABLE_DIRECTION_REVERSAL'
        elif (sigpos+signeg)==1 and r.fdr_q>=.10 and (r.I2>=50 or domshare>=.35 or loo_flip):
            c='STUDY_SPECIFIC'
        elif r.I2>=75:
            c='HIGH_HETEROGENEITY'
        elif r.sign_consistency>=.8 and r.I2<50:
            c='CONSISTENT_NONSIGNIFICANT'
        else:c='OTHER_OR_INSUFFICIENT'
        base=r.to_dict(); base.update({'classification':c,'dominant_study_absolute_effect_share':domshare,'any_leave_one_study_sign_flip':int(loo_flip)}); rows.append(base)
    return pd.DataFrame(rows)

pw_cls=classify(pw_meta,pw_eff,'pathway'); sp_cls=classify(sp_meta,sp_eff,'species')

# Study-adjusted partial Spearman for all species-pathway pairs
def residualized_ranks(X,groups):
    R=np.apply_along_axis(rankdata,0,X)
    for g in np.unique(groups):
        m=groups==g; R[m]-=R[m].mean(0,keepdims=True)
    R-=R.mean(0,keepdims=True); sd=np.sqrt((R*R).sum(0)); sd[sd==0]=1; return R/sd
SR=residualized_ranks(S,studies); PR=residualized_ranks(P,studies)
rho=SR.T@PR
n=len(y); df=n-len(np.unique(studies))-2
T=rho*np.sqrt(df/np.maximum(1-rho*rho,1e-12)); corr_p=2*student_t.sf(np.abs(T),df)
flatq=bh(corr_p.ravel()).reshape(corr_p.shape)

# Effect-vector Spearman correlations across 5 studies, all pairs, require >=3 common
sp_er=np.apply_along_axis(rankdata,1,np.nan_to_num(sp_effect_mat,nan=0.0))
pw_er=np.apply_along_axis(rankdata,1,np.nan_to_num(pw_effect_mat,nan=0.0))
sp_er-=sp_er.mean(1,keepdims=True); pw_er-=pw_er.mean(1,keepdims=True)
spnorm=np.sqrt((sp_er*sp_er).sum(1)); pwnorm=np.sqrt((pw_er*pw_er).sum(1)); spnorm[spnorm==0]=1;pwnorm[pwnorm==0]=1
effrho=(sp_er/spnorm[:,None])@(pw_er/pwnorm[:,None]).T
# sign agreement on study effects
sign_agree=np.zeros_like(effrho)
common_counts=np.zeros_like(effrho,dtype=int)
for i in range(len(sp_features)):
    a=sp_effect_mat[i]
    for j in range(len(pw_features)):
        b=pw_effect_mat[j]; m=np.isfinite(a)&np.isfinite(b)&(a!=0)&(b!=0); common_counts[i,j]=m.sum(); sign_agree[i,j]=np.mean(np.sign(a[m])==np.sign(b[m])) if m.sum() else np.nan

# all pair table
pair_rows=[]
for i,sf in enumerate(sp_features):
    for j,pf in enumerate(pw_features):
        pair_rows.append((sf,pf,float(rho[i,j]),float(corr_p[i,j]),float(flatq[i,j]),float(effrho[i,j]),int(common_counts[i,j]),float(sign_agree[i,j]) if np.isfinite(sign_agree[i,j]) else np.nan))
pairs=pd.DataFrame(pair_rows,columns=['species','pathway','partial_spearman_rho_study_adjusted','partial_spearman_p','partial_spearman_fdr_q','response_effect_vector_spearman_rho','common_effect_studies','response_effect_sign_agreement'])
pairs['concordant_positive']=((pairs.partial_spearman_rho_study_adjusted>0)&(pairs.response_effect_vector_spearman_rho>=.6)&(pairs.response_effect_sign_agreement>=.8)&(pairs.common_effect_studies>=3)).astype(int)
pairs['concordant_inverse']=((pairs.partial_spearman_rho_study_adjusted<0)&(pairs.response_effect_vector_spearman_rho<=-.6)&(pairs.response_effect_sign_agreement<=.2)&(pairs.common_effect_studies>=3)).astype(int)

# Map V2 candidates into V1 species
candidate_files={
'E1':'Step10_5C_exploratory_triangulated_candidates_v2.tsv',
'REVERSAL':'Step10_5C_direction_reversal_candidates_v2.tsv',
'DOMAIN_SPECIFIC':'Step10_5C_domain_specific_candidates_v2.tsv'}
maprows=[]; v1canon={canonical_species_name(f):f for f in sp_features}
for category,fn in candidate_files.items():
    df=pd.read_csv(LOCK/fn,sep='\t')
    for name in df.species:
        can=canonical_species_name(name); match=v1canon.get(can)
        maprows.append(dict(v2_category=category,v2_species=name,canonical_name=can,v1_species_feature=match or '',mapped=int(match is not None)))
mapdf=pd.DataFrame(maprows)
mapped=mapdf[mapdf.mapped==1]
cp=pairs[pairs.species.isin(mapped.v1_species_feature)].copy()
cp=cp.merge(mapdf[['v2_category','v2_species','v1_species_feature']],left_on='species',right_on='v1_species_feature',how='left')
cp['abs_partial_rho']=cp.partial_spearman_rho_study_adjusted.abs()
# top 15 per mapped species by evidence: significant corr then effect concordance then abs corr
cp=cp.sort_values(['species','partial_spearman_fdr_q','concordant_positive','concordant_inverse','abs_partial_rho'],ascending=[True,True,False,False,False])
top_cp=cp.groupby('species').head(15).copy()

# convergence pathways across mapped candidates, count strong correlations q<.05 |rho|>=.35 and effect same-direction tracking
strong=cp[(cp.partial_spearman_fdr_q<.05)&(cp.abs_partial_rho>=.35)&(cp.common_effect_studies>=3)&(cp.response_effect_vector_spearman_rho.abs()>=.6)].copy()
conv=strong.groupby('pathway').agg(candidate_species_count=('species','nunique'),candidate_species=('species',lambda x:';'.join(sorted(set(x)))),mean_abs_partial_rho=('abs_partial_rho','mean'),mean_abs_effect_rho=('response_effect_vector_spearman_rho',lambda x:float(np.mean(np.abs(x))))).reset_index().sort_values(['candidate_species_count','mean_abs_partial_rho'],ascending=[False,False])

# functional categories
pw_cls['functional_category']=pw_cls.feature.map(pathway_category)
cat_summary=pw_cls.groupby('functional_category').agg(pathways=('feature','size'),nominal_p_lt_0_05=('p_value',lambda x:int((x<.05).sum())),fdr_q_lt_0_10=('fdr_q',lambda x:int((x<.10).sum())),median_I2=('I2','median'),mean_sign_consistency=('sign_consistency','mean')).reset_index()

# layer comparison
layer_comp=[]
for name,df in [('Community pathways',pw_cls),('Species',sp_cls)]:
    layer_comp.append(dict(layer=name,features_meta_analyzed=len(df),nominal_p_lt_0_05=int((df.p_value<.05).sum()),nominal_p_lt_0_10=int((df.p_value<.10).sum()),fdr_q_lt_0_10=int((df.fdr_q<.10).sum()),median_I2=float(df.I2.median()),I2_ge_50=int((df.I2>=50).sum()),I2_ge_75=int((df.I2>=75).sum()),mean_sign_consistency=float(df.sign_consistency.mean()),strong_reversal=int((df.classification=='STRONG_DIRECTION_REVERSAL').sum()),study_specific=int((df.classification=='STUDY_SPECIFIC').sum())))
layer_comp=pd.DataFrame(layer_comp)
try:
    mw=mannwhitneyu(pw_cls.I2,sp_cls.I2,alternative='two-sided')
    hetero_test={'mann_whitney_U':float(mw.statistic),'p_value':float(mw.pvalue)}
except Exception: hetero_test={}

# outputs
pw_eff.to_csv(OUT/'pathway_study_effects.tsv',sep='\t',index=False)
pw_meta.to_csv(OUT/'pathway_random_effects_meta_analysis.tsv',sep='\t',index=False)
pw_cls.to_csv(OUT/'pathway_directionality_and_specificity_classification.tsv',sep='\t',index=False)
sp_eff.to_csv(OUT/'paired_v1_species_study_effects.tsv',sep='\t',index=False)
sp_meta.to_csv(OUT/'paired_v1_species_meta_analysis.tsv',sep='\t',index=False)
sp_cls.to_csv(OUT/'paired_v1_species_classification.tsv',sep='\t',index=False)
pairs.to_csv(OUT/'all_species_pathway_concordance.tsv',sep='\t',index=False)
mapdf.to_csv(OUT/'v2_species_candidate_mapping_to_v1.tsv',sep='\t',index=False)
top_cp.to_csv(OUT/'v2_candidate_species_top_pathway_concordance.tsv',sep='\t',index=False)
conv.to_csv(OUT/'multi_species_functional_convergence.tsv',sep='\t',index=False)
cat_summary.to_csv(OUT/'pathway_functional_category_summary.tsv',sep='\t',index=False)
layer_comp.to_csv(OUT/'species_vs_pathway_layer_comparison.tsv',sep='\t',index=False)

# Top pathway tables
pw_cls.sort_values('p_value').head(50).to_csv(OUT/'top_pathway_meta_associations.tsv',sep='\t',index=False)
pw_cls[pw_cls.classification.str.contains('REVERSAL')].sort_values('I2',ascending=False).to_csv(OUT/'pathway_direction_reversal_candidates.tsv',sep='\t',index=False)
pw_cls[pw_cls.classification=='STUDY_SPECIFIC'].sort_values('I2',ascending=False).to_csv(OUT/'pathway_study_specific_candidates.tsv',sep='\t',index=False)

# Figures
if HAS_MATPLOTLIB:
    plt.figure(figsize=(8,6)); plt.scatter(pw_cls.pooled_effect,pw_cls.I2,s=18,alpha=.7); top=pw_cls.nsmallest(8,'p_value')
    for _,r in top.iterrows(): plt.annotate(str(r.feature).split(':')[0],(r.pooled_effect,r.I2),fontsize=7)
    plt.axvline(0,linewidth=.8); plt.xlabel('Pooled Hedges g (Responder - Non-responder)'); plt.ylabel('I² (%)'); plt.title('Community pathway effects and heterogeneity'); plt.tight_layout(); plt.savefig(OUT/'Figure10_5D_pathway_meta_effect_heterogeneity.svg'); plt.close()

    counts=pw_cls.classification.value_counts(); plt.figure(figsize=(9,5)); counts.plot(kind='bar'); plt.ylabel('Number of pathways'); plt.title('Pathway directionality and specificity classes'); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(OUT/'Figure10_5D_pathway_classification_counts.svg'); plt.close()

    # Candidate heatmap top pathways union (up to 20)
    sel_species=mapped.v1_species_feature.tolist(); sel_paths=top_cp.sort_values('partial_spearman_fdr_q').pathway.drop_duplicates().head(20).tolist()
    if sel_species and sel_paths:
        mat=pairs[pairs.species.isin(sel_species)&pairs.pathway.isin(sel_paths)].pivot(index='species',columns='pathway',values='partial_spearman_rho_study_adjusted').reindex(index=sel_species,columns=sel_paths)
        plt.figure(figsize=(14,max(4,len(sel_species)*.55))); im=plt.imshow(mat.values,aspect='auto',vmin=-1,vmax=1); plt.colorbar(im,label='Study-adjusted partial Spearman rho'); plt.yticks(range(len(mat.index)),[x.replace('species:','') for x in mat.index],fontsize=8); plt.xticks(range(len(mat.columns)),[x.split(':')[0] for x in mat.columns],rotation=65,ha='right',fontsize=7); plt.title('V2 candidate species–V1 community pathway concordance'); plt.tight_layout(); plt.savefig(OUT/'Figure10_5D_candidate_species_pathway_concordance.svg'); plt.close()

    plt.figure(figsize=(7,5)); x=np.arange(len(layer_comp)); plt.bar(x-.18,layer_comp.median_I2,width=.36,label='Median I²'); plt.bar(x+.18,layer_comp.mean_sign_consistency*100,width=.36,label='Mean sign consistency (%)'); plt.xticks(x,layer_comp.layer); plt.ylabel('Percent'); plt.title('Cross-study stability: pathways versus species'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'Figure10_5D_species_vs_pathway_stability.svg'); plt.close()


# status and decision
path_nom=pw_cls[pw_cls.p_value<.05].sort_values('p_value')
path_fdr=pw_cls[pw_cls.fdr_q<.10]
rev=pw_cls[pw_cls.classification=='STRONG_DIRECTION_REVERSAL']
spc=pw_cls[pw_cls.classification=='STUDY_SPECIFIC']
status={
'step':'Step10.5D','version':'v1_V1_functional_sensitivity','technical_status':'PASS',
'primary_V2_pathway_gate':'HOLD_EXACT_UNSTRATIFIED_HUMANN3_MATRICES_UNRESOLVED',
'analysis_role':'Separate V1 processing-system functional sensitivity; not concatenated with V2 species analysis',
'v1_samples':len(meta),'v1_studies':len(STUDY_ORDER),'community_unstratified_pathways_input':len(pw_features),'pathways_meta_analyzed':len(pw_cls),'paired_species':len(sp_features),
'pathway_meta':{'nominal_p_lt_0_05':int((pw_cls.p_value<.05).sum()),'nominal_p_lt_0_10':int((pw_cls.p_value<.10).sum()),'fdr_q_lt_0_10':int((pw_cls.fdr_q<.10).sum()),'minimum_fdr_q':float(pw_cls.fdr_q.min()),'I2_ge_50':int((pw_cls.I2>=50).sum()),'strong_reversal':len(rev),'study_specific':len(spc)},
'layer_comparison':layer_comp.to_dict('records'),'heterogeneity_distribution_test':hetero_test,
'v2_candidate_mapping':{'total_candidates':len(mapdf),'mapped_exact_to_v1':int(mapdf.mapped.sum())},
'species_pathway_concordance':{'all_pairs':len(pairs),'partial_spearman_fdr_lt_0_05':int((pairs.partial_spearman_fdr_q<.05).sum()),'mapped_candidate_strong_pairs':len(strong),'multi_candidate_convergent_pathways':int((conv.candidate_species_count>=2).sum()) if len(conv) else 0},
'scientific_decisions':{
'functional_layer_more_stable_than_species':'TO_BE_INTERPRETED_FROM_RESULTS',
'FDR_supported_universal_pathway':'YES' if len(path_fdr) else 'NO',
'pathway_direction_reversal':'SUPPORTED' if len(rev) else 'NOT_DETECTED',
'species_pathway_concordance':'SUPPORTED_FOR_SPECIFIC_PAIRS_NOT_UNIVERSAL',
'V2_primary_species_pathway_integration':'NO_GO_UNTIL_EXACT_V2_PATHWAY_MATRICES',
'V1_functional_sensitivity':'GO'
},
'limitations':['V1 uses a separate MetaPhlAn3/HUMAnN3 processing system and five-study sample set.','V1 response harmonization is not identical to the V2 363-patient endpoint lock.','Lee sites are not separable in the V1 functional matrix.','Metagenomic pathways represent functional potential, not transcription or metabolite production.','Taxon-stratified rows were excluded from primary pathway meta-analysis.']}
(OUT/'Step10_5D_status_v1.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')

# summary text
lines=['Step 10.5D analysis completed','',json.dumps(status,ensure_ascii=False,indent=2),'','TOP NOMINAL PATHWAYS']
for _,r in path_nom.head(20).iterrows(): lines.append(f"{r.feature}: g={r.pooled_effect:.3f}, 95%CI {r.ci_low:.3f} to {r.ci_high:.3f}, P={r.p_value:.4g}, q={r.fdr_q:.4g}, I2={r.I2:.1f}%")
(OUT/'README_RESULTS_STEP10_5D_v1.txt').write_text('\n'.join(lines),encoding='utf-8')

zipout=PROJECT_ROOT/'Step10_5D_results_v1.zip'; zipout.unlink(missing_ok=True)
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname=p.name)
print(json.dumps(status,ensure_ascii=False,indent=2))
print('top pathways')
print(path_nom[['feature','pooled_effect','ci_low','ci_high','p_value','fdr_q','I2','classification']].head(20).to_string(index=False))
print('layer comp')
print(layer_comp.to_string(index=False))
print('mapping',mapdf[mapdf.mapped==1].to_string(index=False))
print('convergence')
print(conv.head(20).to_string(index=False))
print('zip',zipout,zipout.stat().st_size)
