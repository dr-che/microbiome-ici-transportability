from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--results-dir',default='reproduced/step10_10/defensive_results')
parser.add_argument('--output-dir',default='reproduced/figures')
args=parser.parse_args()
ROOT=Path(args.root).resolve()
RES=ROOT/args.results_dir
OUT=ROOT/args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})

metrics=pd.read_csv(RES/'Step10_10A_bias_correction_overall_metrics.tsv',sep='\t')
delta=pd.read_csv(RES/'Step10_10A_bias_correction_delta_bootstrap.tsv',sep='\t')
ep=pd.read_csv(RES/'Step10_10B_endpoint_direction_transfer_summary.tsv',sep='\t')
qap=pd.read_csv(RES/'Step10_10B_endpoint_class_QAP_inference.tsv',sep='\t').iloc[0]
strict=pd.read_csv(RES/'Step10_10B_strict_ORR_recoding_audit.tsv',sep='\t').iloc[0]

fig,axs=plt.subplots(2,2,figsize=(11.5,8.2))
# A
ax=axs[0,0]
methods=['baseline_CLR','target_percentile','CORAL_target_to_source']
labels=['Baseline CLR','Target percentile','CORAL']
x=np.arange(2); w=.23
for j,(m,l) in enumerate(zip(methods,labels)):
 s=metrics[metrics.method==m].set_index('scheme')
 ax.bar(x+(j-1)*w,[s.loc['LODO','roc_auc'],s.loc['LOSFO','roc_auc']],width=w,label=l)
ax.axhline(.5,ls='--',lw=1)
ax.set_xticks(x); ax.set_xticklabels(['Leave one domain out','Publication-family out'])
ax.set_ylabel('ROC AUC'); ax.set_ylim(.38,.56); ax.set_title('A  Explicit correction benchmark',loc='left',fontweight='bold')
ax.legend(frameon=False)
# B
ax=axs[0,1]
order=[('LODO','target_percentile'),('LODO','CORAL_target_to_source'),('LOSFO','target_percentile'),('LOSFO','CORAL_target_to_source')]
d=delta.set_index(['scheme','method']).loc[order].reset_index()
y=np.arange(len(d))
ax.errorbar(d.delta_auc,y,xerr=[d.delta_auc-d.ci_low,d.ci_high-d.delta_auc],fmt='o',capsize=3)
ax.axvline(0,ls='--',lw=1)
ax.set_yticks(y); ax.set_yticklabels(['LODO: percentile','LODO: CORAL','Family-out: percentile','Family-out: CORAL'])
ax.set_xlabel('Change in ROC AUC versus baseline')
ax.set_title('B  Cluster-bootstrap AUC differences',loc='left',fontweight='bold')
# C
ax=axs[1,0]
mm=metrics.copy(); mm['label']=mm.method.map(dict(zip(methods,labels)))
for scheme,marker in [('LODO','o'),('LOSFO','s')]:
 s=mm[mm.scheme==scheme]
 ax.scatter(s.brier,s.roc_auc,marker=marker,s=55,label=scheme)
 for _,r in s.iterrows(): ax.annotate(r.label,(r.brier,r.roc_auc),xytext=(4,3),textcoords='offset points',fontsize=6)
ax.axhline(.5,ls='--',lw=1)
ax.set_xlabel('Brier score (lower is better)'); ax.set_ylabel('ROC AUC')
ax.set_title('C  Discrimination–calibration trade-off',loc='left',fontweight='bold')
ax.legend(frameon=False)
# D
ax=axs[1,1]
ep['dir']=ep.train_endpoint_class.str.replace('_COMPATIBLE','',regex=False).str.replace('_',' ')+' → '+ep.test_endpoint_class.str.replace('_COMPATIBLE','',regex=False).str.replace('_',' ')
ep=ep.sort_values('mean_auc')
ax.barh(np.arange(len(ep)),ep.mean_auc)
ax.axvline(.5,ls='--',lw=1)
ax.set_yticks(np.arange(len(ep))); ax.set_yticklabels(ep.dir)
ax.set_xlabel('Mean directed-transfer ROC AUC')
ax.set_title('D  Endpoint-definition compatibility',loc='left',fontweight='bold')
ax.text(.02,-.31,f"Same vs cross endpoint-class ΔAUC={qap.observed_same_minus_cross_mean_auc:.3f}; QAP P={qap.qap_two_sided_p:.3f}\nStrict RECIST audit: {int(strict.strict_ORR_evaluable_n)}/{int(strict.original_n)} evaluable, 0 discordant labels",transform=ax.transAxes,fontsize=7)
fig.suptitle('Bias-correction and endpoint-definition sensitivity analyses',fontweight='bold',y=.995)
fig.tight_layout(rect=[0,0.02,1,.97])
for ext,dpi in [('png',300),('tiff',600),('pdf',300)]:
 p=OUT/f'Figure5_bias_correction_and_endpoint_sensitivity.{ext}'
 if ext=='tiff': fig.savefig(p,dpi=dpi,pil_kwargs={'compression':'tiff_lzw'})
 else: fig.savefig(p,dpi=dpi)
plt.close(fig)
print(OUT)
