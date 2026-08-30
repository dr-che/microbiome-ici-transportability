#!/usr/bin/env python3
import sys, importlib.util, numpy as np
from pathlib import Path
from joblib import Parallel, delayed
MODULE_PATH=Path('/mnt/data/melanoma_transportability_review/analysis3_run.py')
spec=importlib.util.spec_from_file_location('a3',MODULE_PATH)
a3=importlib.util.module_from_spec(spec);sys.modules['a3']=a3;spec.loader.exec_module(a3)
cap=int(sys.argv[1]); data=a3.load_data(); domains=sorted(np.unique(data.domains)); counts={d:int(np.sum(data.domains==d)) for d in domains}; eligible=sorted([d for d,n in counts.items() if n>=a3.DOWNSAMPLE_N]); es=set(eligible)
def one(r):
    rows=a3.run_downsample_repeat(data,cap,r,eligible); pairs=a3.reciprocal_rows(rows,es)
    d=a3.summarize_directed(rows,cap,'downsample_sources_to_25_all_targets');d.update(a3.summarize_asym(pairs,cap,'downsample_sources_to_25_all_targets'));d['repeat']=r
    return d,rows
res=Parallel(n_jobs=8,prefer='threads',batch_size=1)(delayed(one)(r) for r in range(1,a3.DOWNSAMPLE_REPEATS+1))
s=[x[0] for x in res]; rows=[r for x in res for r in x[1]]
a3.OUT.mkdir(parents=True,exist_ok=True)
a3.write_tsv(a3.OUT/f'Analysis3_downsample_repeat_summary_cap{cap}.tsv',s,list(s[0].keys()))
a3.write_tsv(a3.OUT/f'Analysis3_downsample_directed_metrics_cap{cap}.tsv',rows,list(rows[0].keys()))
print(f'completed cap {cap}: {len(s)} repeats, {len(rows)} directed rows')
