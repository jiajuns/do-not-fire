import glob, pandas as pd
from scipy import stats
fs=[f for f in glob.glob('results/raw/setcomm_desync2/*.csv') if 'coverage' not in f]
d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
for c in ['total_travel_distance','overlap_ratio','map_id','seed']: d[c]=pd.to_numeric(d[c],errors='coerce')
def pick(n): return d[d['method_name']==n]
rows=[]
def pair(A,B,N,lab):
    X=pick(A).set_index(['map_id','seed']); Y=pick(B).set_index(['map_id','seed'])
    X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
    if len(idx)<8: return
    for met in ['total_travel_distance','overlap_ratio']:
        a=X.loc[idx,met].values; c=Y.loc[idx,met].values
        try: _,p=stats.wilcoxon(a,c)
        except: p=float('nan')
        rows.append(dict(N=N,cmp=lab,metric=met.split('_')[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
for N in [8,16]:
    pair(f"DESYNCsync_N{N}",f"Sync_N{N}",N,"desync_vs_sync")
    pair(f"DESYNCsync_N{N}",f"StagIdeal_N{N}",N,"desync_vs_ideal")
    pair(f"Sync_N{N}",f"StagIdeal_N{N}",N,"sync_vs_ideal")
    pair(f"DESYNCeven_N{N}",f"StagIdeal_N{N}",N,"desyncEven_vs_ideal")
o=pd.DataFrame(rows)
print('=== DESYNC2 (neg=first better). WIN=desync_vs_sync sig negative & desync_vs_ideal small ===')
print(o.to_string(index=False) if len(o) else 'none')
