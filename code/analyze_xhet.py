import glob, pandas as pd
from scipy import stats
fs=glob.glob('results/raw/setcomm_xhet/*.csv')
d=pd.concat([pd.read_csv(f) for f in fs if 'coverage' not in f],ignore_index=True)
for c in ['total_travel_distance','overlap_ratio','map_id','seed']: d[c]=pd.to_numeric(d[c],errors='coerce')
ext=d['method_name'].str.extract(r'^(h1040|h540)_(L50|L70)_N(\d+)_(StagEven|CWD|VLW)')
d['het']=ext[0]; d['loss']=ext[1]; d['N']=pd.to_numeric(ext[2],errors='coerce'); d['cond']=ext[3]
d=d.dropna(subset=['cond'])
rows=[]
def pair(sub,A,B,tags,lab):
    X=sub[sub['cond']==A].set_index(['map_id','seed']); Y=sub[sub['cond']==B].set_index(['map_id','seed'])
    X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
    if len(idx)<8: return
    for met in ['total_travel_distance','overlap_ratio']:
        a=X.loc[idx,met].values; c=Y.loc[idx,met].values
        try: _,p=stats.wilcoxon(a,c)
        except: p=float('nan')
        rows.append(dict(**tags,cmp=lab,metric=met.split('_')[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
for ht in ['h1040','h540']:
    for lt in ['L50','L70']:
        for N in [16,20]:
            sub=d[(d['het']==ht)&(d['loss']==lt)&(d['N']==N)]
            pair(sub,'VLW','StagEven',dict(het=ht,loss=lt,N=N),'vlw_vs_even')
            pair(sub,'VLW','CWD',dict(het=ht,loss=lt,N=N),'vlw_vs_cwd')
o=pd.DataFrame(rows)
print('=== XHET partial (neg pct = vlw better; want significant vlw_vs_even) ===')
print(o.to_string(index=False) if len(o) else 'no cell with n>=8 yet')
print('--- rows per cell ---'); print(d.groupby(['het','loss','N','cond']).size().to_string())
