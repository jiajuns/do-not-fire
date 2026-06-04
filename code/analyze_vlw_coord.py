import glob, pandas as pd
from scipy import stats
fs=glob.glob('results/raw/setcomm_vlw/*.csv')
d=pd.concat([pd.read_csv(f) for f in fs if 'coverage' not in f],ignore_index=True)
METRICS=['total_travel_distance','overlap_ratio','action_change_count','action_change_rate',
         'average_selected_overlap_cost_before_penalty','average_selected_overlap_cost_after_penalty']
for c in METRICS+['map_id','seed']:
    if c in d.columns: d[c]=pd.to_numeric(d[c],errors='coerce')
ext=d['method_name'].str.extract(r'^(HOMrel|HOMloss|HETrel|HETloss)_(StagEven|CWD|VLW)_')
d['reg']=ext[0]; d['cond']=ext[1]; d=d.dropna(subset=['cond'])
rows=[]
def pair(sub,A,B,reg,lab):
    X=sub[sub['cond']==A].set_index(['map_id','seed']); Y=sub[sub['cond']==B].set_index(['map_id','seed'])
    X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
    if len(idx)<8: return
    for met in METRICS:
        if met not in d.columns: continue
        a=X.loc[idx,met].values; c=Y.loc[idx,met].values
        if pd.isna(a).all() or pd.isna(c).all(): continue
        try: _,p=stats.wilcoxon(a,c)
        except: p=float('nan')
        base=c.mean()
        pct=round(float(100*(a.mean()-c.mean())/base),2) if base!=0 else float('nan')
        rows.append(dict(regime=reg,cmp=lab,metric=met[:22],pct=pct,n=len(idx),p=round(float(p),4)))
for reg in ['HETrel','HETloss']:
    sub=d[d['reg']==reg]
    pair(sub,'VLW','StagEven',reg,'vlw_vs_even'); pair(sub,'CWD','StagEven',reg,'cwd_vs_even')
o=pd.DataFrame(rows)
print('=== VLW coordination-proximal metrics, HET regimes (neg pct = better) ===')
print(o.to_string(index=False) if len(o) else 'none')
