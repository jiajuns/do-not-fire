import glob, pandas as pd
from scipy import stats
fs=glob.glob('results/raw/setcomm_vlw/*.csv')
d=pd.concat([pd.read_csv(f) for f in fs if 'coverage' not in f],ignore_index=True)
for c in ['total_travel_distance','overlap_ratio','total_communication_volume','map_id','seed']:
    d[c]=pd.to_numeric(d[c],errors='coerce')
ext=d['method_name'].str.extract(r'^(HOMrel|HOMloss|HETrel|HETloss)_(StagEven|CWD|VLW)_')
d['reg']=ext[0]; d['cond']=ext[1]; d=d.dropna(subset=['cond'])
rows=[]
def pair(sub,A,B,reg,lab):
    X=sub[sub['cond']==A].set_index(['map_id','seed']); Y=sub[sub['cond']==B].set_index(['map_id','seed'])
    X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
    if len(idx)<8: return
    for met in ['total_travel_distance','overlap_ratio']:
        a=X.loc[idx,met].values; c=Y.loc[idx,met].values
        try: _,p=stats.wilcoxon(a,c)
        except: p=float('nan')
        rows.append(dict(regime=reg,cmp=lab,metric=met.split('_')[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
for reg in ['HOMrel','HOMloss','HETrel','HETloss']:
    sub=d[d['reg']==reg]
    pair(sub,'VLW','StagEven',reg,'vlw_vs_even'); pair(sub,'CWD','StagEven',reg,'cwd_vs_even'); pair(sub,'VLW','CWD',reg,'vlw_vs_cwd')
o=pd.DataFrame(rows)
print('=== VLW partial paired (neg pct = vlw better) ===')
print(o.to_string(index=False) if len(o) else 'no regime with n>=8 yet')
print('--- rows per regime/cond ---')
print(d.groupby(['reg','cond']).size().to_string())
