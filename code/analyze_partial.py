import glob, pandas as pd
from scipy import stats
frames=[]
for g in ['setcomm_grad','setcomm_ksweep','setcomm_nsweep','setcomm_power']:
    for f in glob.glob(f'results/raw/{g}/*.csv'):
        try:
            df=pd.read_csv(f); df['grp']=g; frames.append(df)
        except Exception: pass
d=pd.concat(frames,ignore_index=True)
for c in ['total_travel_distance','overlap_ratio','total_communication_volume','map_id','seed']:
    d[c]=pd.to_numeric(d[c],errors='coerce')
ext=d['method_name'].str.extract(r'^(clustered|default|dispersed)_(StagEven|CWD|Submod)_N(\d+)_(\d+)')
d['spawn']=ext[0]; d['cond']=ext[1]
d['N']=pd.to_numeric(ext[2],errors='coerce'); d['T']=pd.to_numeric(ext[3],errors='coerce')
print('total rows parsed:', len(d), ' with cond:', d['cond'].notna().sum())
d=d.dropna(subset=['cond'])
rows=[]
def pair(sub,A,B,tags,lab):
    X=sub[sub['cond']==A].set_index(['map_id','seed']); Y=sub[sub['cond']==B].set_index(['map_id','seed'])
    X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]
    idx=X.index.intersection(Y.index)
    if len(idx)<8: return
    for met in ['total_travel_distance','overlap_ratio']:
        a=X.loc[idx,met].values; c=Y.loc[idx,met].values
        try: _,p=stats.wilcoxon(a,c)
        except Exception: p=float('nan')
        rows.append(dict(**tags,cmp=lab,metric=met.split('_')[0],
                         pct=round(100*(a.mean()-c.mean())/c.mean(),2),n=len(idx),p=round(float(p),4)))
for T in [4,6,8,12,16]:
    s=d[(d['grp']=='setcomm_ksweep')&(d['N']==16)&(d['T']==T)&(d['spawn']=='clustered')]
    pair(s,'Submod','StagEven',dict(study='ksweep',k=round(16/T,2),N=16,T=T),'sub_vs_even')
for N in [4,8,12,16,20,24]:
    s=d[(d['grp']=='setcomm_nsweep')&(d['N']==N)&(d['T']==8)&(d['spawn']=='clustered')]
    pair(s,'Submod','StagEven',dict(study='nsweep',k=round(N/8,2),N=N,T=8),'sub_vs_even')
for sp in ['clustered','default','dispersed']:
    s=d[(d['grp']=='setcomm_grad')&(d['N']==16)&(d['T']==8)&(d['spawn']==sp)]
    pair(s,'Submod','StagEven',dict(study='grad',spawn=sp,N=16,T=8),'sub_vs_even')
    pair(s,'Submod','CWD',dict(study='grad',spawn=sp,N=16,T=8),'sub_vs_cwd')
s=d[(d['grp']=='setcomm_power')&(d['N']==16)&(d['T']==8)&(d['spawn']=='clustered')]
pair(s,'Submod','StagEven',dict(study='power',spawn='clustered',N=16,T=8),'sub_vs_even')
pair(s,'Submod','CWD',dict(study='power',spawn='clustered',N=16,T=8),'sub_vs_cwd')
o=pd.DataFrame(rows)
print('\n=== PARTIAL paired (neg pct = submod better; overlap = coupling-sensitive) ===')
print(o.to_string(index=False) if len(o) else 'none with n>=8 yet')
print('\n--- comm volume parity (clustered N16 T8) ---')
sv=d[(d['N']==16)&(d['T']==8)&(d['spawn']=='clustered')].groupby('cond')['total_communication_volume'].mean().round(0)
print(sv.to_string())
