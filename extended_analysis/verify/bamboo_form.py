# -*- coding: utf-8 -*-
"""Bamboo growth form (single-culm/running C800 vs clumping C700) from the inventory codes:
area shares, and the form at case / control / new-pool points, by elevation and slope."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import geopandas as gpd, pandas as pd, numpy as np, pyogrio, gc, warnings, json, time
warnings.filterwarnings('ignore'); t0=time.time()
F4=RAW+"/forest4/f4.shp"
# 1. polygon-level composition
df=pyogrio.read_dataframe(F4,columns=['TypeName','MajorTreeI','SlaveTreeI','Area_Ha'],read_geometry=False)
bam=df[df.TypeName.isin(['竹林','竹闊混淆林'])]
print("=== inventory: bamboo-containing polygons by TypeName x MajorTreeI (area ha) ===")
print(bam.pivot_table(index='TypeName',columns='MajorTreeI',values='Area_Ha',aggfunc='sum').round(0).fillna(0).to_string())
print("\n=== 竹闊混淆林 SlaveTreeI (area ha) top ===")
print(bam[bam.TypeName=='竹闊混淆林'].groupby('SlaveTreeI').Area_Ha.sum().sort_values(ascending=False).head(8).round(0).to_string())
# 2. point join with codes
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv'); POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
P=pd.concat([CASE.assign(kind='case',key=CASE.lid),
             CTRL.assign(kind='ctrl',key=CTRL.bg_lid),
             POOL.assign(kind='pool',key=['p'+str(i) for i in range(len(POOL))])],ignore_index=True)
P=P[P.ft.isin(['竹林','竹闊混淆林'])].reset_index(drop=True)
G=gpd.GeoDataFrame(P,geometry=gpd.points_from_xy(P.x,P.y),crs=3826)
res=[]; N=int(pyogrio.read_info(F4)['features'])
for s0 in range(0,N,40000):
    FT=pyogrio.read_dataframe(F4,columns=['TypeName','MajorTreeI','SlaveTreeI'],skip_features=s0,max_features=40000,on_invalid='ignore')
    FT=FT[FT.TypeName.isin(['竹林','竹闊混淆林'])&FT.geometry.notna()].set_crs(3826,allow_override=True); FT['geometry']=FT.geometry.buffer(0)
    j=gpd.sjoin(G,FT[['MajorTreeI','SlaveTreeI','geometry']],how='inner',predicate='within')
    res.append(pd.DataFrame({'key':j.key.values,'MajorTreeI':j.MajorTreeI.values,'SlaveTreeI':j.SlaveTreeI.values}))
    del FT,j; gc.collect()
R=pd.concat(res).drop_duplicates('key')
P=P.merge(R,on='key',how='left')
P['form']=P.MajorTreeI.map({'C800':'單桿狀竹(running)','C700':'叢生狀竹(clumping)'}).fillna(P.MajorTreeI)
P.to_csv(ROOT+'/verify/bamboo_points_form.csv',index=False)
print(f"\njoined {len(P)} bamboo-containing points  {time.time()-t0:.0f}s")
out={}
for ft in ['竹林','竹闊混淆林']:
    for kind in ['case','ctrl','pool']:
        s=P[(P.ft==ft)&(P.kind==kind)]
        vc=s.form.value_counts()
        out[f'{ft}|{kind}']={k:int(v) for k,v in vc.items()}
        print(f"{ft:6s} {kind:5s} n={len(s):6d}  "+"  ".join(f"{k}:{v} ({100*v/len(s):.1f}%)" for k,v in vc.items()))
# 3. by elevation & slope band, pure bamboo, pool (landscape) and cases
def band_table(s,col,bins,label):
    t=pd.crosstab(pd.cut(s[col],bins),s.form)
    t['running_share_%']=(100*t.get('單桿狀竹(running)',0)/t.sum(axis=1)).round(1)
    print(f"\n--- 竹林 {label}: form by {col} band ---"); print(t.to_string())
    return t
pb=P[(P.ft=='竹林')&(P.kind=='pool')]; pc=P[(P.ft=='竹林')&(P.kind=='case')]
band_table(pb,'elev',[0,300,600,900,1200,1500,2000,4000],'landscape (new pool)')
band_table(pc,'elev',[0,300,600,900,1200,1500,2000,4000],'landslide cases')
band_table(pb,'slope',[0,10,20,30,40,50,90],'landscape (new pool)')
band_table(pc,'slope',[0,10,20,30,40,50,90],'landslide cases')
print("\n--- 竹林 median elevation / slope by form (pool) ---")
print(pb.groupby('form')[['elev','slope']].median().round(1).to_string())
print("\n--- 竹林 median elevation / slope by form (cases) ---")
print(pc.groupby('form')[['elev','slope']].median().round(1).to_string())
# 4. interaction by growth form: form-specific slope interaction vs broadleaf
import statsmodels.formula.api as smf, math
from scipy.stats import norm
COVS=['slope','north','east','planc','profc','elev']
BR_case=CASE[CASE.ft=='闊葉樹林型'][COVS].assign(case=1,grp='broadleaf')
BR_pool=POOL[POOL.ft=='闊葉樹林型'][COVS].assign(case=0,grp='broadleaf')
def fit_form(form):
    a=pc[pc.form==form][COVS].assign(case=1,grp='bamboo'); b=pb[pb.form==form][COVS].assign(case=0,grp='bamboo')
    D=pd.concat([a,b,BR_case,BR_pool],ignore_index=True); D['bamboo']=(D.grp=='bamboo').astype(int); D['slope2']=D.slope**2
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope",data=D).fit(disp=0)
    bb,ss=m.params['bamboo:slope'],m.bse['bamboo:slope']; b0=m.params['bamboo']
    return dict(n_case=len(a),n_ctrl=len(b),OR=round(math.exp(bb),4),ci=[round(math.exp(bb-1.96*ss),4),round(math.exp(bb+1.96*ss),4)],
                p=float(2*(1-norm.cdf(abs(bb/ss)))),main_OR=round(math.exp(b0),3),
                linear_crossover_deg=round(float(-b0/bb),1) if bb>0 else None)
print("\n=== slope interaction vs broadleaf, by bamboo growth form (new pool controls) ===")
forms={}
for f in ['單桿狀竹(running)','叢生狀竹(clumping)']:
    try:
        forms[f]=fit_form(f); print(f, forms[f])
    except Exception as e: print(f,"ERR",e)
json.dump(dict(inventory_area=bam.pivot_table(index='TypeName',columns='MajorTreeI',values='Area_Ha',aggfunc='sum').round(1).fillna(0).to_dict(),
               points=out,by_form=forms),open(ROOT+'/verify/bamboo_form.json','w'),ensure_ascii=False,indent=1)
