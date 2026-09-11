# -*- coding: utf-8 -*- 
"""Stage 6a: event footprints, event-matched models, new control pool. Frozen seeds."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings,gc
import numpy as np,pandas as pd,geopandas as gpd,pyogrio,rasterio
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from scipy.stats import norm
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
P=json.load(open(ROOT+'/verify/verified_facts_s5.json')); OUT=P['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,v,d,s,c,n=""):
    OUT[fid]=dict(value=v,definition=d,source=s,cls=c,note=n); return v
BAMBOO='竹林'; BROAD='闊葉樹林型'; MIXED='竹闊混淆林'
COVS=['slope','north','east','planc','profc','elev']
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
CELL=5000.; SEED_POOL=20260901; SEED_EVCELL=11; PER_CELL=260
U=RAW
DEM=f"{U}/Dataset/01_SOURCE/Terrain_Canopy/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif"
F4=RAW+"/forest4/f4.shp"
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')
lc=pd.read_csv(ROOT+'/../data/landslide_centroids.csv')

def build(a,b,bam,keep):
    A=a[a.ft.isin(keep)][COVS+['ft']].copy(); A['case']=1
    B=b[b.ft.isin(keep)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['bamboo']=D.ft.isin(bam).astype(int); D['slope2']=D.slope**2
    return D
def res(m,key='bamboo:slope',D=None):
    b,s=m.params[key],m.bse[key]
    r=dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
           p=float(2*(1-norm.cdf(abs(b/s)))))
    if D is not None:
        r.update(n=int(len(D)),n_case=int(D.case.sum()),
                 n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),
                 n_bamboo_ctrl=int(((D.case==0)&(D.bamboo==1)).sum()))
    return r
def fit(D,form=FORM,**kw): return res(smf.logit(form,data=D).fit(disp=0,**kw),D=D)
def clogit(D,strat,X=('slope','slope2','north','east','planc','profc','elev','bamboo','bam_slope')):
    D=D.copy(); D['bam_slope']=D.bamboo*D.slope
    t=np.where(D.case==1,1.0,2.0); st=(D.case==1).astype(int)
    m=PHReg(t,D[list(X)].values,status=st.values,strata=D[strat].values,ties='breslow').fit()
    i=list(X).index('bam_slope'); b,s=m.params[i],m.bse[i]
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=float(2*(1-norm.cdf(abs(b/s)))),n=int(len(D)),strata=int(pd.Series(D[strat]).nunique()),
                n_case=int(D.case.sum()),n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()))

log("6a.1 event footprints (frozen: 5 km grid over each polygon's bounding box)")
G=gpd.read_file(ROOT+'/verify/all_polys.gpkg',layer='ls').merge(lc[['lid','trigger']],left_on='pid',right_on='lid')
R=G[G.trigger=='降雨'].copy()
bb=R.geometry.bounds; R=R.assign(**{c:bb[c].values for c in ['minx','miny','maxx','maxy']})
def cellset(grp,c):
    out=set()
    for x0,y0,x1,y1 in grp[['minx','miny','maxx','maxy']].values:
        for cx in range(int(np.floor(x0/c)),int(np.floor(x1/c))+1):
            for cy in range(int(np.floor(y0/c)),int(np.floor(y1/c))+1): out.add((cx,cy))
    return out
FP={ev:cellset(g,CELL) for ev,g in R.groupby('Events')}
json.dump({k:[list(t) for t in v] for k,v in FP.items()},open(ROOT+'/verify/event_cells_5km.json','w'),ensure_ascii=False)
fact('S6.footprint_definition',dict(cell_m=CELL,n_events=len(FP),distinct_cells=len(set().union(*FP.values())),
     rule="a 5 km grid cell belongs to an event's footprint if it intersects the bounding box of any polygon of that event"),
 "per-event interpretation footprint","annual shapefiles","A")
log(f"   {len(FP)} rainfall events, {len(set().union(*FP.values()))} distinct 5 km cells")

log("6a.2 event fixed-effects models on the original control set")
K1=[BROAD,BAMBOO]
ca=CASE[CASE.ft.isin(K1)].copy(); co=CTRL[CTRL.ft.isin(K1)].copy().reset_index(drop=True)
co['cid']=np.arange(len(co))
cell=lambda df,c=CELL: list(zip(np.floor(df.x/c).astype(int),np.floor(df.y/c).astype(int)))
ca['cell']=cell(ca); co['cell']=cell(co)
rows=[]
for ev,cl in FP.items():
    cs=ca[ca.event==ev]; ct=co[co.cell.isin(cl)]
    if len(cs)==0 or len(ct)==0: continue
    A=cs[COVS+['ft']].copy(); A['case']=1; A['uid']=['C'+str(i) for i in cs.index]
    B=ct[COVS+['ft']].copy(); B['case']=0; B['uid']=['B'+str(i) for i in ct.cid]
    t=pd.concat([A,B]); t['event']=ev; rows.append(t)
DM=pd.concat(rows,ignore_index=True); DM['bamboo']=(DM.ft==BAMBOO).astype(int); DM['slope2']=DM.slope**2
FE="case ~ C(event) + slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
r_fe=fit(DM,FE)
r_cp=res(smf.logit(FE,data=DM).fit(disp=0,cov_type='cluster',cov_kwds={'groups':DM.uid}),D=DM)
r_ce=res(smf.logit(FE,data=DM).fit(disp=0,cov_type='cluster',cov_kwds={'groups':DM.event}),D=DM)
rng=np.random.default_rng(SEED_EVCELL)
first=DM[DM.case==0].groupby('uid').event.apply(lambda s: rng.choice(s.values))
keep=set(zip(first.index,first.values))
mask=DM.apply(lambda r: True if r['case']==1 else (r['uid'],r['event']) in keep,axis=1)
r_once=fit(DM[mask],FE)
fact('S6.event_fixed_effects',dict(fe=r_fe,cluster_by_point=r_cp,cluster_by_event=r_ce,each_control_once=r_once),
 "logit with event fixed effects; controls restricted to the footprint of each event",
 "rebuilt table + 5 km footprints","B",f"seed {SEED_EVCELL} for the control-used-once draw")
log(f"   event FE {r_fe['OR']} {r_fe['ci']} p={r_fe['p']:.4g}; once {r_once['OR']}")

log("6a.3 footprint-size sensitivity")
fp={}
for c in (2000.,5000.,10000.,20000.):
    co['c2']=cell(co,c); rows=[]
    for ev,grp in R.groupby('Events'):
        cl=cellset(grp,c); cs=ca[ca.event==ev]; ct=co[co.c2.isin(cl)]
        if len(cs)==0 or len(ct)==0: continue
        A=cs[COVS+['ft']].copy(); A['case']=1; B=ct[COVS+['ft']].copy(); B['case']=0
        t=pd.concat([A,B]); t['event']=ev; rows.append(t)
    D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    fp[f'{int(c/1000)}km']=fit(D,FE)
    log(f"   {int(c/1000):2d} km grid: {fp[f'{int(c/1000)}km']['OR']} p={fp[f'{int(c/1000)}km']['p']:.4g}")
from shapely.ops import unary_union
pts=gpd.GeoDataFrame(co.copy(),geometry=gpd.points_from_xy(co.x,co.y),crs=3826)
for BUF in (2000.,5000.):
    rows=[]
    for ev,grp in R.groupby('Events'):
        u=unary_union(grp.geometry.buffer(BUF).values); sel=pts[pts.geometry.within(u)]; cs=ca[ca.event==ev]
        if len(cs)==0 or len(sel)==0: continue
        A=cs[COVS+['ft']].copy(); A['case']=1; B=sel[COVS+['ft']].copy(); B['case']=0
        t=pd.concat([A,B]); t['event']=ev; rows.append(t)
    D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    fp[f'buffer{int(BUF/1000)}km']=fit(D,FE)
    log(f"   {int(BUF/1000):2d} km buffer: {fp[f'buffer{int(BUF/1000)}km']['OR']}")
fact('S6.footprint_sensitivity',fp,"event fixed-effects interaction under six footprint definitions",
 "rebuilt table","B")

log("6a.4 conditional logistic regression (stratified Cox equivalence)")
cl_ev=clogit(DM,'event')
fact('S6.clogit_event',cl_ev,"conditional logistic regression stratified by event, original controls",
 "rebuilt table","B","fitted as a stratified Cox model with cases at t=1 and controls censored at t=2")
log(f"   clogit by event {cl_ev['OR']} {cl_ev['ci']} p={cl_ev['p']:.4g}")

log("6a.5 new control pool (seed %d, %d points per 5 km footprint cell)"%(SEED_POOL,PER_CELL))
allc=sorted(set().union(*FP.values()))
rng=np.random.default_rng(SEED_POOL)
cx=np.repeat([c[0] for c in allc],PER_CELL); cy=np.repeat([c[1] for c in allc],PER_CELL)
X=(cx+rng.random(len(cx)))*CELL; Y=(cy+rng.random(len(cy)))*CELL
src=rasterio.open(DEM); Tr=src.transform; A_=src.read(1).astype(np.float64); L=20.; NOD=src.nodata
col=np.round((X-Tr.c)/Tr.a-0.5).astype(int); row=np.round((Y-Tr.f)/Tr.e-0.5).astype(int)
inr=(row>=1)&(row<A_.shape[0]-1)&(col>=1)&(col<A_.shape[1]-1)
X,Y,row,col=X[inr],Y[inr],row[inr],col[inr]
Z1=A_[row-1,col-1];Z2=A_[row-1,col];Z3=A_[row-1,col+1];Z4=A_[row,col-1];Z5=A_[row,col];Z6=A_[row,col+1]
Z7=A_[row+1,col-1];Z8=A_[row+1,col];Z9=A_[row+1,col+1]
ok=np.all(np.vstack([Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9])!=NOD,axis=0)
gx=(Z6-Z4)/(2*L); gy=(Z8-Z2)/(2*L); asp=np.arctan2(gy,-gx)
Dc=((Z4+Z6)/2-Z5)/L**2; Ec=((Z2+Z8)/2-Z5)/L**2; Fc=(-Z1+Z3+Z7-Z9)/(4*L**2)
Gg=gx; H=-gy; den=Gg**2+H**2; den2=np.where(den==0,np.nan,den)
PL=pd.DataFrame(dict(x=X,y=Y,slope=np.degrees(np.arctan(np.hypot(gx,gy))),north=np.cos(asp),east=-np.sin(asp),
    planc=2*(Dc*H**2+Ec*Gg**2-Fc*Gg*H)/den2,profc=-2*(Dc*Gg**2+Ec*H**2+Fc*Gg*H)/den2,elev=Z5))
PL=PL[ok&np.isfinite(PL.planc)&np.isfinite(PL.profc)].reset_index(drop=True)
del A_; gc.collect()
GP=gpd.GeoDataFrame(PL,geometry=gpd.points_from_xy(PL.x,PL.y),crs=3826)
res_=[]
info=pyogrio.read_info(F4); N=int(info['features'])
for s0 in range(0,N,40000):
    FT=pyogrio.read_dataframe(F4,columns=['TypeName'],skip_features=s0,max_features=40000,on_invalid='ignore')
    FT=FT[FT.geometry.notna()].set_crs(3826,allow_override=True); FT['geometry']=FT.geometry.buffer(0)
    j=gpd.sjoin(GP,FT[['TypeName','geometry']],how='inner',predicate='within')
    res_.append(pd.DataFrame({'idx':j.index.values,'TypeName':j.TypeName.values}))
    del FT,j; gc.collect()
RR=pd.concat(res_).drop_duplicates('idx').set_index('idx')
POOL=PL.join(RR,how='inner').rename(columns={'TypeName':'ft'}).reset_index(drop=True)
POOL.to_csv(ROOT+'/verify/newctrl_pool.csv',index=False)
cnt=POOL.ft.value_counts()
fact('S6.new_control_pool',dict(seed=SEED_POOL,per_cell=PER_CELL,candidates=int(len(cx)),
     valid_terrain=int(len(PL)),on_forest_land=int(len(POOL)),
     by_type={k:int(v) for k,v in cnt.head(8).items()}),
 "an independent control sample drawn uniformly inside the union of event footprints, "
 "typed with the official forest map and given terrain from the DEM",
 "dem_20m.tif + forest4.shp","B")
log(f"   pool {len(POOL)} forest-land points; bamboo {int(cnt.get(BAMBOO,0))}, broadleaf {int(cnt.get(BROAD,0))}, mixed {int(cnt.get(MIXED,0))}")

log("6a.6 event-matched models on the new pool")
POOL['cell']=cell(POOL)
def matched(ratio,seed=1):
    rg=np.random.default_rng(seed); rows=[]
    for ev,cl in FP.items():
        cs=ca[ca.event==ev]
        if len(cs)==0: continue
        sel=POOL[POOL.ft.isin(K1)&POOL.cell.isin(cl)]
        if len(sel)==0: continue
        k=min(len(sel),max(300,ratio*len(cs)))
        sub=sel.iloc[rg.choice(len(sel),k,replace=False)]
        A=cs[COVS+['ft']].copy(); A['case']=1; B=sub[COVS+['ft']].copy(); B['case']=0
        t=pd.concat([A,B]); t['event']=ev; rows.append(t)
    D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    return D
newm={}
for ratio in (5,10,20):
    D=matched(ratio); newm[f'{ratio}to1']=fit(D,FE)
    log(f"   {ratio:2d}:1  {newm[f'{ratio}to1']['OR']} {newm[f'{ratio}to1']['ci']} p={newm[f'{ratio}to1']['p']:.4g}")
Dun=build(CASE,POOL,[BAMBOO],K1); newm['unmatched']=fit(Dun)
log(f"   unmatched  {newm['unmatched']['OR']} {newm['unmatched']['ci']} p={newm['unmatched']['p']:.4g}")
fact('S6.new_pool_models',dict(seed_draw=1,**newm),
 "event fixed-effects interaction using the new control pool at fixed control:case ratios",
 "rebuilt table + new control pool","B")

log("6a.7 event x spatial-block conditional logit")
evc={}
for c in (20000.,10000.,5000.):
    ca2=ca.assign(cc=cell(ca,c)); po2=POOL[POOL.ft.isin(K1)].assign(cc=cell(POOL[POOL.ft.isin(K1)],c))
    rg=np.random.default_rng(SEED_EVCELL); rows=[]
    for (ev,cl),cs in ca2.groupby(['event','cc']):
        sel=po2[po2.cc==cl]
        if len(sel)==0: continue
        k=min(len(sel),max(40,4*len(cs)))
        sub=sel.iloc[rg.choice(len(sel),k,replace=False)]
        A=cs[COVS+['ft']].copy(); A['case']=1; B=sub[COVS+['ft']].copy(); B['case']=0
        t=pd.concat([A,B]); t['stratum']=f"{ev}|{cl[0]}_{cl[1]}"; rows.append(t)
    D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    g=D.groupby('stratum').case.agg(['sum','size']); D=D[D.stratum.isin(g[(g['sum']>0)&(g['sum']<g['size'])].index)]
    evc[f'{int(c/1000)}km']=clogit(D,'stratum')
    log(f"   event x {int(c/1000):2d} km: {evc[f'{int(c/1000)}km']['strata']} strata, OR {evc[f'{int(c/1000)}km']['OR']} p={evc[f'{int(c/1000)}km']['p']:.4g}")
fact('S6.event_by_block_clogit',dict(seed=SEED_EVCELL,controls_per_stratum="min(available, max(40, 4x cases))",**evc),
 "conditional logistic regression stratified by event x spatial block; absorbs event-level and "
 "within-event spatial variation, including rainfall, at a scale finer than any available gridded rainfall product",
 "rebuilt table + new control pool","B")
json.dump({'facts':OUT,'log':P['log']+LOG},open(ROOT+'/verify/verified_facts_s6a.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 6a written")
