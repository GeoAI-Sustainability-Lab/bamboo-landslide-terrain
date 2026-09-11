# -*- coding: utf-8 -*-
"""Stage 10 - event rainfall exposure as a covariate.
Inputs: per-event IMERG V07 rasters (total, max 24-h, max 72-h; mm; 0.1 deg) produced by gee/gee_rain_rasters.py,
CHIRPS 2018-2025 climatology (mean annual precipitation, mean annual maximum daily precipitation; 0.05 deg).
Models: (a) event fixed effects with island-wide controls inside 5 km affected cells (as S6.event_fixed_effects)
plus point-level event rainfall; (b) the same on the new control pool, 5 controls per case, 25 draws (seeds 1000+k,
identical to S6.new_pool_models_mc); (c) island-wide design plus rainfall climatology.
Output: verified_facts_s10.json, rain_exposure_table.csv (per case)."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, glob, os, time, warnings, numpy as np, pandas as pd, rasterio
import statsmodels.formula.api as smf
from scipy.stats import norm
from pyproj import Transformer
warnings.filterwarnings('ignore'); t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
OUT={}
def fact(fid,value,definition,source,cls,note=""): OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note)
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]; COVS=['slope','north','east','planc','profc','elev']; CELL=5000.
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv'); POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
FP={k:set(map(tuple,v)) for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
WINDOWS=json.load(open(ROOT+'/gee/event_windows_curated.json'))
tr=Transformer.from_crs(3826,4326,always_xy=True)
# ---------------------------------------------------------------- rasters
RAST={}
for ev in FP:
    fn=ROOT+'/gee/rain/'+ev.replace('、','_').replace('/','_')+'.tif'
    if not os.path.exists(fn): log(f'missing raster {ev}'); continue
    with rasterio.open(fn) as src: RAST[ev]=(src.read(),src.transform,src.nodata)
with rasterio.open(ROOT+'/gee/rain/chirps_clim_2018_2025.tif') as src: CLIM=(src.read(),src.transform,src.nodata)
def sample(rst,xs,ys):
    arr,T,nd=rst; lon,lat=tr.transform(np.asarray(xs,float),np.asarray(ys,float))
    col=np.floor((lon-T.c)/T.a).astype(int); row=np.floor((lat-T.f)/T.e).astype(int)
    ok=(row>=0)&(row<arr.shape[1])&(col>=0)&(col<arr.shape[2])
    out=np.full((arr.shape[0],len(lon)),np.nan)
    out[:,ok]=arr[:,row[ok],col[ok]]
    return out
for D in (CASE,CTRL,POOL):
    c=sample(CLIM,D.x.values,D.y.values); D['map_mm']=c[0]; D['annmax_mm']=c[1]
log(f"rasters loaded for {len(RAST)} events; CHIRPS climatology sampled")
# ---------------------------------------------------------------- case exposure table
ca=CASE[CASE.ft.isin(K1)].copy()
vals=np.full((3,len(ca)),np.nan)
for ev,g in ca.groupby('event'):
    if ev in RAST:
        vals[:,ca.index.get_indexer(g.index)]=sample(RAST[ev],g.x.values,g.y.values)
ca['total']=vals[0]; ca['max24']=vals[1]; ca['max72']=vals[2]
ca[['lid','event','ft','slope','x','y','total','max24','max72','map_mm','annmax_mm']].to_csv(ROOT+'/verify/rain_exposure_table.csv',index=False)
expo={'bamboo_cases':{k:round(float(ca[ca.ft==BAMBOO][k].median()),1) for k in ['max24','max72','total']},
      'broadleaf_cases':{k:round(float(ca[ca.ft==BROAD][k].median()),1) for k in ['max24','max72','total']},
      'share_cases_with_max24_lt_50mm':round(float((ca.max24<50).mean()),3),'n_missing':int(ca.max24.isna().sum())}
log(f"case exposure medians (max24): bamboo {expo['bamboo_cases']['max24']} broadleaf {expo['broadleaf_cases']['max24']}")
# ---------------------------------------------------------------- (a) event FE with island-wide controls in 5 km cells + rainfall
cell=lambda df,c=CELL: list(zip(np.floor(df.x/c).astype(int),np.floor(df.y/c).astype(int)))
co=CTRL[CTRL.ft.isin(K1)].copy().reset_index(drop=True); co['cid']=np.arange(len(co)); co['cell']=cell(co); ca['cell']=cell(ca)
rows=[]
for ev,cl in FP.items():
    cs=ca[ca.event==ev]; ct=co[co.cell.isin(cl)]
    if len(cs)==0 or len(ct)==0 or ev not in RAST: continue
    A=cs[COVS+['ft','total','max24','max72']].copy(); A['case']=1; A['uid']=['C'+str(i) for i in cs.index]
    B=ct[COVS+['ft']].copy(); r=sample(RAST[ev],ct.x.values,ct.y.values); B['total']=r[0]; B['max24']=r[1]; B['max72']=r[2]; B['case']=0; B['uid']=['B'+str(i) for i in ct.cid]
    t=pd.concat([A,B]); t['event']=ev; rows.append(t)
DM=pd.concat(rows,ignore_index=True); DM['bamboo']=(DM.ft==BAMBOO).astype(int); DM['slope2']=DM.slope**2
for k in ['total','max24','max72']: DM['l'+k]=np.log1p(DM[k])
DM=DM.dropna(subset=['lmax24','ltotal','lmax72']).reset_index(drop=True)
def fit(D,form,term='bamboo:slope'):
    m=smf.logit(form,data=D).fit(disp=0,maxiter=300)
    b,s=m.params[term],m.bse[term]
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5),n=int(len(D)),n_case=int(D.case.sum()),n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum())),m
FE="case ~ C(event) + slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
res={}
res['fe_baseline'],_=fit(DM,FE)
res['fe_plus_max24'],m1=fit(DM,FE+"+lmax24")
res['fe_plus_max24_total'],m2=fit(DM,FE+"+lmax24+ltotal")
res['fe_plus_max72'],m3=fit(DM,FE+"+lmax72")
res['rain_coefficients']={'log_max24_in_fe_plus_max24':dict(beta=round(float(m1.params['lmax24']),4),OR_per_log_unit=round(math.exp(m1.params['lmax24']),3),p=round(float(m1.pvalues['lmax24']),6)),
                          'log_max72_in_fe_plus_max72':dict(beta=round(float(m3.params['lmax72']),4),OR_per_log_unit=round(math.exp(m3.params['lmax72']),3),p=round(float(m3.pvalues['lmax72']),6))}
# within-event exposure of bamboo vs broadleaf controls (is bamboo land less exposed?)
ctrl_rows=DM[DM.case==0]
diff=ctrl_rows.groupby('event').apply(lambda g: (g[g.bamboo==1].max24.mean()-g[g.bamboo==0].max24.mean()) if (g.bamboo==1).sum()>=5 else np.nan).dropna()
res['within_event_control_exposure']={'events_with_ge5_bamboo_controls':int(len(diff)),'mean_difference_max24_bamboo_minus_broadleaf_mm':round(float(diff.mean()),1),'median_difference_mm':round(float(diff.median()),1)}
log(f"FE baseline {res['fe_baseline']['OR']}  +max24 {res['fe_plus_max24']['OR']}  +max24+total {res['fe_plus_max24_total']['OR']}  +max72 {res['fe_plus_max72']['OR']}")
fact('S10.event_fe_with_rainfall',res,"event fixed-effects interaction with point-level event rainfall (IMERG V07 max 24-h, max 72-h, total over the event window) as covariates; island-wide controls inside 5 km affected cells","gee/rain rasters + rebuilt table","B",
     note="rainfall at 0.1 degree; windows in gee/event_windows_curated.json")
# ---------------------------------------------------------------- (b) new pool 5:1, 25 draws, with rainfall
po=POOL[POOL.ft.isin(K1)].copy(); po['cell']=cell(po)
def summarise(ors,ps):
    ors=np.array(ors); ps=np.array(ps)
    return dict(OR_mean=round(float(ors.mean()),4),OR_range_mc=[round(float(np.percentile(ors,2.5)),4),round(float(np.percentile(ors,97.5)),4)],p_median=float(np.median(ps)),p_max=float(ps.max()),share_p_lt_05=float((ps<0.05).mean()),n_draws=len(ors))
o0=[];p0=[];o1=[];p1=[]
for k in range(25):
    rg=np.random.default_rng(1000+k); rows=[]
    for ev,cl in FP.items():
        cs=ca[ca.event==ev]
        if len(cs)==0 or ev not in RAST: continue
        sel=po[po.cell.isin(cl)]
        if len(sel)==0: continue
        kk=min(len(sel),max(300,5*len(cs))); sub=sel.iloc[rg.choice(len(sel),kk,replace=False)]
        A=cs[COVS+['ft','max24']].copy(); A['case']=1
        B=sub[COVS+['ft']].copy(); B['max24']=sample(RAST[ev],sub.x.values,sub.y.values)[1]; B['case']=0
        t=pd.concat([A,B]); t['event']=ev; rows.append(t)
    D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2; D['lmax24']=np.log1p(D.max24); D=D.dropna(subset=['lmax24'])
    r0,_=fit(D,FE); r1,_=fit(D,FE+"+lmax24"); o0.append(r0['OR']); p0.append(r0['p']); o1.append(r1['OR']); p1.append(r1['p'])
fact('S10.new_pool_5to1_with_rainfall',dict(baseline=summarise(o0,p0),plus_max24=summarise(o1,p1)),"event fixed effects on the new control pool, 5 controls per case, 25 draws (seeds 1000-1024), with and without log max 24-h rainfall","gee/rain rasters + new control pool","C")
log(f"new pool 5:1 baseline {np.mean(o0):.4f} -> +max24 {np.mean(o1):.4f}")
# ---------------------------------------------------------------- (c) island-wide design + climatology
a=CASE[CASE.ft.isin(K1)].copy(); a['case']=1; b=CTRL[CTRL.ft.isin(K1)].copy(); b['case']=0
D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2; D['lmap']=np.log(D.map_mm); D['lannmax']=np.log(D.annmax_mm)
D=D.dropna(subset=['lmap','lannmax'])
base="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
c0,_=fit(D,base); c1,mm1=fit(D,base+"+lmap"); c2,mm2=fit(D,base+"+lmap+lannmax")
fact('S10.island_wide_with_climatology',dict(baseline=c0,plus_mean_annual_precip=c1,plus_map_and_annual_max_daily=c2,
     climatology_coefficients=dict(log_map=round(float(mm1.params['lmap']),3),log_map_p=round(float(mm1.pvalues['lmap']),5),log_annmax=round(float(mm2.params['lannmax']),3),log_annmax_p=round(float(mm2.pvalues['lannmax']),5)),
     medians=dict(bamboo_ctrl_map=round(float(b[b.ft==BAMBOO].map_mm.median()),0),broadleaf_ctrl_map=round(float(b[b.ft==BROAD].map_mm.median()),0),bamboo_ctrl_annmax=round(float(b[b.ft==BAMBOO].annmax_mm.median()),0),broadleaf_ctrl_annmax=round(float(b[b.ft==BROAD].annmax_mm.median()),0))),
     "island-wide design with CHIRPS 2018-2025 rainfall climatology (log mean annual precipitation, log mean annual maximum daily precipitation) as covariates","gee/rain/chirps_clim_2018_2025.tif","B")
log(f"island-wide baseline {c0['OR']} +MAP {c1['OR']} +MAP+annmax {c2['OR']}")
fact('S10.rainfall_windows',{ev:dict(window=v['window'],peak_day=v['peak_day'],peak_area_mean_mm=v['peak_mm'],window_total_area_mean_mm=v['window_total_area_mean_mm']) for ev,v in WINDOWS.items()},
     "event rainfall windows (event-name dates or typhoon warning periods padded by one day) with the IMERG daily area-mean peak inside the window","gee/gee_rain_rasters.py","A")
fact('S10.case_exposure',expo,"median event rainfall at bamboo and broadleaf landslide cases","rain_exposure_table.csv","B")
json.dump(dict(facts=OUT),open(ROOT+'/verify/verified_facts_s10.json','w'),ensure_ascii=False,indent=1)
log('done')
