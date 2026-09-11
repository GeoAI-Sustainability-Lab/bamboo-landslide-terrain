# -*- coding: utf-8 -*-
"""Stage 11c - FE linear interaction without rainfall, control-point-clustered covariance (audit item A)."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
# -*- coding: utf-8 -*-
"""Stage 11 - response to the pre-submission audit (2026-09-07).
(1) pooled spline: delta-method band with ordinary covariance (reproduces 9.9-39.5) and with event-cluster-robust
    covariance (39 case events + all controls as one cluster, as in S5.event_cluster_robust);
(2) the same spline basis (fixed knots) in the event fixed-effects + log max-24 h rainfall design: ordinary and
    control-point-clustered covariance; distinct control points versus rows;
(3) cluster bootstrap of the whole curve (events resampled with their cases, controls resampled), pointwise
    percentile band and the range where the 97.5th percentile is below 1; number of valid draws;
(4) linear interaction in the FE(+rain) design with control-point-clustered SE.
Output: verified_facts_s11.json"""
import json, math, time, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import norm
t0=time.time(); OUT={}
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
def fact(fid,value,definition,source,cls,note=""): OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note)
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]; COVS=['slope','north','east','planc','profc','elev']
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
KNOT=35.43232547287678; LB=0.0; UB=79.36055186759805
SPL_FIX=f"bs(slope, knots=({KNOT},), degree=3, lower_bound={LB}, upper_bound={UB})"
def contrast(m,D):
    di=m.model.data.orig_exog.design_info
    med={c:float(np.median(D[c])) for c in ['north','east','planc','profc','elev']}
    extra={c:D[c].iloc[0] for c in D.columns if c not in med and c not in ('slope','bamboo','case','ft','slope2','uid','cid','cell') and D[c].dtype!=object}
    def row(b):
        r=pd.DataFrame({'slope':GRID,'bamboo':b,**med})
        for c,v in extra.items(): r[c]=v
        if 'event' in D.columns: r['event']=D['event'].iloc[0]
        return r
    d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0)))
    return d_
def band_from(d_,params,V):
    lo=d_@params; se=np.sqrt(np.einsum('ij,jk,ik->i',d_,V,d_))
    hi=np.exp(lo+1.96*se); l=np.exp(lo-1.96*se)
    p=GRID[hi<1]; contiguous=bool(len(p)>0 and np.allclose(np.diff(p),0.1,atol=1e-6))
    band=[float(p.min()),float(p.max())] if len(p) else None
    s=np.sign(lo); cx=None
    for i in range(len(GRID)-1):
        if s[i]<0 and s[i+1]>=0: cx=float(GRID[i]-lo[i]*(GRID[i+1]-GRID[i])/(lo[i+1]-lo[i])); break
    at={str(int(g)):dict(OR=round(float(np.exp(lo[k])),3),ci=[round(float(l[k]),3),round(float(hi[k]),3)]) for k,g in enumerate(GRID) if int(g)==g and int(g) in (10,20,25,30,35,40,45,50)}
    return dict(supported_band=band,contiguous=contiguous,crossover_deg=round(cx,2) if cx else None,at=at),lo,se
import os, rasterio
from pyproj import Transformer
tr=Transformer.from_crs(3826,4326,always_xy=True)
FP={k:set(map(tuple,v)) for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
RAST={}
for ev in FP:
    fn=ROOT+'/gee/rain/'+ev.replace('、','_').replace('/','_')+'.tif'
    if os.path.exists(fn):
        with rasterio.open(fn) as src: RAST[ev]=(src.read(),src.transform,src.nodata)
def sample(rst,xs,ys):
    arr,T,nd=rst; lon,lat=tr.transform(np.asarray(xs,float),np.asarray(ys,float))
    col=np.floor((lon-T.c)/T.a).astype(int); row=np.floor((lat-T.f)/T.e).astype(int)
    ok=(row>=0)&(row<arr.shape[1])&(col>=0)&(col<arr.shape[2]); out=np.full((arr.shape[0],len(lon)),np.nan); out[:,ok]=arr[:,row[ok],col[ok]]; return out
CELL=5000.; cell=lambda df: list(zip(np.floor(df.x/CELL).astype(int),np.floor(df.y/CELL).astype(int)))
ca=CASE[CASE.ft.isin(K1)].copy(); co=CTRL[CTRL.ft.isin(K1)].copy().reset_index(drop=True); co['cid']=np.arange(len(co)); co['cell']=cell(co); ca['cell']=cell(ca)
vals=np.full((3,len(ca)),np.nan)
for ev,g in ca.groupby('event'):
    if ev in RAST: vals[:,ca.index.get_indexer(g.index)]=sample(RAST[ev],g.x.values,g.y.values)
ca['total']=vals[0]; ca['max24']=vals[1]; ca['max72']=vals[2]
rows=[]
for ev,cl in FP.items():
    cs=ca[ca.event==ev]; ct=co[co.cell.isin(cl)]
    if len(cs)==0 or len(ct)==0 or ev not in RAST: continue
    A=cs[COVS+['ft','max24']].copy(); A['case']=1; A['uid']=['C'+str(i) for i in cs.index]
    Bq=ct[COVS+['ft']].copy(); r=sample(RAST[ev],ct.x.values,ct.y.values); Bq['max24']=r[1]; Bq['case']=0; Bq['uid']=['B'+str(i) for i in ct.cid]
    t=pd.concat([A,Bq]); t['event']=ev; rows.append(t)
DM=pd.concat(rows,ignore_index=True); DM['bamboo']=(DM.ft==BAMBOO).astype(int); DM['slope2']=DM.slope**2; DM['lmax24']=np.log1p(DM.max24)
DM=DM.dropna(subset=['lmax24']).reset_index(drop=True)
n_rows=int(len(DM)); n_ctrl_rows=int((DM.case==0).sum()); n_ctrl_distinct=int(DM[DM.case==0].uid.nunique())
log(f"FE dataset rows {n_rows}, control rows {n_ctrl_rows}, distinct control points {n_ctrl_distinct}")
pid=pd.factorize(DM['uid'])[0]

FE0="case ~ C(event) + slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
m0=smf.logit(FE0,data=DM).fit(disp=0,maxiter=300)
m0c=smf.logit(FE0,data=DM).fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':pid})
ev_grp=pd.factorize(DM['event'])[0]
def lin(m,term='bamboo:slope'):
    b,s=m.params[term],m.bse[term]; return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5))
res=dict(ordinary=lin(m0),control_point_clustered=lin(m0c),n_rows=n_rows,n_control_rows=n_ctrl_rows,n_distinct_control_points=n_ctrl_distinct,n_case_rows=int((DM.case==1).sum()))
log(f"FE (no rain) linear: ordinary {res['ordinary']} point-clustered {res['control_point_clustered']}")
F=json.load(open(ROOT+'/verify/verified_facts_s11.json'))
F['facts']['S11.fe_linear_point_clustered']=dict(value=res,definition='event fixed-effects linear interaction (no rainfall covariate) on the 5 km affected-area design, ordinary and control-point-clustered covariance (each control point is one cluster across the events it enters)',source='rebuilt table',cls='A',note='')
json.dump(F,open(ROOT+'/verify/verified_facts_s11.json','w'),ensure_ascii=False,indent=1)
log('done')
