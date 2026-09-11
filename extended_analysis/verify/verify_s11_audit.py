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
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
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
# ---------------------------------------------------------------- (1) pooled spline
A=CASE[CASE.ft.isin(K1)][COVS+['ft','event']].copy(); A['case']=1
B=CTRL[CTRL.ft.isin(K1)][COVS+['ft']].copy(); B['case']=0; B['event']='__controls__'
D=pd.concat([A,B],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int)
form=f"case ~ {SPL_FIX}*bamboo + north+east+planc+profc+elev"
m_ord=smf.logit(form,data=D).fit(disp=0)
d_=contrast(m_ord,D)
ord_,lo_ord,se_ord=band_from(d_,m_ord.params.values,m_ord.cov_params().values)
grp=pd.factorize(D['event'])[0]
m_cl=smf.logit(form,data=D).fit(disp=0,cov_type='cluster',cov_kwds={'groups':grp})
cl_,lo_cl,se_cl=band_from(d_,m_cl.params.values,m_cl.cov_params().values)
log(f"pooled spline: ordinary band {ord_['supported_band']} crossover {ord_['crossover_deg']}; event-cluster band {cl_['supported_band']} (clusters {len(np.unique(grp))})")
fact('S11.pooled_spline_bands',dict(ordinary=ord_,event_cluster_robust=cl_,n_clusters=int(len(np.unique(grp))),knots=dict(interior=KNOT,lower=LB,upper=UB),
     note='same fitted curve; only the covariance differs. Bands are pointwise 95% limits, not simultaneous'),
     "delta-method 95% band of the bamboo-versus-broadleaf spline curve under ordinary and event-cluster-robust covariance (cases clustered by event, all controls one cluster)","rebuilt table","A")
# ---------------------------------------------------------------- (2) FE + rain design with the same knots
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
FEL="case ~ C(event) + slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope+lmax24"
mfe=smf.logit(FEL,data=DM).fit(disp=0,maxiter=300)
mfe_cl=smf.logit(FEL,data=DM).fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':pid})
def lin(m,term='bamboo:slope'):
    b,s=m.params[term],m.bse[term]; return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5))
fe_lin=dict(ordinary=lin(mfe),control_point_clustered=lin(mfe_cl),n_rows=n_rows,n_control_rows=n_ctrl_rows,n_distinct_control_points=n_ctrl_distinct,n_clusters=int(len(np.unique(pid))))
log(f"FE+rain linear: ordinary {fe_lin['ordinary']}  point-clustered {fe_lin['control_point_clustered']}")
FES=f"case ~ C(event) + {SPL_FIX}*bamboo + north+east+planc+profc+elev+lmax24"
mfs=smf.logit(FES,data=DM).fit(disp=0,maxiter=300)
d2=contrast(mfs,DM)
fes_ord,lo2,se2=band_from(d2,mfs.params.values,mfs.cov_params().values)
mfs_cl=smf.logit(FES,data=DM).fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':pid})
fes_cl,_,_=band_from(d2,mfs_cl.params.values,mfs_cl.cov_params().values)
log(f"FE+rain spline: ordinary band {fes_ord['supported_band']} crossover {fes_ord['crossover_deg']}; point-clustered band {fes_cl['supported_band']}")
fact('S11.fe_rain_spline_bands',dict(linear_interaction=fe_lin,spline_ordinary=fes_ord,spline_control_point_clustered=fes_cl,
     note='event fixed effects + log(1+max 24-h IMERG rainfall); same interior knot and boundaries as the pooled spline; controls enter once per event they fall in'),
     "spline curve and linear interaction in the event fixed-effects + rainfall design, ordinary and control-point-clustered covariance","rebuilt table","B")
# ---------------------------------------------------------------- (3) cluster bootstrap of the whole curve
rng=np.random.default_rng(20260902)
ca_idx=np.where(D.case.values==1)[0]; co_idx=np.where(D.case.values==0)[0]
cs_ev=D.loc[ca_idx,'event'].values; events=np.unique(cs_ev); by_ev={e:ca_idx[cs_ev==e] for e in events}
curves=[]; cx=[]; n_try=0; n_fail_fit=0; n_few=0
for _ in range(600):
    n_try+=1
    pick=rng.choice(events,len(events),replace=True); ci=np.concatenate([by_ev[e] for e in pick]); oi=rng.choice(co_idx,len(co_idx),replace=True)
    d=D.iloc[np.concatenate([ci,oi])]
    if int(((d.case==1)&(d.bamboo==1)).sum())<20: n_few+=1; continue
    try:
        sp=smf.logit(form,data=d).fit(disp=0); dd=contrast(sp,d); lo=dd@sp.params.values; curves.append(lo)
        s=np.sign(lo); v=None
        for i in range(len(GRID)-1):
            if s[i]<0 and s[i+1]>=0: v=float(GRID[i]-lo[i]*(GRID[i+1]-GRID[i])/(lo[i+1]-lo[i])); break
        if v is not None: cx.append(v)
    except Exception: n_fail_fit+=1
C=np.exp(np.array(curves)); q025=np.percentile(C,2.5,axis=0); q975=np.percentile(C,97.5,axis=0); q50=np.percentile(C,50,axis=0)
p=GRID[q975<1]; bband=[float(p.min()),float(p.max())] if len(p) else None; contiguous=bool(len(p)>0 and np.allclose(np.diff(p),0.1,atol=1e-6))
cx=np.array(cx)
boot=dict(draws=600,curves_fitted=int(len(curves)),draws_skipped_fewer_than_20_bamboo_cases=n_few,fit_failures=n_fail_fit,crossings_found=int(len(cx)),
          band_q975_below_1=bband,contiguous=contiguous,
          crossover=dict(median=round(float(np.median(cx)),2),ci95=[round(float(np.percentile(cx,2.5)),2),round(float(np.percentile(cx,97.5)),2)],ci50=[round(float(np.percentile(cx,25)),2),round(float(np.percentile(cx,75)),2)]),
          at={str(g):dict(median=round(float(q50[k]),3),q025=round(float(q025[k]),3),q975=round(float(q975[k]),3)) for k,g in enumerate(GRID) if int(g)==g and int(g) in (10,20,25,30,35,40,45,50)})
log(f"bootstrap: curves {len(curves)}, crossings {len(cx)}, q97.5<1 band {bband}, crossover median {boot['crossover']['median']} 95% {boot['crossover']['ci95']}")
fact('S11.curve_cluster_bootstrap',boot,"cluster bootstrap of the whole spline curve (events resampled with all their cases, controls resampled), pointwise percentiles on the frozen grid","rebuilt table","C",
     note="the range where the 97.5th percentile of the bootstrap curve is below 1 is a pointwise statement, not a simultaneous band")
np.save(ROOT+'/verify/s11_boot_curves.npy',C)
import os as _o
_p=ROOT+'/verify/verified_facts_s11.json'
_F=json.load(open(_p)) if _o.path.exists(_p) else dict(facts={})   # merge: keep the facts appended by the later stage-11 scripts
_F['facts'].update(OUT)
json.dump(_F,open(_p,'w'),ensure_ascii=False,indent=1)
log('done')
