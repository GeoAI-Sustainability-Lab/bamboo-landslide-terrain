# -*- coding: utf-8 -*-
"""Stage 11b - the registered cluster bootstrap (S6.crossover_bootstrap: seed 20260902, bs(slope, df=4), cases_xy/controls_xy)
re-run while storing the whole curve of every draw, so that the crossing statistics already in the manuscript and the new
pointwise percentile band come from the same 600 draws. Also characterises the draws without a crossing inside 5-60 deg.
Appends S11.registered_bootstrap_curves to verified_facts_s11.json; saves s11b_boot_curves.npy"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, time, numpy as np, pandas as pd, statsmodels.formula.api as smf
from patsy import dmatrix
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]; COVS=['slope','north','east','planc','profc','elev']
SPL="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
CA=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CO=pd.read_csv(ROOT+'/verify/controls_xy.csv')
CTRL=CO
def build(a,b,bam,keep):
    A=a[a.ft.isin(keep)][COVS+['ft']].copy(); A['case']=1
    B=b[b.ft.isin(keep)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['bamboo']=D.ft.isin(bam).astype(int); D['slope2']=D.slope**2
    return D
D1=build(CA,CO,[BAMBOO],K1)
cs_ev=CA[CA.ft.isin(K1)].event.values
rng=np.random.default_rng(20260902)
ca_idx=np.where(D1.case.values==1)[0]; co_idx=np.where(D1.case.values==0)[0]
events=np.unique(cs_ev); by_ev={e:ca_idx[cs_ev==e] for e in events}
cx=[]; curves=[]; nocross=[]; n_few=0; n_fail=0
for k in range(600):
    pick=rng.choice(events,len(events),replace=True)
    ci=np.concatenate([by_ev[e] for e in pick])
    oi=rng.choice(co_idx,len(co_idx),replace=True)
    d=D1.iloc[np.concatenate([ci,oi])]
    if int(((d.case==1)&(d.bamboo==1)).sum())<20: n_few+=1; continue
    try:
        sp=smf.logit(SPL,data=d).fit(disp=0); di=sp.model.data.orig_exog.design_info
        med={c:float(np.median(d[c])) for c in ['north','east','planc','profc','elev']}
        row=lambda b: pd.DataFrame({'slope':GRID,'bamboo':b,**med})
        lo=(np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0))))@sp.params.values
        curves.append(lo)
        s=np.sign(lo); v=None
        for i in range(len(GRID)-1):
            if s[i]<0 and s[i+1]>=0: v=float(GRID[i]-lo[i]*(GRID[i+1]-GRID[i])/(lo[i+1]-lo[i])); break
        if v is not None: cx.append(v)
        else: nocross.append(dict(draw=k,n_bamboo_cases=int(((d.case==1)&(d.bamboo==1)).sum()),OR_at_5=float(np.exp(lo[0])),OR_at_40=float(np.exp(lo[np.argmin(np.abs(GRID-40))])),OR_at_60=float(np.exp(lo[-1])),max_OR=float(np.exp(lo.max()))))
    except Exception as e: n_fail+=1
cx=np.array(cx); C=np.exp(np.array(curves))
q025=np.percentile(C,2.5,axis=0); q975=np.percentile(C,97.5,axis=0); q50=np.percentile(C,50,axis=0)
p=GRID[q975<1]; bband=[float(p.min()),float(p.max())] if len(p) else None; contiguous=bool(len(p)>0 and np.allclose(np.diff(p),0.1,atol=1e-6))
bp95=float(CTRL[CTRL.ft==BAMBOO].slope.quantile(0.95))
nc=pd.DataFrame(nocross)
below_all=int((nc.OR_at_60<1).sum()) if len(nc) else 0; above_all=int((nc.OR_at_5>=1).sum()) if len(nc) else 0
res=dict(seed=20260902,draws=600,curves_fitted=int(len(curves)),draws_skipped_fewer_than_20_bamboo_cases=n_few,fit_failures=n_fail,
         crossings_found=int(len(cx)),no_crossing=int(len(nocross)),
         no_crossing_detail=dict(curve_still_below_1_at_60=below_all,curve_above_1_at_5=above_all,rows=nocross),
         crossover=dict(median=round(float(np.median(cx)),2),ci95=[round(float(np.percentile(cx,2.5)),2),round(float(np.percentile(cx,97.5)),2)],
                        ci50=[round(float(np.percentile(cx,25)),2),round(float(np.percentile(cx,75)),2)],share_below_bamboo_p95=round(float(100*(cx<bp95).mean()),1)),
         band_q975_below_1=bband,contiguous=contiguous,
         at={str(g):dict(median=round(float(q50[k]),3),q025=round(float(q025[k]),3),q975=round(float(q975[k]),3)) for k,g in enumerate(GRID) if int(g)==g and int(g) in (10,15,20,25,30,35,40,45,50)})
log(f"registered bootstrap: crossings {len(cx)}/600, median {res['crossover']['median']} 95% {res['crossover']['ci95']} 50% {res['crossover']['ci50']}; no crossing {len(nocross)} (below 1 at 60: {below_all}, above 1 at 5: {above_all}); band q97.5<1 {bband} contiguous {contiguous}")
np.save(ROOT+'/verify/s11b_boot_curves.npy',C)
F=json.load(open(ROOT+'/verify/verified_facts_s11.json'))
F['facts']['S11.registered_bootstrap_curves']=dict(value=res,
  definition="the registered crossing bootstrap (S6.crossover_bootstrap, seed 20260902, bs(slope, df=4) refitted in each draw) re-run storing every curve; pointwise 2.5/97.5 percentiles on the frozen grid and the draws without a crossing inside 5-60 deg",
  source="rebuilt table",cls="C",note="pointwise band, not a simultaneous band; the 16 draws without a crossing are characterised in no_crossing_detail")
json.dump(F,open(ROOT+'/verify/verified_facts_s11.json','w'),ensure_ascii=False,indent=1)
log('done')
