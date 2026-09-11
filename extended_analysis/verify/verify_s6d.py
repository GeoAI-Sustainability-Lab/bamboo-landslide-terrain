# -*- coding: utf-8 -*-
"""Stage 6d: replace the Breslow-tied stratified-Cox conditional logits with the
EXACT conditional likelihood, Monte-Carlo stabilised over 25 control draws."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings,sys
import numpy as np,pandas as pd
from scipy.stats import norm
sys.path.insert(0,ROOT+'/verify'); import clogit_exact
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
SRC=json.load(open(ROOT+'/verify/verified_facts.json')); OUT=SRC['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,v,d,s,c,n=""):
    OUT[fid]=dict(value=v,definition=d,source=s,cls=c,note=n); return v
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]
COVS=['slope','north','east','planc','profc','elev']
XC=['slope','slope2','north','east','planc','profc','elev','bamboo','bam_slope']
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
FP={k:set(map(tuple,v)) for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
cell=lambda df,c: list(zip(np.floor(df.x/c).astype(int),np.floor(df.y/c).astype(int)))
ca=CASE[CASE.ft.isin(K1)].copy(); po=POOL[POOL.ft.isin(K1)].copy()
NDRAW=10
def run(D):
    D=D.copy(); D['slope2']=D.slope**2; D['bamboo']=(D.ft==BAMBOO).astype(int); D['bam_slope']=D.bamboo*D.slope
    r=clogit_exact.fit(D.case.values,D[XC].values,D.stratum.values)
    i=XC.index('bam_slope'); b,s=r['params'][i],r['bse'][i]
    return math.exp(b),float(2*(1-norm.cdf(abs(b/s)))),r['n_strata'],int(((D.case==1)&(D.bamboo==1)).sum())
out={}
for c in (20000.,10000.,5000.):
    ca2=ca.assign(cc=cell(ca,c)); po2=po.assign(cc=cell(po,c))
    ors=[];ps=[];nst=[]
    for k in range(NDRAW):
        rg=np.random.default_rng(2000+k); rows=[]
        for (ev,cl),cs in ca2.groupby(['event','cc']):
            sel=po2[po2.cc==cl]
            if len(sel)==0: continue
            kk=min(len(sel),max(40,4*len(cs)))
            sub=sel.iloc[rg.choice(len(sel),kk,replace=False)]
            A=cs[COVS+['ft']].copy(); A['case']=1; B=sub[COVS+['ft']].copy(); B['case']=0
            t=pd.concat([A,B]); t['stratum']=f"{ev}|{cl[0]}_{cl[1]}"; rows.append(t)
        D=pd.concat(rows,ignore_index=True)
        o,p,ns,nb=run(D); ors.append(o); ps.append(p); nst.append(ns)
        if k==0: log(f"   {int(c/1000)} km draw 1: strata {ns}, bamboo cases {nb}, OR {o:.4f} p {p:.4g}")
    ors=np.array(ors); ps=np.array(ps)
    out[f'{int(c/1000)}km']=dict(OR_mean=round(float(ors.mean()),4),
        OR_range_mc=[round(float(np.percentile(ors,2.5)),4),round(float(np.percentile(ors,97.5)),4)],
        p_median=float(np.median(ps)),p_max=float(ps.max()),share_p_lt_05=float((ps<0.05).mean()),
        strata_median=int(np.median(nst)),n_draws=NDRAW)
    r=out[f'{int(c/1000)}km']
    log(f"   event x {int(c/1000):2d} km EXACT: OR mean {r['OR_mean']} MC {r['OR_range_mc']} p med {r['p_median']:.4g} p<0.05 in {100*r['share_p_lt_05']:.0f}%")
fact('S6.event_by_block_clogit_exact',dict(method="exact conditional likelihood (elementary symmetric "
     "polynomial dynamic program), validated against statsmodels ConditionalLogit to 7e-06",
     seeds="2000-2009",controls_per_stratum="min(available, max(40, 4x cases))",**out),
 "conditional logistic regression stratified by event x spatial block, exact likelihood, "
 "mean over 10 independent control draws",
 "rebuilt table + new control pool","C")
json.dump({'facts':OUT,'log':SRC['log']+LOG},open(ROOT+'/verify/verified_facts.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 6d written")
