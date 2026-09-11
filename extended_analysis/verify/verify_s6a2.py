# -*- coding: utf-8 -*-
"""Stage 6a-bis: replace every single-draw estimate by a Monte-Carlo-stabilised one.
Any quantity that depends on a random control draw is reported as the mean over 25
independent draws, with the 2.5-97.5 percentile range of the point estimates."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings
import numpy as np,pandas as pd,geopandas as gpd
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from scipy.stats import norm
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
P=json.load(open(ROOT+'/verify/verified_facts_s6a.json')); OUT=P['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,v,d,s,c,n=""):
    OUT[fid]=dict(value=v,definition=d,source=s,cls=c,note=n); return v
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]
COVS=['slope','north','east','planc','profc','elev']; CELL=5000.
FE="case ~ C(event) + slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
NDRAW=25
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')
POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
FP={k:set(map(tuple,v)) for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
cell=lambda df,c=CELL: list(zip(np.floor(df.x/c).astype(int),np.floor(df.y/c).astype(int)))
ca=CASE[CASE.ft.isin(K1)].copy(); ca['cell']=cell(ca)
po=POOL[POOL.ft.isin(K1)].copy(); po['cell']=cell(po)
co=CTRL[CTRL.ft.isin(K1)].copy().reset_index(drop=True); co['cid']=np.arange(len(co)); co['cell']=cell(co)

def summarise(ors,ps,ns,label):
    ors=np.array(ors); ps=np.array(ps)
    return dict(OR_mean=round(float(ors.mean()),4),
                OR_range_mc=[round(float(np.percentile(ors,2.5)),4),round(float(np.percentile(ors,97.5)),4)],
                p_median=float(np.median(ps)),p_max=float(ps.max()),
                share_p_lt_05=float((ps<0.05).mean()),n_draws=len(ors),
                n_bamboo_ctrl_median=int(np.median(ns)))
def fit_or(D,form=FE):
    m=smf.logit(form,data=D).fit(disp=0)
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return math.exp(b),float(2*(1-norm.cdf(abs(b/s))))
def clogit_or(D,strat):
    D=D.copy(); D['bam_slope']=D.bamboo*D.slope
    X=['slope','slope2','north','east','planc','profc','elev','bamboo','bam_slope']
    m=PHReg(np.where(D.case==1,1.0,2.0),D[X].values,status=(D.case==1).astype(int).values,
            strata=D[strat].values,ties='breslow').fit()
    i=X.index('bam_slope'); b,s=m.params[i],m.bse[i]
    return math.exp(b),float(2*(1-norm.cdf(abs(b/s))))

log(f"6a2.1 new-pool matched models, {NDRAW} draws each")
out={}
for ratio in (5,10,20):
    ors=[];ps=[];ns=[]
    for k in range(NDRAW):
        rg=np.random.default_rng(1000+k); rows=[]
        for ev,cl in FP.items():
            cs=ca[ca.event==ev]
            if len(cs)==0: continue
            sel=po[po.cell.isin(cl)]
            if len(sel)==0: continue
            kk=min(len(sel),max(300,ratio*len(cs)))
            sub=sel.iloc[rg.choice(len(sel),kk,replace=False)]
            A=cs[COVS+['ft']].copy(); A['case']=1; B=sub[COVS+['ft']].copy(); B['case']=0
            t=pd.concat([A,B]); t['event']=ev; rows.append(t)
        D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
        o,p=fit_or(D); ors.append(o); ps.append(p); ns.append(int(((D.case==0)&(D.bamboo==1)).sum()))
    out[f'{ratio}to1']=summarise(ors,ps,ns,f'{ratio}:1')
    log(f"   {ratio:2d}:1  OR mean {out[f'{ratio}to1']['OR_mean']} MC range {out[f'{ratio}to1']['OR_range_mc']} p median {out[f'{ratio}to1']['p_median']:.4g} p<0.05 in {100*out[f'{ratio}to1']['share_p_lt_05']:.0f}%")
fact('S6.new_pool_models_mc',out,
 f"event fixed-effects interaction on the new control pool, mean over {NDRAW} independent control draws "
 "(seeds 1000-1024); MC range = 2.5-97.5 percentile of the point estimates across draws",
 "rebuilt table + new control pool","C")

log(f"6a2.2 event x spatial-block conditional logit, {NDRAW} draws each")
out2={}
for c in (20000.,10000.,5000.):
    ca2=ca.assign(cc=cell(ca,c)); po2=po.assign(cc=cell(po,c))
    ors=[];ps=[];ns=[];nst=[]
    for k in range(NDRAW):
        rg=np.random.default_rng(2000+k); rows=[]
        for (ev,cl),cs in ca2.groupby(['event','cc']):
            sel=po2[po2.cc==cl]
            if len(sel)==0: continue
            kk=min(len(sel),max(40,4*len(cs)))
            sub=sel.iloc[rg.choice(len(sel),kk,replace=False)]
            A=cs[COVS+['ft']].copy(); A['case']=1; B=sub[COVS+['ft']].copy(); B['case']=0
            t=pd.concat([A,B]); t['stratum']=f"{ev}|{cl[0]}_{cl[1]}"; rows.append(t)
        D=pd.concat(rows,ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
        g=D.groupby('stratum').case.agg(['sum','size']); D=D[D.stratum.isin(g[(g['sum']>0)&(g['sum']<g['size'])].index)]
        o,p=clogit_or(D,'stratum'); ors.append(o); ps.append(p)
        ns.append(int(((D.case==0)&(D.bamboo==1)).sum())); nst.append(int(D.stratum.nunique()))
    out2[f'{int(c/1000)}km']=dict(**summarise(ors,ps,ns,''),strata_median=int(np.median(nst)))
    r=out2[f'{int(c/1000)}km']
    log(f"   event x {int(c/1000):2d} km: {r['strata_median']} strata, OR mean {r['OR_mean']} MC range {r['OR_range_mc']} p median {r['p_median']:.4g} p<0.05 in {100*r['share_p_lt_05']:.0f}%")
fact('S6.event_by_block_clogit_mc',out2,
 f"conditional logistic regression stratified by event x spatial block, mean over {NDRAW} control draws (seeds 2000-2024)",
 "rebuilt table + new control pool","C")

log(f"6a2.3 each control used once, {NDRAW} draws")
rows=[]
for ev,cl in FP.items():
    cs=ca[ca.event==ev]; ct=co[co.cell.isin(cl)]
    if len(cs)==0 or len(ct)==0: continue
    A=cs[COVS+['ft']].copy(); A['case']=1; A['uid']=['C'+str(i) for i in cs.index]
    B=ct[COVS+['ft']].copy(); B['case']=0; B['uid']=['B'+str(i) for i in ct.cid]
    t=pd.concat([A,B]); t['event']=ev; rows.append(t)
DM=pd.concat(rows,ignore_index=True); DM['bamboo']=(DM.ft==BAMBOO).astype(int); DM['slope2']=DM.slope**2
DM.to_pickle(ROOT+'/verify/DM_event_matched.pkl')
ors=[];ps=[];ns=[]
ctl=DM[DM.case==0]
for k in range(NDRAW):
    rg=np.random.default_rng(3000+k)
    pick=ctl.groupby('uid').event.apply(lambda s: rg.choice(s.values))
    keep=set(zip(pick.index,pick.values))
    m=DM.apply(lambda r: True if r['case']==1 else (r['uid'],r['event']) in keep,axis=1)
    D=DM[m]; o,p=fit_or(D); ors.append(o); ps.append(p); ns.append(int(((D.case==0)&(D.bamboo==1)).sum()))
out3=summarise(ors,ps,ns,'')
fact('S6.each_control_once_mc',out3,
 f"event fixed-effects interaction with every control assigned to exactly one event, mean over {NDRAW} draws (seeds 3000-3024)",
 "rebuilt table","C")
log(f"   once: OR mean {out3['OR_mean']} MC range {out3['OR_range_mc']} p median {out3['p_median']:.4g}")
json.dump({'facts':OUT,'log':P['log']+LOG},open(ROOT+'/verify/verified_facts_s6a2.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 6a-bis written")
