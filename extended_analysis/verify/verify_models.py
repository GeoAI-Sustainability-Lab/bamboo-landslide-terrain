# -*- coding: utf-8 -*-
"""Stages 5-7: reproduce every published number, then recompute the revision analyses
under frozen definitions and fixed seeds. Depends on verify_all.py outputs."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import os, json, math, time, warnings, gc, pickle
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import norm, chi2
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
P=json.load(open(ROOT+'/verify/verified_facts_partial.json'))
OUT=P['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,value,definition,source,cls,note=""):
    OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note); return value

BAMBOO='竹林'; BROAD='闊葉樹林型'; MIXED='竹闊混淆林'
COVS=['slope','north','east','planc','profc','elev']
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
SPL ="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
S1=ROOT+'/../data/step1_dataset.csv'; RJ=ROOT+'/../expected_outputs/results.json'
E=json.load(open(RJ))
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')

def build(case_df,ctrl_df,bamboo_set,keep_set):
    a=case_df[case_df.ft.isin(keep_set)][COVS+['ft']].copy(); a['case']=1
    b=ctrl_df[ctrl_df.ft.isin(keep_set)][COVS+['ft']].copy(); b['case']=0
    D=pd.concat([a,b],ignore_index=True)
    D['bamboo']=D.ft.isin(bamboo_set).astype(int); D['slope2']=D.slope**2
    return D
def interaction(D,form=FORM,cov_type=None,groups=None):
    if cov_type: m=smf.logit(form,data=D).fit(disp=0,cov_type=cov_type,cov_kwds={'groups':groups})
    else: m=smf.logit(form,data=D).fit(disp=0)
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=float(2*(1-norm.cdf(abs(b/s)))),beta=float(b),se=float(s),
                n=int(len(D)),n_case=int(D.case.sum()),
                n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),
                n_bamboo_ctrl=int(((D.case==0)&(D.bamboo==1)).sum())),m
def spline_curve(D,grid=GRID):
    sp=smf.logit(SPL,data=D).fit(disp=0)
    di=sp.model.data.orig_exog.design_info
    med={c:float(np.median(D[c])) for c in ['north','east','planc','profc','elev']}
    row=lambda bam: pd.DataFrame({'slope':grid,'bamboo':bam,**med})
    d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0)))
    lo=d_@sp.params.values
    V=sp.cov_params().values; se=np.sqrt(np.einsum('ij,jk,ik->i',d_,V,d_))
    return grid,np.exp(lo),np.exp(lo-1.96*se),np.exp(lo+1.96*se),lo,se,sp
def crossover(grid,logor):
    s=np.sign(logor)
    for i in range(len(grid)-1):
        if s[i]<0 and s[i+1]>=0:
            x0,x1=grid[i],grid[i+1]; y0,y1=logor[i],logor[i+1]
            return float(x0-y0*(x1-x0)/(y1-y0))
    return None
def band(grid,hi95,lo95):
    prot=grid[hi95<1]; worse=grid[lo95>1]
    def contig(a):
        if len(a)==0: return None
        return [float(a.min()),float(a.max()),bool(np.allclose(np.diff(a),0.1,atol=1e-6))]
    return contig(prot),contig(worse)

# ================================================================= STAGE 5
log("STAGE 5  reproduction of the published numbers")
d1=pd.read_csv(S1).dropna()
Drepo=d1[d1.ft.isin([BROAD,BAMBOO])].copy()
Drepo['bamboo']=(Drepo.ft==BAMBOO).astype(int); Drepo['slope2']=Drepo.slope**2
r_repo,m_repo=interaction(Drepo)
Dmine=build(CASE,CTRL,[BAMBOO],[BROAD,BAMBOO])
r_mine,_=interaction(Dmine)
pub=E['E5_interaction_specs']['specs']['terrain']
fact('S5.headline_interaction',dict(published=dict(OR=pub['OR'],ci=pub['ci'],p=pub['p']),
     from_published_table=r_repo,from_rebuilt_table=r_mine,
     identical_tables=bool(len(Drepo)==len(Dmine) and
        np.allclose(np.sort(Drepo.slope.values),np.sort(Dmine.slope.values)))),
 "bamboo x slope interaction odds ratio per degree, published specification",
 "step1_dataset.csv (published) and the independently rebuilt table","A")
log(f"   published {pub['OR']} | from step1 {r_repo['OR']} | from rebuilt {r_mine['OR']}")

g,orc,l95,h95,lo,se,sp=spline_curve(Drepo,np.array(E['E4_spline']['grid'],float))
diff=np.max(np.abs(orc-np.array(E['E4_spline']['or'])))
fact('S5.spline_curve_reproduction',dict(max_abs_OR_diff=float(diff),
     grid=E['E4_spline']['grid'],reproduced_or=[round(float(v),4) for v in orc],
     published_or=E['E4_spline']['or']),
 "spline odds-ratio curve evaluated on the published grid (6-62 deg by 2), compared with results.json E4",
 "step1_dataset.csv","A")
g2,orc2,l2,h2,lo2,se2,_=spline_curve(Drepo,GRID)
cx=crossover(g2,lo2); pb,wb=band(g2,h2,l2)
fact('S5.crossover_published_design',dict(crossover_deg=round(cx,2) if cx else None,
     published_reported=E['E4_spline']['crossover_slope'],
     significant_protection_band=pb,significantly_worse_band=wb),
 "crossover = interpolated slope at which the fitted log odds ratio changes sign on the frozen grid "
 "5.0-60.0 deg step 0.1; protection band = slopes whose delta-method 95% upper bound is below 1",
 "step1_dataset.csv","A")
log(f"   spline max |OR| diff vs results.json = {diff:.2e}; crossover {cx:.2f} deg; protection band {pb}")

# E6 cluster-robust: needs event labels -> use the rebuilt table
Dev=build(CASE,CTRL,[BAMBOO],[BROAD,BAMBOO])
ev=np.concatenate([CASE[CASE.ft.isin([BROAD,BAMBOO])].event.values,
                   np.array(['CTRL']*int((CTRL.ft.isin([BROAD,BAMBOO])).sum()))])
r_cl,_=interaction(Dev,cov_type='cluster',groups=ev)
fact('S5.event_cluster_robust',dict(reproduced=r_cl,published=E['E6_gee_cluster']),
 "event-cluster-robust (sandwich) standard errors, controls in one cluster",
 "rebuilt table with event labels","B",
 "the published E6 used 40 clusters; the exact control-cluster convention is not recorded in the repo")
# E8 size sensitivity
rows=[]
for thr in (0.0,0.1,0.5,1.0):
    sub=CASE[(CASE.ft.isin([BROAD,BAMBOO]))&(CASE.area_ha>=thr)]
    D=build(sub,CTRL,[BAMBOO],[BROAD,BAMBOO]); rr,_=interaction(D)
    rows.append(dict(area_min_ha=thr,**{k:rr[k] for k in ('OR','ci','p','n_bamboo_case')}))
fact('S5.size_sensitivity',dict(reproduced=rows,published=E['E8_size_sensitivity']['rows']),
 "interaction refitted with cases restricted to landslides at or above an area threshold",
 "rebuilt table","A")
log(f"   size sensitivity reproduced: {[r['OR'] for r in rows]} vs published {[r['OR'] for r in E['E8_size_sensitivity']['rows']]}")

# full coefficient table
ct=pd.DataFrame({'beta':m_repo.params,'se':m_repo.bse,'z':m_repo.tvalues,'p':m_repo.pvalues,
                 'OR':np.exp(m_repo.params),'lo':np.exp(m_repo.params-1.96*m_repo.bse),
                 'hi':np.exp(m_repo.params+1.96*m_repo.bse)})
fact('S5.full_coefficients',json.loads(ct.round(6).to_json(orient='index')),
 "complete coefficient table of the published specification","step1_dataset.csv","A")
fact('S5.model_fit',dict(n=int(m_repo.nobs),llf=float(m_repo.llf),llnull=float(m_repo.llnull),
     pseudo_r2=float(m_repo.prsquared),aic=float(m_repo.aic)),
 "fit statistics of the published specification","step1_dataset.csv","A")
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
X=Drepo[['slope','slope2','north','east','planc','profc','elev','bamboo']].copy(); X['bam_slope']=X.bamboo*X.slope
Xc=sm.add_constant(X)
raw={c:round(float(vif(Xc.values,i)),2) for i,c in enumerate(Xc.columns) if c!='const'}
Xn=X.copy(); mu=float(Xn.slope.mean()); Xn['slope_c']=Xn.slope-mu; Xn['slope2_c']=Xn.slope_c**2
Xn['bam_slope_c']=Xn.bamboo*Xn.slope_c
Xc2=sm.add_constant(Xn[['slope_c','slope2_c','north','east','planc','profc','elev','bamboo','bam_slope_c']])
cen={c:round(float(vif(Xc2.values,i)),2) for i,c in enumerate(Xc2.columns) if c!='const'}
fact('S5.vif',dict(raw=raw,centred=cen,slope_mean_deg=round(mu,3)),
 "variance inflation factors, before and after centring slope","step1_dataset.csv","A")
# spline spec
di_probe=dmatrix('bs(slope, df=4)',Drepo,return_type='dataframe').design_info
knots=None
for f,info in di_probe.factor_infos.items():
    for k,v in (info.state.get('transforms') or {}).items():
        knots=[float(x) for x in getattr(v,'_all_knots',[])]
fact('S5.spline_spec',dict(basis='bs(slope, df=4)',degree=3,all_knots=knots,
     interior_knots=[k for k in knots if 0<k<max(knots)] if knots else None,
     evaluation='design_info of the fitted model reused; covariates cancel in the bamboo contrast'),
 "cubic B-spline specification actually used","step1_dataset.csv","A")
log(f"   VIF raw slope {raw['slope']} -> centred {cen['slope_c']}; spline interior knot {[k for k in knots if 0<k<max(knots)]}")
json.dump({'facts':OUT,'log':P['log']+LOG},open(ROOT+'/verify/verified_facts_s5.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 5 written")
