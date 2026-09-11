# -*- coding: utf-8 -*-
"""Stage 7: independent cross-checks. Re-derive the headline quantities with different
implementations to catch bugs in the primary path."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import norm
from scipy.interpolate import BSpline
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
SRC=json.load(open(ROOT+'/verify/verified_facts.json')); OUT=SRC['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,v,d,s,c,n=""):
    OUT[fid]=dict(value=v,definition=d,source=s,cls=c,note=n); return v
BAMBOO='竹林'; BROAD='闊葉樹林型'
d=pd.read_csv(ROOT+'/../data/step1_dataset.csv').dropna()
d=d[d.ft.isin([BROAD,BAMBOO])].copy()
d['bamboo']=(d.ft==BAMBOO).astype(int); d['slope2']=d.slope**2

# ---- 1. hand-written IRLS vs statsmodels
log("cross-check 1: hand-written Newton-Raphson logistic regression")
cols=['slope','slope2','north','east','planc','profc','elev','bamboo']
X=np.column_stack([np.ones(len(d))]+[d[c].values for c in cols]+[d.bamboo.values*d.slope.values])
names=['const']+cols+['bamboo:slope']
y=d.case.values.astype(float)
beta=np.zeros(X.shape[1])
for it in range(200):
    eta=X@beta; p=1/(1+np.exp(-eta)); W=p*(1-p)
    XtWX=X.T@(X*W[:,None]); g=X.T@(y-p)
    step=np.linalg.solve(XtWX,g); beta=beta+step
    if np.max(np.abs(step))<1e-12: break
cov=np.linalg.inv(XtWX); se=np.sqrt(np.diag(cov))
i=names.index('bamboo:slope')
m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope",data=d).fit(disp=0)
db=abs(beta[i]-m.params['bamboo:slope']); ds=abs(se[i]-m.bse['bamboo:slope'])
fact('S7.irls_crosscheck',dict(beta_manual=float(beta[i]),beta_statsmodels=float(m.params['bamboo:slope']),
     abs_diff_beta=float(db),se_manual=float(se[i]),se_statsmodels=float(m.bse['bamboo:slope']),
     abs_diff_se=float(ds),OR_manual=round(float(math.exp(beta[i])),6),iterations=it+1),
 "the headline coefficient re-estimated with a hand-written Newton-Raphson solver and an "
 "analytically inverted information matrix, independent of statsmodels",
 "step1_dataset.csv","A")
log(f"   beta diff {db:.3e}, se diff {ds:.3e}, OR {math.exp(beta[i]):.6f}")

# ---- primary spline path (patsy formula), reused by cross-checks 2, 3 and 5
SPL="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
sp=smf.logit(SPL,data=d).fit(disp=0); di=sp.model.data.orig_exog.design_info
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
med={c:float(np.median(d[c])) for c in ['north','east','planc','profc','elev']}
row=lambda b: pd.DataFrame({'slope':GRID,'bamboo':b,**med})
D_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0)))
lo_p=D_@sp.params.values; V=sp.cov_params().values
se_p=np.sqrt(np.einsum('ij,jk,ik->i',D_,V,D_))

# ---- 2. the whole spline path rebuilt without patsy
log("cross-check 2: spline path rebuilt without patsy (scipy BSpline.design_matrix, hand-built design matrix, hand-built contrast)")
s=d.slope.values
knots=np.r_[[s.min()]*4,[np.median(s)],[s.max()]*4]          # cubic, one interior knot at the median: 5 basis functions
Bfull=BSpline.design_matrix(s,knots,3).toarray()
Bp=np.asarray(dmatrix("bs(slope, df=4)-1",d))                 # patsy drops the first basis function when the model has an intercept
mx=float(np.max(np.abs(Bfull[:,1:]-Bp)))
B=Bfull[:,1:]; cov=d[['north','east','planc','profc','elev']].values; bam=d.bamboo.values[:,None]
Xs=np.column_stack([np.ones(len(d)),B,bam,bam*B,cov])           # same terms as the primary formula bs(slope,df=4)*bamboo + covariates
ms=sm.Logit(d.case.values,Xs).fit(disp=0,maxiter=200,method='newton')
Bg=BSpline.design_matrix(GRID,knots,3).toarray()[:,1:]
Cg=np.column_stack([np.zeros((len(GRID),1)),np.zeros_like(Bg),np.ones((len(GRID),1)),Bg,np.zeros((len(GRID),5))])
lo_m=Cg@ms.params; se_m=np.sqrt(np.einsum('ij,jk,ik->i',Cg,ms.cov_params(),Cg))
def _crossover(grid,logor):
    sg=np.sign(logor)
    for i in range(len(grid)-1):
        if sg[i]<0 and sg[i+1]>=0:
            x0,x1=grid[i],grid[i+1]; y0,y1=logor[i],logor[i+1]; return float(x0-y0*(x1-x0)/(y1-y0))
    return None
def _band(grid,hi):
    p=grid[hi<1]; return [float(p.min()),float(p.max())] if len(p) else None
cx_m=_crossover(GRID,lo_m); band_m=_band(GRID,np.exp(lo_m+1.96*se_m))
cx_p=_crossover(GRID,lo_p); band_p=_band(GRID,np.exp(lo_p+1.96*se_p))
fact('S7.bspline_basis_crosscheck',dict(max_abs_diff=mx,crossover_from_manual_basis=round(cx_m,4),primary_path_crossover=round(cx_p,4),
     protection_band_manual=band_m,primary_path_band=band_p),
 "the whole spline path rebuilt without patsy: scipy BSpline.design_matrix for the basis, statsmodels Logit on a hand-built design matrix, contrast formed by hand",
 "step1_dataset.csv","A")
log(f"   basis max abs diff {mx:.3e}; crossover manual {cx_m:.4f} vs primary {cx_p:.4f}; band manual {band_m} vs primary {band_p}")

# ---- 3. delta-method band vs parametric bootstrap of the parameter vector
log("cross-check 3: delta-method confidence band vs parametric simulation")
lo=lo_p; se_delta=se_p
rng=np.random.default_rng(99)
draws=rng.multivariate_normal(sp.params.values,V,size=20000)
lo_sim=draws@D_.T
se_sim=lo_sim.std(axis=0)
band_delta=np.exp(lo+1.96*se_delta)<1
band_sim=np.exp(np.percentile(lo_sim,97.5,axis=0))<1
fact('S7.delta_vs_simulation',dict(max_abs_se_diff=float(np.max(np.abs(se_delta-se_sim))),
     protection_band_delta=[float(GRID[band_delta].min()),float(GRID[band_delta].max())],
     protection_band_simulated=[float(GRID[band_sim].min()),float(GRID[band_sim].max())],
     n_sim=20000,seed=99),
 "the delta-method standard error of the log odds-ratio curve compared with a 20,000-draw "
 "multivariate-normal simulation of the parameter vector",
 "step1_dataset.csv","A")
log(f"   se max diff {np.max(np.abs(se_delta-se_sim)):.3e}; band delta {GRID[band_delta].min()}-{GRID[band_delta].max()} vs sim {GRID[band_sim].min()}-{GRID[band_sim].max()}")

# ---- 4. conditional-logit implementation: Breslow-tied stratified Cox is not exact; clogit_exact agrees with statsmodels ConditionalLogit
log("cross-check 4: conditional logistic regression implementation (Breslow Cox vs exact; clogit_exact vs statsmodels)")
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.duration.hazard_regression import PHReg
rng=np.random.default_rng(3)
n=900; g=rng.integers(0,60,n)
x1=rng.normal(size=n); x2=rng.normal(size=n)
eta=0.4*x1-0.3*x2+rng.normal(0,0.5,size=60)[g]
yy=(rng.random(n)<1/(1+np.exp(-eta))).astype(int)
Z=np.column_stack([x1,x2])
cl=ConditionalLogit(yy,Z,groups=g).fit(disp=0)
ph=PHReg(np.where(yy==1,1.0,2.0),Z,status=yy,strata=g,ties='breslow').fit()
import sys as _sys; _sys.path.insert(0,ROOT+'/verify'); import clogit_exact
ex=clogit_exact.fit(yy,Z,g)
dpar=float(np.max(np.abs(ex['params']-np.array(cl.params)))); dse=float(np.max(np.abs(ex['bse']-np.array(cl.bse))))
dbres=float(np.max(np.abs(np.array(cl.params)-np.array(ph.params))))
fact('S7.clogit_method',dict(
     breslow_cox_vs_exact=f"Breslow tie handling in a stratified Cox model is NOT equal to conditional logistic regression when strata contain many cases; on a 60-stratum simulation the coefficients differed by {dbres:.1e}",
     exact_implementation_vs_statsmodels=dict(max_param_diff=float(f"{dpar:.4g}"),max_se_diff=float(f"{dse:.4g}")),
     decision="all conditional-logit results are reported from the exact implementation; the earlier Breslow-tied numbers are superseded"),
 "validation of the conditional logistic regression implementation",
 "simulation + statsmodels ConditionalLogit","A")
log(f"   Breslow Cox vs conditional logit max coef diff {dbres:.3e}; clogit_exact vs statsmodels: params {dpar:.3e}, se {dse:.3e}")

# ---- 5. crossover invariance to the covariate values used for prediction
log("cross-check 5: crossover invariance to prediction covariates")
def cx_at(vals):
    r=lambda b: pd.DataFrame({'slope':GRID,'bamboo':b,**vals})
    dd=np.asarray(dmatrix(di,r(1)))-np.asarray(dmatrix(di,r(0)))
    l=dd@sp.params.values; sg=np.sign(l)
    for i in range(len(GRID)-1):
        if sg[i]<0 and sg[i+1]>=0: return float(GRID[i]-l[i]*(GRID[i+1]-GRID[i])/(l[i+1]-l[i]))
    return None
q1={c:float(d[c].quantile(0.1)) for c in med}; q9={c:float(d[c].quantile(0.9)) for c in med}
fact('S7.crossover_invariance',dict(at_median=cx_at(med),at_p10=cx_at(q1),at_p90=cx_at(q9)),
 "the crossover recomputed with the non-slope covariates set to their 10th, 50th and 90th percentiles; "
 "they cancel in the bamboo contrast, so the value must not change",
 "step1_dataset.csv","A")
log(f"   crossover at p10/median/p90 = {cx_at(q1)}, {cx_at(med)}, {cx_at(q9)}")
json.dump({'facts':OUT,'log':SRC['log']+LOG},open(ROOT+'/verify/verified_facts.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("cross-checks written")
