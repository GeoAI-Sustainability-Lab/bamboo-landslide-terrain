# -*- coding: utf-8 -*-
"""Stage 11e - the two stage-11 facts that were first computed inline on 2026-09-07 and registered by hand, now as a script:
  S11.envelope_exact          bamboo-versus-broadleaf odds ratio within the observed range of bamboo (5th to 95th percentiles
                              of slope and elevation of the 663 bamboo controls, applied at their unrounded values)
  S11.latitude_band_LR_tests  likelihood-ratio tests of latitude-band heterogeneity (bands by TWD97 northing as in
                              regional_form_check.py): slope-interaction-only (2 df), main-effect-only (2 df), joint (4 df)
Deterministic (class B) from cases_annot.csv and controls_annot.csv. Appends to verified_facts_s11.json; the values are
asserted against the registered ones."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm, chi2
V=ROOT+'/verify/'
CASE=pd.read_csv(V+'cases_annot.csv'); CTRL=pd.read_csv(V+'controls_annot.csv')
COVS=['slope','north','east','planc','profc','elev']
def region(y): return np.where(y>=2_720_000,'north',np.where(y<2_600_000,'south','central'))
a=CASE[CASE.ft.isin(['竹林','闊葉樹林型'])][COVS+['ft','y']].copy(); a['case']=1
b=CTRL[CTRL.ft.isin(['竹林','闊葉樹林型'])][COVS+['ft','y']].copy(); b['case']=0
D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft=='竹林').astype(int); D['slope2']=D.slope**2; D['region']=region(D.y.values)
# ---- envelope: exact 5th-95th percentiles of the bamboo controls
bc=D[(D.case==0)&(D.bamboo==1)]
s_lo,s_hi=np.percentile(bc.slope,[5,95]); e_lo,e_hi=np.percentile(bc.elev,[5,95])
E=D[(D.slope>=s_lo)&(D.slope<=s_hi)&(D.elev>=e_lo)&(D.elev<=e_hi)]
m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+bamboo",data=E).fit(disp=0,maxiter=200)
beta,se=m.params['bamboo'],m.bse['bamboo']; p=float(2*(1-norm.cdf(abs(beta/se))))
env={'rule':'5th to 95th percentiles of the 663 bamboo controls, slope and elevation, applied exactly (no rounding of the bounds)',
     'slope_bounds_deg':[round(float(s_lo),3),round(float(s_hi),3)],'elev_bounds_m':[round(float(e_lo),1),round(float(e_hi),1)],
     'OR':round(math.exp(beta),3),'ci':[round(math.exp(beta-1.96*se),3),round(math.exp(beta+1.96*se),3)],'p':p,
     'n_case':int(E.case.sum()),'n_bamboo_case':int(((E.case==1)&(E.bamboo==1)).sum()),'n_ctrl':int((E.case==0).sum()),'n_bamboo_ctrl':int(((E.case==0)&(E.bamboo==1)).sum()),
     'model':'Eq. (1) terrain set: slope, slope^2, northness, eastness, plan and profile curvature, elevation + bamboo indicator'}
print('envelope',json.dumps(env))
# ---- latitude-band LR tests (model forms as in regional_form_check.py)
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope+C(region)+C(region):slope"
m0=smf.logit(FORM,data=D).fit(disp=0,maxiter=300)
m1=smf.logit(FORM+"+C(region):bamboo",data=D).fit(disp=0,maxiter=300)
m2=smf.logit(FORM+"+C(region):bamboo+C(region):bamboo:slope",data=D).fit(disp=0,maxiter=300)
def lr(ma,mb,df): L=2*(mb.llf-ma.llf); return {'LR':round(float(L),3),'df':df,'p':float(1-chi2.cdf(L,df))}
lrt={'slope_only':dict(lr(m1,m2,2),definition='region x bamboo x slope terms added to a model that already lets the bamboo main effect differ by region'),
     'joint':dict(lr(m0,m2,4),definition='region x bamboo and region x bamboo x slope terms together'),
     'main_effect_only':lr(m0,m1,2),'n':int(len(D))}
lrt['slope_only']['p']=round(lrt['slope_only']['p'],4); lrt['joint']['p_text']='<0.001'
print('LR tests',json.dumps(lrt))
# ---- register (append) and assert against the registered values
F=json.load(open(V+'verified_facts_s11.json'))
old_env=F['facts'].get('S11.envelope_exact',{}).get('value'); old_lr=F['facts'].get('S11.latitude_band_LR_tests',{}).get('value')
if old_env:
    assert old_env['OR']==env['OR'] and old_env['ci']==env['ci'] and old_env['n_case']==env['n_case'] and old_env['n_bamboo_case']==env['n_bamboo_case'], (old_env,env)
    env['previous_value_with_rounded_lower_bound_8_0']=old_env.get('previous_value_with_rounded_lower_bound_8_0')
if old_lr:
    assert abs(old_lr['slope_only']['LR']-lrt['slope_only']['LR'])<0.002 and abs(old_lr['joint']['LR']-lrt['joint']['LR'])<0.002 and abs(old_lr['main_effect_only']['LR']-lrt['main_effect_only']['LR'])<0.002, (old_lr,lrt)
F['facts']['S11.envelope_exact']={'value':env,'definition':'terrain-adjusted bamboo-versus-broadleaf odds ratio within the observed slope and elevation range of bamboo','source':'cases_annot.csv + controls_annot.csv','cls':'B'}
F['facts']['S11.latitude_band_LR_tests']={'value':lrt,'definition':'likelihood-ratio tests of latitude-band heterogeneity of the bamboo main effect and its slope interaction','source':'cases_annot.csv + controls_annot.csv','cls':'B'}
json.dump(F,open(V+'verified_facts_s11.json','w'),ensure_ascii=False,indent=1)
print('registered; asserted against the previous values' if (old_env and old_lr) else 'registered')
