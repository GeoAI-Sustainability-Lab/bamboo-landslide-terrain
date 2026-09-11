# -*- coding: utf-8 -*-
"""Regional (latitude-band) subsets and bamboo growth-form subsets of the headline
bamboo-by-slope interaction, original case-control design (pure 竹林 vs 闊葉樹林型,
original controls).  Deterministic (class B).  Output: regional_form_check.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import pandas as pd, numpy as np, json, math
import statsmodels.formula.api as smf
from scipy.stats import norm, chi2
COVS=['slope','north','east','planc','profc','elev']
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
PF=pd.read_csv(ROOT+'/verify/bamboo_points_form.csv')   # growth form of bamboo-containing points
form_case=dict(zip(PF[PF.kind=='case'].key,PF[PF.kind=='case'].form))
form_ctrl=dict(zip(PF[PF.kind=='ctrl'].key,PF[PF.kind=='ctrl'].form))
# region by TWD97 northing (EPSG:3826): north >= 2,720,000 m (~24.6 N); south < 2,600,000 m (~23.5 N)
def region(y): return np.where(y>=2_720_000,'north',np.where(y<2_600_000,'south','central'))
CASE['region']=region(CASE.y.values); CTRL['region']=region(CTRL.y.values)
CASE['form']=CASE.lid.map(form_case); CTRL['form']=CTRL.bg_lid.map(form_ctrl)
def build(cs,ct):
    a=cs[cs.ft.isin(['竹林','闊葉樹林型'])][COVS+['ft','region','form']].copy(); a['case']=1
    b=ct[ct.ft.isin(['竹林','闊葉樹林型'])][COVS+['ft','region','form']].copy(); b['case']=0
    D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft=='竹林').astype(int); D['slope2']=D.slope**2
    return D
def fit(D):
    m=smf.logit(FORM,data=D).fit(disp=0,maxiter=200)
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']; b0=m.params['bamboo']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=round(float(2*(1-norm.cdf(abs(b/s)))),5),main_OR=round(math.exp(b0),3),
                linear_crossover_deg=(round(float(-b0/b),1) if b>0 else None),
                n=int(len(D)),n_case=int(D.case.sum()),n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),
                n_bamboo_ctrl=int(((D.case==0)&(D.bamboo==1)).sum()),
                bamboo_case_median_slope=round(float(D[(D.case==1)&(D.bamboo==1)].slope.median()),1),
                bamboo_case_slope_p90=round(float(D[(D.case==1)&(D.bamboo==1)].slope.quantile(.9)),1)),m
D=build(CASE,CTRL)
out={'definition':{'region':'TWD97 (EPSG:3826) northing bands: north y>=2,720,000 m (~24.6N); central 2,600,000<=y<2,720,000; south y<2,600,000 m (~23.5N)',
                   'design':'original case-control design, pure 竹林 vs 闊葉樹林型, original controls, model = published E1 form'}}
out['all'],_=fit(D)
out['by_region']={}
for r in ['north','central','south']:
    out['by_region'][r],_=fit(D[D.region==r])
# pooled test of regional heterogeneity of the interaction: LR test of region*bamboo*slope terms
m0=smf.logit(FORM+"+C(region)+C(region):slope",data=D).fit(disp=0,maxiter=300)
m1=smf.logit(FORM+"+C(region)+C(region):slope+C(region):bamboo+C(region):bamboo:slope",data=D).fit(disp=0,maxiter=300)
lr=2*(m1.llf-m0.llf); df=int(m1.df_model-m0.df_model)
out['region_heterogeneity_LR']=dict(LR=round(float(lr),3),df=df,p=round(float(1-chi2.cdf(lr,df)),4))
# bamboo cases / controls by region x form
bc=D[(D.bamboo==1)]
out['bamboo_points_by_region_form']={f"{k[0]}|{'case' if k[1]==1 else 'ctrl'}":{str(kk):int(vv) for kk,vv in v.form.value_counts(dropna=False).items()} for k,v in bc.groupby(['region','case'])}
# growth form with ORIGINAL controls (complements bamboo_form.json which used the new pool)
out['by_form_original_controls']={}
for f in ['單桿狀竹(running)','叢生狀竹(clumping)']:
    Df=D[(D.bamboo==0)|(D.form==f)]
    out['by_form_original_controls'][f],_=fit(Df)
# elevation / slope by form (cases and original controls)
out['elev_slope_by_form']={}
for kind,cs in [('case',1),('ctrl',0)]:
    s=D[(D.bamboo==1)&(D.case==cs)]
    out['elev_slope_by_form'][kind]={f:dict(n=int(len(g)),elev_median=round(float(g.elev.median()),0),elev_p10=round(float(g.elev.quantile(.1)),0),
                                         elev_p90=round(float(g.elev.quantile(.9)),0),slope_median=round(float(g.slope.median()),1))
                                    for f,g in s.groupby('form')}
# bamboo forest cases by region (all bamboo-containing types, for context)
out['n_cases_by_region_all_types']={r:int(v) for r,v in CASE.region.value_counts().items()}
json.dump(out,open(ROOT+'/verify/regional_form_check.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
