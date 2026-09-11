# -*- coding: utf-8 -*-
"""Stage 10b - pre-event Sentinel-2 greenness re-derived on Earth Engine (gee/gee_s2_vi2.py, gee/gee_s2_ctrl.py).
Cases: cloud-free median NDVI / NDMI / NBR of the 80-to-5 days before each polygon's pre-event image date.
Controls, primary (like-for-like): the same window of the pre-event image date of the nearest case.
Controls, secondary (earlier definition): 2019-2024 cloud-free median (if extracted).
Island-wide interaction model with NDVI / NDMI added. Output: verified_facts_s10b.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm
OUT={}
def fact(fid,value,definition,source,cls,note=""): OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note)
K1=['竹林','闊葉樹林型']; BAMBOO='竹林'; BROAD='闊葉樹林型'
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
sc=pd.read_csv(ROOT+'/gee/s2_cases.csv'); sc['uid']=sc.uid.astype(str)
CASE['lid']=CASE.lid.astype(str)
def fit(d,extra=""):
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev"+extra+"+bamboo+bamboo:slope",data=d).fit(disp=0,maxiter=200)
    bb,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    r=dict(OR=round(math.exp(bb),4),ci=[round(math.exp(bb-1.96*s),4),round(math.exp(bb+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(bb/s)))),5),n=int(len(d)),n_case=int(d.case.sum()),n_bamboo_case=int(((d.case==1)&(d.bamboo==1)).sum()))
    for k in ['NDVI','NDMI','NBR']:
        if k in m.params: r[k+'_beta']=round(float(m.params[k]),3); r[k+'_p']=round(float(m.pvalues[k]),5)
    return r
def build(ctrl_file):
    sk=pd.read_csv(ctrl_file); sk['uid']=sk.uid.astype(str)
    a=CASE[CASE.ft.isin(K1)].merge(sc.rename(columns={'uid':'lid'}),on='lid',how='left'); a['case']=1
    b=CTRL[CTRL.ft.isin(K1)].copy(); b['bg_lid']=b.bg_lid.astype(str); b=b.merge(sk.rename(columns={'uid':'bg_lid'}),on='bg_lid',how='left'); b['case']=0
    for D in (a,b):
        for k in ['NDVI','NDMI','NBR']: D.loc[(D.n_obs.fillna(0)<1),k]=np.nan
    cov=dict(cases_with_value=int(a.NDVI.notna().sum()),cases_total=int(len(a)),controls_with_value=int(b.NDVI.notna().sum()),controls_total=int(len(b)),
             cases_with_ge3_obs=int((a.n_obs>=3).sum()),median_obs_cases=float(a.n_obs.median()),median_obs_controls=float(b.n_obs.median()))
    med={}
    for lab,D in [('cases',a),('controls',b)]:
        for ft,nm in [(BAMBOO,'bamboo'),(BROAD,'broadleaf')]:
            s=D[D.ft==ft]; med[f'{lab}_{nm}']={k:round(float(s[k].median()),3) for k in ['NDVI','NDMI','NBR']}
    D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    sub=D.dropna(subset=['NDVI','NDMI'])
    return dict(coverage=cov,medians=med,base_on_covered_subset=fit(sub),plus_ndvi=fit(sub,"+NDVI"),plus_ndmi=fit(sub,"+NDMI"),plus_ndvi_ndmi=fit(sub,"+NDVI+NDMI"),plus_nbr=fit(sub,"+NBR"))
res=dict(like_for_like=build(ROOT+'/gee/s2_controls.csv'))
asg=pd.read_csv(ROOT+'/gee/s2_controls_assigned_dates.csv')
res['like_for_like']['control_date_assignment']=dict(rule='pre-event image date of the nearest landslide case (planar distance, TWD97 TM2)',
    median_distance_m=round(float(asg.dist_nearest_case_m.median())),p90_distance_m=round(float(asg.dist_nearest_case_m.quantile(0.9))),n_dates=int(asg.before.nunique()))
if os.path.exists(ROOT+'/gee/s2_controls_multiyear.csv'):
    res['multiyear_controls']=build(ROOT+'/gee/s2_controls_multiyear.csv')
res['manuscript_values']=dict(base=1.032,plus_greenness=1.036,plus_moisture=1.038,greenness_cases=[0.75,0.73],greenness_controls=[0.79,0.80],
                              note='earlier section 3.1 values (controls: 2019-2024 median); the original per-point extraction was not preserved')
fact('S10.pre_event_greenness',res,"pre-event Sentinel-2 NDVI/NDMI/NBR (cases: 80-5 days before the pre-event image; controls: same window of the nearest case's pre-event image date, and 2019-2024 median as the earlier definition) added to the island-wide interaction model","gee/s2_cases.csv, gee/s2_controls.csv, gee/s2_controls_multiyear.csv","B",
     note="SCL classes 3, 8, 9, 10, 11 masked; median of all cloud-free observations; points with no observation set to missing")
json.dump(dict(facts=OUT),open(ROOT+'/verify/verified_facts_s10b.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
