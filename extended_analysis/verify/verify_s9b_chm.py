# -*- coding: utf-8 -*-
"""Stage 9b - canopy height (Meta/WRI CHM v2 mosaic, 2010-2019) checks re-derived from the packaged
per-point table combined_chm.csv (chm < 0 = no coverage). Output: verified_facts_s9b.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import pandas as pd, numpy as np, math, json, statsmodels.formula.api as smf
from scipy.stats import norm
chm=pd.read_csv(ROOT+'/../data/combined_chm.csv')[['lid','chm']]
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv').merge(chm,on='lid',how='left')
CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv').merge(chm.rename(columns={'lid':'bg_lid'}),on='bg_lid',how='left')
COVS=['slope','north','east','planc','profc','elev']; K1=['竹林','闊葉樹林型']
OUT={}
def fact(fid,value,definition,source,cls,note=""): OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note)
a=CASE[CASE.ft.isin(K1)].copy(); a['case']=1; b=CTRL[CTRL.ft.isin(K1)].copy(); b['case']=0
D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft=='竹林').astype(int); D['slope2']=D.slope**2
cov=D.chm>=0
def inter(d,extra=""):
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev"+extra+"+bamboo+bamboo:slope",data=d).fit(disp=0)
    bb,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(bb),4),ci=[round(math.exp(bb-1.96*s),4),round(math.exp(bb+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(bb/s)))),5),
                n=int(len(d)),n_case=int(d.case.sum()),n_bamboo_case=int(((d.case==1)&(d.bamboo==1)).sum()))
res=dict(coverage_share_all_points=round(float(cov.mean()),4),
         base_on_covered_subset=inter(D[cov]),plus_canopy_height=inter(D[cov],"+chm"),
         published_text=dict(base=1.032,plus_greenness_and_canopy=1.034,note='earlier section 3.1: canopy height leaves the interaction essentially unchanged, 1.034 to 1.034; figure script used 1.0318 vs 1.0337 (greenness + canopy)'))
# median canopy height by type (all six types), landscape (controls) and cases; pre-event canopy of cases vs population
ALLC=CTRL[CTRL.chm>=0]; ALLK=CASE[CASE.chm>=0]
res['median_canopy_m_controls']={k:float(v) for k,v in ALLC.groupby('ft').chm.median().round(1).items()}
res['median_canopy_m_cases']={k:float(v) for k,v in ALLK.groupby('ft').chm.median().round(1).items()}
res['mean_canopy_m_controls']={k:float(v) for k,v in ALLC.groupby('ft').chm.mean().round(1).items()}
res['mean_canopy_m_cases']={k:float(v) for k,v in ALLK.groupby('ft').chm.mean().round(1).items()}
res['manuscript_values']=dict(median_landscape={'竹林':8,'竹闊混淆林':10,'闊葉樹林型':13,'針葉樹林型':15},broadleaf_cases_vs_population=[9.4,12.7],bamboo_cases_vs_population=[8.2,8.6])
fact('S9.canopy_height_checks',res,"canopy-height robustness re-derived from combined_chm.csv (CHM v2 mosaic 2010-2019, 1.2 m, coverage limited)","combined_chm.csv","A",
     note="greenness (Sentinel-2) values of the earlier text cannot be re-derived until the GEE extraction is repeated")
json.dump(dict(facts=OUT),open(ROOT+'/verify/verified_facts_s9b.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
