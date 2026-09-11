# -*- coding: utf-8 -*-
"""Stage 13 - descriptive quantities quoted in the text whose only previous computation was inside figure code or the
core-analysis output file (expected_outputs/results.json). Recomputed here from the frozen analysis tables, compared with
the core-analysis values, and registered.
  S13.landslide_density_by_type   rainfall landslides per km2 of each forest type (case counts / type area from the map)
  S13.slope_distribution_by_type  median, 5th and 95th percentile slope of the controls of each forest type
  S13.odds_ratio_by_type          odds of being a landslide relative to broadleaf forest, unadjusted and terrain-adjusted
                                  (terrain set of Eq. 1: slope, slope^2, northness, eastness, plan and profile curvature,
                                  elevation)
  S13.bamboo_share_5deg_bands     share of bamboo among bamboo and broadleaf controls in 5-degree slope bands, with the
                                  number of bamboo controls per band (common-support description)
Deterministic (class B). Output: verified_facts_s13.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, numpy as np, pandas as pd, statsmodels.formula.api as smf
V=ROOT+'/verify/'
CASE=pd.read_csv(V+'cases_annot.csv'); CTRL=pd.read_csv(V+'controls_annot.csv')
F=json.load(open(V+'verified_facts.json'))['facts']; R=json.load(open(ROOT+'/../expected_outputs/results.json'))
AREA=F['S3.forest_map']['value']['area_ha_by_type']
SIX=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹林','竹闊混淆林','待成林地']
EN={'闊葉樹林型':'broadleaf','針葉樹林型':'conifer','針闊葉樹混淆':'conifer-broadleaf','竹林':'bamboo','竹闊混淆林':'bamboo-broadleaf','待成林地':'immature'}
out={}
# ---- densities
n=CASE.ft.value_counts(); tot_area_km2=sum(AREA[t] for t in SIX)/100.0; mean_d=len(CASE)/tot_area_km2
dens={EN[t]:dict(n_rain=int(n[t]),area_km2=round(AREA[t]/100.0,1),density_per_km2=round(n[t]/(AREA[t]/100.0),4),ratio_to_mean=round((n[t]/(AREA[t]/100.0))/mean_d,3)) for t in SIX}
core={r['en']:r for r in R['E1_density']['rows']}
cmp_d={EN[t]:dict(here=dens[EN[t]]['density_per_km2'],core=[r['density_per_km2'] for r in R['E1_density']['rows'] if r['type']==t][0]) for t in SIX}
out['S13.landslide_density_by_type']={'value':dict(rows=dens,mean_density_per_km2=round(mean_d,3),ratio_max_to_min=round(max(d['density_per_km2'] for d in dens.values())/min(d['density_per_km2'] for d in dens.values()),1),comparison_with_core=cmp_d),
    'definition':'rainfall-triggered landslides (cases on the six analysed types) per km2 of each forest type; area from the forest type map','source':'cases_annot.csv + S3.forest_map','cls':'B'}
# ---- slope distributions of controls
sl={EN[t]:dict(n=int((CTRL.ft==t).sum()),median=round(float(CTRL[CTRL.ft==t].slope.median()),1),p5=round(float(CTRL[CTRL.ft==t].slope.quantile(.05)),1),p95=round(float(CTRL[CTRL.ft==t].slope.quantile(.95)),1),
              q1=round(float(CTRL[CTRL.ft==t].slope.quantile(.25)),1),q3=round(float(CTRL[CTRL.ft==t].slope.quantile(.75)),1)) for t in SIX}
out['S13.slope_distribution_by_type']={'value':sl,'definition':'slope statistics of the 13,131 controls by forest type (degrees)','source':'controls_annot.csv','cls':'B'}
# ---- odds ratio of each type relative to broadleaf, raw and terrain-adjusted: each type is compared with broadleaf in a
#      separate two-type sample (the same convention as the bamboo-versus-broadleaf interaction model)
orr={}
for t in SIX:
    if t=='闊葉樹林型': continue
    a=CASE[CASE.ft.isin(['闊葉樹林型',t])].copy(); a['case']=1; b=CTRL[CTRL.ft.isin(['闊葉樹林型',t])].copy(); b['case']=0
    D=pd.concat([a,b],ignore_index=True); D['slope2']=D.slope**2; D['isT']=(D.ft==t).astype(int)
    m0=smf.logit("case ~ isT",data=D).fit(disp=0,maxiter=200)
    m1=smf.logit("case ~ isT+slope+slope2+north+east+planc+profc+elev",data=D).fit(disp=0,maxiter=200)
    orr[EN[t]]=dict(n_case=int((CASE.ft==t).sum()),n=int(len(D)),OR_raw=round(math.exp(m0.params['isT']),3),raw_ci=[round(math.exp(m0.params['isT']-1.96*m0.bse['isT']),3),round(math.exp(m0.params['isT']+1.96*m0.bse['isT']),3)],
                   OR_adj=round(math.exp(m1.params['isT']),3),adj_ci=[round(math.exp(m1.params['isT']-1.96*m1.bse['isT']),3),round(math.exp(m1.params['isT']+1.96*m1.bse['isT']),3)],p_adj=float(m1.pvalues['isT']))
core3={r['type']:r for r in R['E3_or_by_type']['rows']}
cmp3={EN[t]:dict(here=[orr[EN[t]]['OR_raw'],orr[EN[t]]['OR_adj']],core=[core3[t]['OR_raw'],core3[t]['OR_adj']]) for t in SIX if t!='闊葉樹林型'}
out['S13.odds_ratio_by_type']={'value':dict(rows=orr,reference='broadleaf',comparison_with_core=cmp3),'definition':'logistic regression of case status on a forest-type indicator, each type against broadleaf in a two-type sample, unadjusted and with the Eq. (1) terrain set','source':'cases_annot.csv + controls_annot.csv','cls':'B'}
# ---- bamboo share of bamboo and broadleaf controls in 5-degree bands
bc=CTRL[CTRL.ft=='竹林'].slope.values; br=CTRL[CTRL.ft=='闊葉樹林型'].slope.values
edges=np.arange(5,61,5); bands={}
for lo,hi in zip(edges[:-1],edges[1:]):
    nb=int(((bc>lo)&(bc<=hi)).sum()); nr=int(((br>lo)&(br<=hi)).sum())
    bands[f'{lo}-{hi}']=dict(n_bamboo=nb,n_broadleaf=nr,bamboo_share_pct=round(100*nb/max(nb+nr,1),2))
above45=int((bc>45).sum())
out['S13.bamboo_share_5deg_bands']={'value':dict(bands=bands,bamboo_controls_above_45=above45,bamboo_control_p95=round(float(np.percentile(bc,95)),1)),
    'definition':'share of bamboo among the bamboo and broadleaf controls in 5-degree slope bands (lo, hi], and the number of bamboo controls per band','source':'controls_annot.csv','cls':'B'}
json.dump({'facts':out,'log':'verify_s13_reported_quantities.py'},open(V+'verified_facts_s13.json','w'),ensure_ascii=False,indent=1)
for k,v in out.items(): print(k, json.dumps(v['value'],ensure_ascii=False)[:700]); print()
# consistency with the core-analysis file
for t,c in cmp_d.items(): assert abs(c['here']-c['core'])<0.0006, (t,c)
for t,c in cmp3.items(): assert abs(c['here'][0]-c['core'][0])<0.002 and abs(c['here'][1]-c['core'][1])<0.002, (t,c)
print('densities and odds ratios by type agree with expected_outputs/results.json')
