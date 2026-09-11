# -*- coding: utf-8 -*-
"""Stage 11d - within-period repeat / reactivation check (audit item B).
For every case of the bamboo-versus-broadleaf subset, test whether its centroid lies inside a landslide polygon of an
EARLIER 2018-2025 event (polygon whose post-event image date precedes the case's pre-event image date), and refit the
interaction without those cases. Also the looser rule: centroid inside any other 2018-2025 polygon of a different event.
Appends S11.within_period_reactivation to verified_facts_s11.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, glob, math, time, numpy as np, pandas as pd, geopandas as gpd, pyogrio, statsmodels.formula.api as smf
from scipy.stats import norm
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
U=RAW
POLY={y:glob.glob(f"{U}/_stage_polygons/{y}/*.shp")[0] for y in range(2018,2024)}
POLY[2024]=f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113/Event_Inventory_2024_ARDSWC_V7.shp"
POLY[2025]=(f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114/20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/Event_Inventory_2025_ARDSWC_V2.shp")
frames=[]
for y in sorted(POLY):
    g=pyogrio.read_dataframe(POLY[y],on_invalid='ignore').set_crs(3826,allow_override=True)
    g=g.rename(columns={'BeforeImag':'BeforeImg','AfterImage':'AfterImg'})
    g['year']=y; g['pid']=[f"{y}_{i}" for i in range(len(g))]
    keep=[c for c in ['pid','year','Events','BeforeDate','AfterDate','geometry'] if c in g.columns]
    frames.append(g[keep])
G=gpd.GeoDataFrame(pd.concat(frames,ignore_index=True),crs=3826); G['geometry']=G.geometry.buffer(0)
G['BeforeDate']=pd.to_numeric(G['BeforeDate'],errors='coerce'); G['AfterDate']=pd.to_numeric(G['AfterDate'],errors='coerce')
log(f"polygons {len(G)}, with dates {int(G.BeforeDate.notna().sum())}")
BAMBOO='竹林'; BROAD='闊葉樹林型'; K1=[BROAD,BAMBOO]; COVS=['slope','north','east','planc','profc','elev']
CA=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CO=pd.read_csv(ROOT+'/verify/controls_xy.csv')
ca=CA[CA.ft.isin(K1)].copy()
ca=ca.merge(G[['pid','BeforeDate','AfterDate']],left_on='lid',right_on='pid',how='left')
pts=gpd.GeoDataFrame(ca,geometry=gpd.points_from_xy(ca.x,ca.y),crs=3826)
J=gpd.sjoin(pts[['lid','event','BeforeDate','geometry']],G[['pid','Events','AfterDate','BeforeDate','geometry']].rename(columns={'AfterDate':'pAfter','BeforeDate':'pBefore','Events':'pEvent'}),how='inner',predicate='within')
J=J[J.pid!=J.lid]
earlier=J[(J.pAfter.notna())&(J.BeforeDate.notna())&(J.pAfter<=J.BeforeDate)]
other=J[J.pEvent!=J.event]
lids_earlier=set(earlier.lid); lids_other=set(other.lid)
def fit(excl):
    A=ca[~ca.lid.isin(excl)][COVS+['ft']].copy(); A['case']=1
    B=CO[CO.ft.isin(K1)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope",data=D).fit(disp=0)
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5),n_case=int(D.case.sum()),n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()))
base=fit(set()); r_e=fit(lids_earlier); r_o=fit(lids_other)
nb_e=int(ca[ca.lid.isin(lids_earlier)].ft.eq(BAMBOO).sum()); nb_o=int(ca[ca.lid.isin(lids_other)].ft.eq(BAMBOO).sum())
res=dict(rule_earlier_polygon=dict(definition='case centroid inside a 2018-2025 polygon of another record whose post-event image date is on or before the case pre-event image date',
                                   n_cases_flagged=len(lids_earlier),n_bamboo_flagged=nb_e,refit=r_e),
         rule_any_other_event=dict(definition='case centroid inside a 2018-2025 polygon of a different event (either order)',n_cases_flagged=len(lids_other),n_bamboo_flagged=nb_o,refit=r_o),
         base=base,n_cases=int(len(ca)),n_bamboo_cases=int((ca.ft==BAMBOO).sum()),cases_with_dates=int(ca.BeforeDate.notna().sum()))
log(f"earlier-polygon rule: {len(lids_earlier)} cases ({nb_e} bamboo) -> {r_e}; any-other-event rule: {len(lids_other)} ({nb_o} bamboo) -> {r_o}; base {base}")
F=json.load(open(ROOT+'/verify/verified_facts_s11.json'))
F['facts']['S11.within_period_reactivation']=dict(value=res,definition='within-period repeat check: cases whose centroid lies inside an earlier 2018-2025 landslide polygon, and the interaction refitted without them',source='rebuilt table',cls='A',note='complements the 2004-2017 scar screen; the inventory itself does not flag reactivations')
json.dump(F,open(ROOT+'/verify/verified_facts_s11.json','w'),ensure_ascii=False,indent=1)
log('done')
