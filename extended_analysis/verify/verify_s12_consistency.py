# -*- coding: utf-8 -*-
"""Stage 12 (2026-09-09) - register three statements of the revised text that had no entry in the fact registers:
(1) distance from each of the 13,131 controls to the nearest 2018-2025 landslide polygon (median; share within 100 m),
    against all 13,900 polygons and against the 11,090 rainfall polygons;
(2) agreement between the forest type at the polygon centroid and the area-majority forest type of the polygon,
    for the 7,454 rainfall cases on the six analysed types and for the 5,915 bamboo-versus-broadleaf cases;
(3) mean share of pure-bamboo area inside the polygons of the 171 bamboo cases.
Output: verified_facts_s12.json (class A: raw layers)."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, time, warnings
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from shapely.strtree import STRtree
warnings.filterwarnings('ignore'); t0=time.time()
V=ROOT+'/verify/'
ctrl=pd.read_csv(V+'controls_xy.csv'); cases=pd.read_csv(V+'cases_annot.csv')
polys=gpd.read_file(V+'all_polys.gpkg')  # 13,900 polygons, EPSG:3826, MultiPolygon Z
polys['geometry']=polys.geometry.force_2d() if hasattr(polys.geometry,'force_2d') else polys.geometry
print('polys',len(polys),'controls',len(ctrl),'cases',len(cases),round(time.time()-t0),'s',flush=True)
# ---- (1) control distance to nearest polygon
gc=gpd.GeoDataFrame(ctrl,geometry=gpd.points_from_xy(ctrl.x,ctrl.y),crs='EPSG:3826')
def nearest_dist(points,geoms):
    tree=STRtree(geoms.values); idx=tree.nearest(points.geometry.values); return np.array([points.geometry.values[i].distance(geoms.values[j]) for i,j in enumerate(idx)])
d_all=nearest_dist(gc,polys.geometry)
rain_ids=set(cases.lid) if 'lid' in cases else None
rain_polys=polys[polys.pid.astype(str).isin(set(cases.lid.astype(str)))] if 'lid' in cases else polys
# rainfall polygons: use the event field (rain events = those present in the case table)
rain_events=set(cases.event.unique()); rp=polys[polys.Events.isin(rain_events)]
d_rain=nearest_dist(gc,rp.geometry)
out={}
out['S12.control_distance_to_nearest_polygon']={'value':{
    'all_2018_2025_polygons':{'n_polygons':int(len(polys)),'median_m':float(np.round(np.median(d_all),0)),'share_within_100m_pct':float(np.round(100*np.mean(d_all<=100),1)),'share_within_500m_pct':float(np.round(100*np.mean(d_all<=500),1)),'share_inside_pct':float(np.round(100*np.mean(d_all==0),2))},
    'rainfall_polygons_of_case_events':{'n_polygons':int(len(rp)),'median_m':float(np.round(np.median(d_rain),0)),'share_within_100m_pct':float(np.round(100*np.mean(d_rain<=100),1)),'share_within_500m_pct':float(np.round(100*np.mean(d_rain<=500),1)),'share_inside_pct':float(np.round(100*np.mean(d_rain==0),2))},
    'n_controls':int(len(gc))},
    'definition':'planar distance (EPSG:3826) from each of the 13,131 controls to the nearest landslide polygon of the 2018-2025 inventory','source':'controls_xy.csv + all_polys.gpkg','cls':'A'}
print(json.dumps(out,ensure_ascii=False),round(time.time()-t0),'s',flush=True)
# ---- (2)(3) centroid type vs area-majority type; bamboo area share of bamboo cases
F4=RAW+'/forest4/f4.shp'; NF=pyogrio.read_info(F4)['features']
six=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹林','竹闊混淆林','待成林地']
cp=polys[polys.pid.astype(str).isin(set(cases.lid.astype(str)))].copy()
cp['lid']=cp.pid.astype(str); cases['lid']=cases.lid.astype(str)
cp=cp.merge(cases[['lid','ft']],on='lid',how='inner').reset_index(drop=True); print('case polygons matched',len(cp),flush=True)
cp['geometry']=cp.geometry.buffer(0)
ctree=STRtree(cp.geometry.values)
areas=[dict() for _ in range(len(cp))]
for s0 in range(0,NF,40000):
    FT=pyogrio.read_dataframe(F4,columns=['TypeName'],skip_features=s0,max_features=40000,on_invalid='ignore',force_2d=True)
    FT=FT[FT.geometry.notna()]
    hits=ctree.query(FT.geometry.values,predicate='intersects')  # (2, n) pairs: forest idx, case idx
    for fi,ci in zip(hits[0],hits[1]):
        a=cp.geometry.values[ci].intersection(FT.geometry.values[fi]).area
        if a>0: areas[ci][FT.TypeName.values[fi]]=areas[ci].get(FT.TypeName.values[fi],0.0)+a
    print('forest chunk',s0,round(time.time()-t0),'s',flush=True)
maj=[max(a,key=a.get) if a else None for a in areas]
bam_share=[(a.get('竹林',0.0)/g.area) if a else np.nan for a,g in zip(areas,cp.geometry.values)]
cp['majority_ft']=maj; cp['bamboo_area_share']=bam_share
agree=(cp.majority_ft==cp.ft)
bb=cp[cp.ft.isin(['竹林','闊葉樹林型'])]
out['S12.centroid_vs_area_majority_type']={'value':{
    'six_types':{'n':int(len(cp)),'agree_pct':float(np.round(100*agree.mean(),1)),'n_disagree':int((~agree).sum())},
    'bamboo_vs_broadleaf':{'n':int(len(bb)),'agree_pct':float(np.round(100*(bb.majority_ft==bb.ft).mean(),1)),'n_disagree':int((bb.majority_ft!=bb.ft).sum())},
    'bamboo_cases':{'n':int((cp.ft=='竹林').sum()),'agree_pct':float(np.round(100*agree[cp.ft=='竹林'].mean(),1)),'mean_bamboo_area_share_pct':float(np.round(100*np.nanmean(cp.loc[cp.ft=='竹林','bamboo_area_share']),1)),'median_bamboo_area_share_pct':float(np.round(100*np.nanmedian(cp.loc[cp.ft=='竹林','bamboo_area_share']),1))}},
    'definition':'forest type read at the polygon centroid (as in the analysis table) compared with the type holding the largest intersected area of the polygon; bamboo area share = intersected pure-bamboo area / polygon area','source':'all_polys.gpkg + forest4/f4.shp + cases_annot.csv','cls':'A'}
json.dump({'facts':out,'log':f'verify_s12_consistency.py {time.strftime("%Y-%m-%d %H:%M")} {round(time.time()-t0)} s'},open(V+'verified_facts_s12.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out['S12.centroid_vs_area_majority_type'],ensure_ascii=False),round(time.time()-t0),'s')
