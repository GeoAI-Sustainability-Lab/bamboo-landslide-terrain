# -*- coding: utf-8 -*-
"""Controls that fall inside a landslide polygon of the 2018-2025 inventory, and the interaction without them.
Controls are sampled uniformly within forest, so a few of them land inside a mapped landslide of the study
period. This script counts them (all polygons of all_polys.gpkg, written by verify_all.py), lists them by
forest type, and refits the island-wide interaction on the bamboo-versus-broadleaf sample with those controls
removed. Output: controls_inside_polygons.json (read by make_tableS1.py)."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
import statsmodels.formula.api as smf
V=ROOT+'/verify/'
CTRL=pd.read_csv(V+'controls_annot.csv'); CASE=pd.read_csv(V+'cases_annot.csv')
polys=gpd.read_file(V+'all_polys.gpkg')
if polys.crs is not None and polys.crs.to_epsg()!=3826: polys=polys.to_crs(3826)
geoms=polys.geometry.values
tree=STRtree(geoms)
pts=[Point(x,y) for x,y in zip(CTRL.x,CTRL.y)]
pi,gi=tree.query(pts,predicate='within')
inside=np.zeros(len(CTRL),bool); inside[np.unique(pi)]=True
CTRL['inside']=inside
n_in=int(inside.sum())
by_type={k:int(v) for k,v in CTRL.loc[inside,'ft'].value_counts().items()}
print(f'controls inside a 2018-2025 polygon: {n_in} of {len(CTRL)}  by type {by_type}')
K=['竹林','闊葉樹林型']
F="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
def fit(ctrl):
    d=pd.concat([CASE[CASE.ft.isin(K)],ctrl[ctrl.ft.isin(K)]],ignore_index=True)
    d['bamboo']=(d.ft=='竹林').astype(int); d['slope2']=d.slope**2
    m=smf.logit(F,data=d).fit(disp=0); b,se=m.params['bamboo:slope'],m.bse['bamboo:slope']
    c=ctrl[ctrl.ft.isin(K)]
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*se),4),round(math.exp(b+1.96*se),4)],
                p=round(float(m.pvalues['bamboo:slope']),5),n_ctrl=int(len(c)),n_bamboo_ctrl=int((c.ft=='竹林').sum()))
out={'controls_inside_2018_2025_polygons':n_in,'by_type':by_type,'baseline':fit(CTRL),'excluding_inside':fit(CTRL[~CTRL.inside])}
print(json.dumps(out,ensure_ascii=False,indent=1))
json.dump(out,open(V+'controls_inside_polygons.json','w'),ensure_ascii=False,indent=1)
