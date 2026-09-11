# -*- coding: utf-8 -*-
"""Stage 10d - pre-event Sentinel-2 vegetation indices on Earth Engine, following the original script
(legacy/gee_preevent_ndvi.js): for each landslide case the cloud-free median composite of
COPERNICUS/S2_SR_HARMONIZED between 80 and 5 days before the polygon's pre-event image date, sampled at
the case point (10 m); for controls a multi-year (2019-2024) cloud-free median.  SCL classes 3, 8, 9, 10,
11 are masked.  Indices: NDVI (B8, B4), NDMI (B8, B11), NBR (B8, B12).  Outputs: gee/s2_cases.csv, gee/s2_controls.csv"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import ee, json, io, time, requests, numpy as np, pandas as pd, pyogrio
from pyproj import Transformer
key=_os.environ['GEE_SERVICE_ACCOUNT_KEY']; sa=json.load(open(key))['client_email']
ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=_os.environ['GEE_PROJECT'])
K1=['竹林','闊葉樹林型']
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
# per-polygon pre-event image dates from the inventory shapefiles (pid = lid)
S0=json.load(open(ROOT+'/verify/verified_facts.json'))['facts']['S0.sources']['value']
dates={}
import geopandas as gpd
polys=gpd.read_file(ROOT+'/verify/all_polys.gpkg')[['pid','year']]
for k,v in S0.items():
    if not k.startswith('poly_'): continue
    y=k.split('_')[1]; df=pyogrio.read_dataframe(v['path'],read_geometry=False)
    cols={c.lower():c for c in df.columns}; bd=df[cols['beforedate']]
    for i,b in enumerate(bd):
        try: dates[f'{y}_{i}']=str(int(float(b)))
        except Exception: pass
CASE['before']=CASE.lid.map(dates)
ca=CASE[CASE.ft.isin(K1)&CASE.before.notna()].copy(); co=CTRL[CTRL.ft.isin(K1)].copy()
print('cases with dates',len(ca),'of',int(CASE.ft.isin(K1).sum()),'controls',len(co),flush=True)
tr=Transformer.from_crs(3826,4326,always_xy=True)
def to_fc(df,idcol,with_date):
    feats=[]
    for r in df.itertuples():
        lon,lat=tr.transform(r.x,r.y)
        props={'uid':getattr(r,idcol)}
        if with_date:
            b=r.before; props['before']=f'{b[:4]}-{b[4:6]}-{b[6:8]}'
        feats.append(ee.Feature(ee.Geometry.Point([lon,lat]),props))
    return ee.FeatureCollection(feats)
S2=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
def mask(img):
    scl=img.select('SCL'); m=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(m).divide(10000)
def idx(img):
    return img.normalizedDifference(['B8','B4']).rename('NDVI').addBands(img.normalizedDifference(['B8','B11']).rename('NDMI')).addBands(img.normalizedDifference(['B8','B12']).rename('NBR'))
def per_case(f):
    bd=ee.Date(ee.String(f.get('before')))
    col=S2.filterBounds(f.geometry()).filterDate(bd.advance(-80,'day'),bd.advance(-5,'day')).map(mask).map(idx)
    n=col.size()
    comp=ee.Image(ee.Algorithms.If(n.gt(0),col.median(),ee.Image.constant([0,0,0]).rename(['NDVI','NDMI','NBR']).updateMask(ee.Image.constant(0))))
    v=comp.reduceRegion(ee.Reducer.mean(),f.geometry(),10)
    return f.set({'NDVI':v.get('NDVI'),'NDMI':v.get('NDMI'),'NBR':v.get('NBR'),'n_obs':n})
def fetch(fc,name):
    url=fc.getDownloadURL('csv',['uid','NDVI','NDMI','NBR','n_obs'])
    for k in range(5):
        r=requests.get(url,timeout=1200)
        if r.status_code==200: break
        print('retry',name,r.status_code,r.text[:200],flush=True); time.sleep(30)
    r.raise_for_status(); return pd.read_csv(io.StringIO(r.text))
# cases in chunks of 500 (per-feature composites are expensive)
parts=[]; t0=time.time()
for i in range(0,len(ca),500):
    chunk=ca.iloc[i:i+500]
    fc=to_fc(chunk,'lid',True).map(per_case)
    parts.append(fetch(fc,f'cases{i}')); print('cases',i+len(chunk),'/',len(ca),f'{time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_cases.csv',index=False)
# controls: multi-year median 2019-2024 sampled at 10 m (chunks of 2000)
base=S2.filterBounds(ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])).filterDate('2019-01-01','2025-01-01').map(mask).map(idx)
comp=base.median().addBands(base.select('NDVI').count().rename('n_obs'))
parts=[]
for i in range(0,len(co),2000):
    chunk=co.iloc[i:i+2000]; fc=to_fc(chunk,'bg_lid',False)
    s=comp.reduceRegions(fc,ee.Reducer.mean(),10).map(lambda f: f.set('n_obs',f.get('n_obs')))
    url=s.getDownloadURL('csv',['uid','NDVI','NDMI','NBR','n_obs'])
    r=requests.get(url,timeout=1200); r.raise_for_status(); parts.append(pd.read_csv(io.StringIO(r.text)))
    print('controls',i+len(chunk),'/',len(co),f'{time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_controls.csv',index=False)
print('done')
