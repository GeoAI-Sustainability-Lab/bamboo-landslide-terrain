# -*- coding: utf-8 -*-
"""Stage 10d (v2) - pre-event Sentinel-2 indices grouped by pre-event image date: one cloud-free median
composite per unique BeforeDate (window BeforeDate-80 d to BeforeDate-5 d), sampled at all cases sharing
that date (10 m). Controls: multi-year 2019-2024 median. Outputs: gee/s2_cases.csv, gee/s2_controls.csv"""
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
S0=json.load(open(ROOT+'/verify/verified_facts.json'))['facts']['S0.sources']['value']
dates={}
for k,v in S0.items():
    if not k.startswith('poly_'): continue
    y=k.split('_')[1]; df=pyogrio.read_dataframe(v['path'],read_geometry=False)
    cols={c.lower():c for c in df.columns}; bd=df[cols['beforedate']]
    for i,b in enumerate(bd):
        try: dates[f'{y}_{i}']=str(int(float(b)))
        except Exception: pass
CASE['before']=CASE.lid.map(dates)
ca=CASE[CASE.ft.isin(K1)&CASE.before.notna()].copy(); co=CTRL[CTRL.ft.isin(K1)].copy()
print('cases with dates',len(ca),'of',int(CASE.ft.isin(K1).sum()),'unique dates',ca.before.nunique(),'controls',len(co),flush=True)
tr=Transformer.from_crs(3826,4326,always_xy=True)
def to_fc(df,idcol):
    feats=[]
    for r in df.itertuples():
        lon,lat=tr.transform(r.x,r.y); feats.append(ee.Feature(ee.Geometry.Point([lon,lat]),{'uid':str(getattr(r,idcol))}))
    return ee.FeatureCollection(feats)
TW=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4])
S2=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(TW)
def mask(img):
    scl=img.select('SCL'); m=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(m).divide(10000)
def idx(img):
    return img.normalizedDifference(['B8','B4']).rename('NDVI').addBands(img.normalizedDifference(['B8','B11']).rename('NDMI')).addBands(img.normalizedDifference(['B8','B12']).rename('NBR'))
def sample(comp,fc,name):
    s=comp.reduceRegions(fc,ee.Reducer.mean(),10)
    url=s.getDownloadURL('csv',['uid','NDVI','NDMI','NBR','n_obs'])
    for k in range(8):
        try:
            r=requests.get(url,timeout=1800)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
            print('retry',name,r.status_code,r.text[:160].replace('\n',' '),flush=True)
        except Exception as e:
            print('retry',name,'exception',str(e)[:120],flush=True)
        time.sleep(30)
    raise RuntimeError('download failed '+name)
parts=[]; t0=time.time()
for d,grp in ca.groupby('before'):
    ds=f'{d[:4]}-{d[4:6]}-{d[6:8]}'; bd=ee.Date(ds)
    col=S2.filterDate(bd.advance(-80,'day'),bd.advance(-5,'day')).map(mask).map(idx)
    comp=col.median().addBands(col.select('NDVI').count().rename('n_obs'))
    fc=to_fc(grp,'lid')
    for i in range(0,len(grp),1500):
        sub=to_fc(grp.iloc[i:i+1500],'lid')
        p=sample(comp,sub,ds); p['before']=ds; parts.append(p)
    print(f'{ds} n={len(grp)} done {time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_cases.csv',index=False)
base=S2.filterDate('2019-01-01','2025-01-01').map(mask).map(idx)
comp=base.median().addBands(base.select('NDVI').count().rename('n_obs'))
parts=[]
for i in range(0,len(co),1500):
    p=sample(comp,to_fc(co.iloc[i:i+1500],'bg_lid'),f'ctrl{i}'); parts.append(p); print('controls',i+1500,f'{time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_controls.csv',index=False)
print('done')
