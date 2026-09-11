# -*- coding: utf-8 -*-
"""Stage 10d (controls, earlier definition): 2019-2024 cloud-free median Sentinel-2 indices for the
bamboo and broadleaf controls, sampled in 300-point chunks with tileScale=16. Resumable: each finished chunk is
written to gee/s2_my_parts/ctrl_<i>.csv and skipped on restart. Output: gee/s2_controls_multiyear.csv"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import ee, json, io, os, time, glob, requests, pandas as pd
from pyproj import Transformer
key=_os.environ['GEE_SERVICE_ACCOUNT_KEY']; sa=json.load(open(key))['client_email']
ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=_os.environ['GEE_PROJECT'])
K1=['竹林','闊葉樹林型']
CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv'); co=CTRL[CTRL.ft.isin(K1)].copy().reset_index(drop=True)
os.makedirs(ROOT+'/gee/s2_my_parts',exist_ok=True)
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
base=S2.filterDate('2019-01-01','2025-01-01').map(mask).map(idx)
comp=base.median().addBands(base.select('NDVI').count().rename('n_obs'))
def sample(fc,name,tile=16):
    for k in range(10):
        try:
            s=comp.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale=10,tileScale=tile)
            url=s.getDownloadURL('csv',['uid','NDVI','NDMI','NBR','n_obs'])
            r=requests.get(url,timeout=1800)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
            print('retry',name,r.status_code,r.text[:120].replace('\n',' '),flush=True)
        except Exception as e:
            print('retry',name,'exception',str(e)[:140],flush=True)
        time.sleep(45)
    raise RuntimeError('download failed '+name)
t0=time.time(); STEP=300
for i in range(0,len(co),STEP):
    out=ROOT+f'/gee/s2_my_parts/ctrl_{i:05d}.csv'
    if os.path.exists(out): continue
    p=sample(to_fc(co.iloc[i:i+STEP],'bg_lid'),f'ctrl{i}'); p.to_csv(out,index=False)
    print('multiyear controls',min(i+STEP,len(co)),'of',len(co),f'{time.time()-t0:.0f}s',flush=True)
parts=[pd.read_csv(f) for f in sorted(glob.glob(ROOT+'/gee/s2_my_parts/ctrl_*.csv'))]
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_controls_multiyear.csv',index=False)
print('done',sum(len(p) for p in parts),flush=True)
