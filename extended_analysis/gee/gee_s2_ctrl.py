# -*- coding: utf-8 -*-
"""Stage 10d (controls) - pre-event Sentinel-2 indices for the bamboo and broadleaf controls.
(1) like-for-like: each control takes the pre-event image date of its nearest landslide case (same rule as
    the event assignment of controls in the mixed model) and is sampled from the same 80-to-5-day cloud-free
    median composite as the cases -> gee/s2_controls.csv
(2) earlier definition: 2019-2024 cloud-free median, sampled with tileScale to stay within the
    Earth Engine memory limit -> gee/s2_controls_multiyear.csv"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import ee, json, io, time, requests, numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
key=_os.environ['GEE_SERVICE_ACCOUNT_KEY']; sa=json.load(open(key))['client_email']
ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=_os.environ['GEE_PROJECT'])
K1=['竹林','闊葉樹林型']
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
sc=pd.read_csv(ROOT+'/gee/s2_cases.csv'); sc['uid']=sc.uid.astype(str)
ca=CASE[CASE.ft.isin(K1)].copy(); ca['lid']=ca.lid.astype(str)
ca=ca.merge(sc[['uid','before']].drop_duplicates('uid').rename(columns={'uid':'lid'}),on='lid',how='inner')
co=CTRL[CTRL.ft.isin(K1)].copy()
tree=cKDTree(ca[['x','y']].values); d,i=tree.query(co[['x','y']].values,k=1)
co['before']=ca.before.values[i]; co['dist_nearest_case_m']=d
co[['bg_lid','before','dist_nearest_case_m']].to_csv(ROOT+'/gee/s2_controls_assigned_dates.csv',index=False)
print('controls',len(co),'dates',co.before.nunique(),'median distance to nearest case m',round(float(np.median(d))),flush=True)
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
def sample(comp,fc,name,tile=1):
    for k in range(8):
        try:
            s=comp.reduceRegions(collection=fc,reducer=ee.Reducer.mean(),scale=10,tileScale=tile)
            url=s.getDownloadURL('csv',['uid','NDVI','NDMI','NBR','n_obs'])
            r=requests.get(url,timeout=1800)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
            print('retry',name,r.status_code,r.text[:160].replace('\n',' '),flush=True)
        except Exception as e:
            print('retry',name,'exception',str(e)[:160],flush=True)
        time.sleep(30)
    raise RuntimeError('download failed '+name)
t0=time.time(); parts=[]
for ds,grp in co.groupby('before'):
    bd=ee.Date(ds)
    col=S2.filterDate(bd.advance(-80,'day'),bd.advance(-5,'day')).map(mask).map(idx)
    comp=col.median().addBands(col.select('NDVI').count().rename('n_obs'))
    for i in range(0,len(grp),1500):
        p=sample(comp,to_fc(grp.iloc[i:i+1500],'bg_lid'),ds); p['before']=ds; parts.append(p)
    print(f'{ds} n={len(grp)} done {time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_controls.csv',index=False)
print('controls like-for-like written',flush=True)
base=S2.filterDate('2019-01-01','2025-01-01').map(mask).map(idx)
comp=base.median().addBands(base.select('NDVI').count().rename('n_obs'))
parts=[]
for i in range(0,len(co),300):
    p=sample(comp,to_fc(co.iloc[i:i+300],'bg_lid'),f'ctrl{i}',tile=16); parts.append(p); print('multiyear controls',i+300,f'{time.time()-t0:.0f}s',flush=True)
pd.concat(parts,ignore_index=True).to_csv(ROOT+'/gee/s2_controls_multiyear.csv',index=False)
print('done',flush=True)
