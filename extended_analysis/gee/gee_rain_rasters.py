# -*- coding: utf-8 -*-
"""Stage 10b - per-event rainfall rasters from GPM IMERG V07 (half-hourly, 0.1 deg) on Earth Engine.
Event windows are the calendar dates in the inventory event names or the typhoon warning periods, padded
by one day (table below); the daily area-mean series inside each window is reported as a check that the
rainfall peak lies inside the window.  For each window three images are exported for the Taiwan box
(119.9-122.1 E, 21.8-25.4 N, 0.1 deg): total precipitation (mm), maximum 24-h accumulation (mm) and
maximum 72-h accumulation (mm), computed from the half-hourly series by a forward time join.
Climatology from CHIRPS daily (0.05 deg), 2018-2025: mean annual precipitation and mean annual maximum
daily precipitation.  Outputs: gee/rain/<event>_{total,max24,max72}.tif, gee/rain/chirps_clim.tif,
gee/event_windows_curated.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import ee, json, os, io, zipfile, requests, time, datetime as dt, numpy as np, pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import Transformer
key=_os.environ['GEE_SERVICE_ACCOUNT_KEY']; sa=json.load(open(key))['client_email']
ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=_os.environ['GEE_PROJECT'])
OUT=ROOT+'/gee/rain'; os.makedirs(OUT,exist_ok=True)
WIN={ # inclusive start, inclusive end (local dates); source: event name dates / CWA warning periods, padded 1 day
 '0613暨0619豪雨':('2018-06-13','2018-06-20'),'0613暨0619豪雨暨瑪莉亞颱風':('2018-06-13','2018-07-11'),'瑪莉亞颱風':('2018-07-09','2018-07-11'),
 '0823熱帶低壓':('2018-08-22','2018-08-29'),'0909熱帶低壓':('2018-09-07','2018-09-10'),'0909熱帶低壓暨山竹颱風':('2018-09-07','2018-09-16'),
 '0428豪雨':('2019-04-26','2019-04-29'),'0518及0611豪雨':('2019-05-16','2019-06-14'),'丹娜絲颱風':('2019-07-16','2019-07-19'),
 '利奇馬颱風':('2019-08-07','2019-08-10'),'利奇馬颱風、0815豪雨、白鹿颱風':('2019-08-07','2019-08-26'),'白鹿颱風':('2019-08-23','2019-08-26'),
 '0521暨0528豪雨':('2020-05-20','2020-05-29'),
 '彩雲颱風':('2021-06-02','2021-06-06'),'0621豪雨':('2021-06-19','2021-06-23'),'烟花颱風':('2021-07-20','2021-07-25'),'0731豪雨暨盧碧颱風':('2021-07-30','2021-08-09'),
 '璨樹颱風':('2021-09-10','2021-09-13'),'圓規颱風暨1013豪雨':('2021-10-09','2021-10-14'),
 '軒嵐諾暨梅花颱風':('2022-08-31','2022-09-14'),'1007豪雨、尼莎颱風暨1022豪雨':('2022-10-05','2022-10-24'),'1030豪雨':('2022-10-28','2022-11-01'),
 '杜蘇芮颱風':('2023-07-24','2023-07-29'),'卡努颱風':('2023-08-01','2023-08-06'),'海葵颱風':('2023-09-01','2023-09-06'),'小犬颱風':('2023-10-02','2023-10-07'),
 '0629豪雨':('2024-06-27','2024-07-02'),'凱米颱風':('2024-07-22','2024-07-27'),'山陀兒颱風':('2024-09-28','2024-10-05'),'1023豪雨':('2024-10-21','2024-10-27'),
 '康芮颱風':('2024-10-29','2024-11-02'),'1112豪雨':('2024-11-10','2024-11-15'),'天兔颱風':('2024-11-13','2024-11-17'),
 '0613豪雨':('2025-06-11','2025-06-14'),'丹娜絲颱風、薇帕颱風、0721豪雨':('2025-07-04','2025-07-23'),'0728豪雨':('2025-07-26','2025-08-03'),
 '楊柳颱風':('2025-08-11','2025-08-15'),'樺加沙颱風':('2025-09-20','2025-09-25'),'1020豪雨':('2025-10-17','2025-10-23'),'鳳凰颱風':('2025-11-09','2025-11-14'),
}
IMERG=ee.ImageCollection('NASA/GPM_L3/IMERG_V07').select('precipitation')
REGION=ee.Geometry.Rectangle([119.9,21.8,122.1,25.4],'EPSG:4326',False)
FP={k:[tuple(t) for t in v] for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
tr=Transformer.from_crs(3826,4326,always_xy=True)
def area_geom(cells):
    u=unary_union([box(cx*5000,cy*5000,(cx+1)*5000,(cy+1)*5000) for cx,cy in cells]).simplify(200)
    geoms=[u] if u.geom_type=='Polygon' else list(u.geoms)
    return ee.Geometry.MultiPolygon([[[list(tr.transform(x,y)) for x,y in g.exterior.coords]] for g in geoms],proj='EPSG:4326',geodesic=False)
def rolling_max(coll,hours):
    ms=hours*3600*1000
    filt=ee.Filter.And(ee.Filter.maxDifference(difference=ms-1,leftField='system:time_start',rightField='system:time_start'),
                       ee.Filter.lessThanOrEquals(leftField='system:time_start',rightField='system:time_start'))
    joined=ee.Join.saveAll('w').apply(coll,coll,filt)
    def acc(im):
        return ee.ImageCollection.fromImages(ee.List(im.get('w'))).sum().rename('p').set('system:time_start',im.get('system:time_start'))
    return ee.ImageCollection(joined.map(acc)).max()
def download(img,name,scale):
    url=img.getDownloadURL({'scale':scale,'crs':'EPSG:4326','region':REGION,'format':'GEO_TIFF'})
    for k in range(4):
        r=requests.get(url,timeout=300)
        if r.status_code==200: break
        time.sleep(10)
    r.raise_for_status()
    open(f'{OUT}/{name}.tif','wb').write(r.content)
report={}
for ev,(s,e) in WIN.items():
    if ev not in FP: print('not in footprints:',ev); continue
    t0=time.time()
    start=ee.Date(s); end=ee.Date(e).advance(1,'day')   # end exclusive
    # local-time offset: IMERG is UTC; Taiwan is UTC+8 -> shift window by -8 h so local calendar days are covered
    start=start.advance(-8,'hour'); end=end.advance(-8,'hour')
    coll=IMERG.filterDate(start,end).map(lambda im: im.multiply(0.5).rename('p').copyProperties(im,['system:time_start']))
    total=coll.sum().rename('total'); m24=rolling_max(coll,24).rename('max24'); m72=rolling_max(coll,72).rename('max72')
    stack=total.addBands(m24).addBands(m72).toFloat()
    download(stack,ev.replace('、','_').replace('/','_'),11132)
    # daily area-mean check series
    geom=area_geom(FP[ev]); ndays=(dt.date.fromisoformat(e)-dt.date.fromisoformat(s)).days+1
    days=[(dt.date.fromisoformat(s)+dt.timedelta(days=i)).isoformat() for i in range(ndays)]
    def daily(dstr):
        d0=ee.Date(dstr).advance(-8,'hour'); im=IMERG.filterDate(d0,d0.advance(1,'day')).sum().multiply(0.5).rename('p')
        return ee.Feature(None,{'d':dstr,'p':im.reduceRegion(ee.Reducer.mean(),geom,10000,bestEffort=True).get('p')})
    fc=ee.FeatureCollection(ee.List(days).map(daily))
    vals=fc.aggregate_array('p').getInfo(); ds=fc.aggregate_array('d').getInfo()
    ser={d:(None if v is None else round(float(v),1)) for d,v in zip(ds,vals)}
    pk=max((v,d) for d,v in ser.items() if v is not None)
    report[ev]=dict(window=[s,e],days=ndays,area_mean_daily_mm=ser,peak_day=pk[1],peak_mm=pk[0],window_total_area_mean_mm=round(sum(v for v in ser.values() if v),1),n_cells=len(FP[ev]))
    print(f"{ev:28s} {s}..{e} ({ndays} d)  peak {pk[1]} {pk[0]:.0f} mm  total {report[ev]['window_total_area_mean_mm']:.0f} mm  [{time.time()-t0:.0f}s]",flush=True)
# climatology: CHIRPS daily 2018-2025
CH=ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation')
years=list(range(2018,2026))
ann=ee.ImageCollection([CH.filterDate(f'{y}-01-01',f'{y+1}-01-01').sum().rename('annual') for y in years])
amax=ee.ImageCollection([CH.filterDate(f'{y}-01-01',f'{y+1}-01-01').max().rename('annmaxday') for y in years])
clim=ann.mean().rename('map_mm').addBands(amax.mean().rename('mean_annual_max_daily_mm')).toFloat()
download(clim,'chirps_clim_2018_2025',5566)
json.dump(report,open(ROOT+'/gee/event_windows_curated.json','w'),ensure_ascii=False,indent=1)
print('done')
