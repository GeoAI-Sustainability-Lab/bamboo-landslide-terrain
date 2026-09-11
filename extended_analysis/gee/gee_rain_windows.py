# -*- coding: utf-8 -*-
"""Stage 10a - event rainfall windows from GPM IMERG V07 (half-hourly, 0.1 deg) on Earth Engine.
For each rainfall event of the inventory: bracket = median pre-/post-event image dates of its polygons;
affected area = union of its 5 km footprint cells; the daily area-mean IMERG series inside the bracket
locates the peak day; the event window is the contiguous run of days with area-mean rain >= 10 mm around
the peak (at most 5 days either side, at least peak-1..peak+1). Output: gee/event_windows.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import ee, json, numpy as np, pandas as pd, datetime as dt
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import Transformer
key=_os.environ['GEE_SERVICE_ACCOUNT_KEY']; sa=json.load(open(key))['client_email']
ee.Initialize(ee.ServiceAccountCredentials(sa,key),project=_os.environ['GEE_PROJECT'])
IMERG=ee.ImageCollection('NASA/GPM_L3/IMERG_V07').select('precipitation')   # mm/hr, half-hourly
FP={k:[tuple(t) for t in v] for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
ev=pd.read_csv(ROOT+'/verify/event_image_dates.csv')
tr=Transformer.from_crs(3826,4326,always_xy=True)
def area_geom(cells):
    polys=[box(cx*5000,cy*5000,(cx+1)*5000,(cy+1)*5000) for cx,cy in cells]
    u=unary_union(polys).simplify(200)
    geoms=[u] if u.geom_type=='Polygon' else list(u.geoms)
    coords=[]
    for g in geoms:
        ring=[list(tr.transform(x,y)) for x,y in g.exterior.coords]
        coords.append([ring])
    return ee.Geometry.MultiPolygon(coords,proj='EPSG:4326',geodesic=False)
def to_date(v):
    s=str(int(v)); return dt.date(int(s[:4]),int(s[4:6]),int(s[6:8]))
out={}
for ev_name,cells in FP.items():
    row=ev[ev.event==ev_name]
    if len(row)==0: print('no dates for',ev_name); continue
    row=row.iloc[0]
    b=to_date(row.before_med); a=to_date(row.after_med)
    if (a-b).days<3: a=b+dt.timedelta(days=3)
    geom=area_geom(cells)
    days=[b+dt.timedelta(days=i) for i in range((a-b).days+1)]
    dlist=ee.List([d.isoformat() for d in days])
    def daily(dstr):
        d0=ee.Date(dstr); im=IMERG.filterDate(d0,d0.advance(1,'day')).sum().multiply(0.5).rename('p')
        v=im.reduceRegion(ee.Reducer.mean(),geom,10000,bestEffort=True).get('p')
        return ee.Feature(None,{'d':dstr,'p':v})
    fc=ee.FeatureCollection(dlist.map(daily))
    vals=fc.aggregate_array('p').getInfo(); ds=fc.aggregate_array('d').getInfo()
    ser=pd.Series([np.nan if v is None else float(v) for v in vals],index=pd.to_datetime(ds))
    if ser.isna().all(): print('no rain data',ev_name); continue
    pk=ser.idxmax();
    # contiguous run >= 10 mm around the peak, max 5 days each side, at least +-1
    lo=pk; hi=pk
    for i in range(1,6):
        d=pk-pd.Timedelta(days=i)
        if d in ser.index and ser[d]>=10: lo=d
        else: break
    for i in range(1,6):
        d=pk+pd.Timedelta(days=i)
        if d in ser.index and ser[d]>=10: hi=d
        else: break
    lo=min(lo,pk-pd.Timedelta(days=1)); hi=max(hi,pk+pd.Timedelta(days=1))
    lo=max(lo,ser.index[0]); hi=min(hi,ser.index[-1])
    out[ev_name]=dict(bracket=[b.isoformat(),a.isoformat()],peak_day=pk.date().isoformat(),peak_area_mean_mm=round(float(ser.max()),1),
                      window=[lo.date().isoformat(),(hi+pd.Timedelta(days=1)).date().isoformat()],window_days=int((hi-lo).days+1),
                      window_total_area_mean_mm=round(float(ser[lo:hi].sum()),1),n_cells=len(cells),
                      series={d.date().isoformat():round(float(v),1) for d,v in ser.items() if v>=5})
    print(f"{ev_name:28s} bracket {b}..{a}  peak {pk.date()} {ser.max():.0f} mm  window {lo.date()}..{hi.date()} ({(hi-lo).days+1} d, {ser[lo:hi].sum():.0f} mm)",flush=True)
json.dump(out,open(ROOT+'/gee/event_windows.json','w'),ensure_ascii=False,indent=1)
print('done',len(out))
