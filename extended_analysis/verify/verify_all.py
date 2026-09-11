# -*- coding: utf-8 -*-
"""single deterministic end-to-end verification.

Rebuilds the analysis table from the raw official layers, verifies it against the
published artefacts, and recomputes every number used in the revision, with fixed
seeds and frozen definitions. Writes verified_facts.json.

Run:  python3 verify_all.py
"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import os, sys, json, math, time, hashlib, warnings, gc
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

T0 = time.time()
OUT = {}          # fact register
LOG = []

def log(msg):
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True); LOG.append(s)

def fact(fid, value, definition, source, cls, note=""):
    OUT[fid] = dict(value=value, definition=definition, source=source, cls=cls, note=note)
    return value

def sha256(path, cap=None):
    h = hashlib.sha256(); n = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk); n += len(chunk)
            if cap and n >= cap: break
    return h.hexdigest()[:16]

# ----------------------------------------------------------------- paths
U   = RAW
DEM = f"{U}/Dataset/01_SOURCE/Terrain_Canopy/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif"
F4  = RAW+"/forest4/f4.shp"
OVL = f"{U}/Dataset/99_INBOX/Downloads_archives/overlay_result.csv"
LC  = ROOT+"/../data/landslide_centroids.csv"
BG  = ROOT+"/../data/bg_points.csv"
S1  = ROOT+"/../data/step1_dataset.csv"
RJ  = ROOT+"/../expected_outputs/results.json"
POLY = {}
import glob
for y in range(2018, 2024):
    g = glob.glob(f"{U}/_stage_polygons/{y}/*.shp"); POLY[y] = g[0]
POLY[2024] = f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113/Event_Inventory_2024_ARDSWC_V7.shp"
POLY[2025] = (f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114/"
              "20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/"
              "Event_Inventory_2025_ARDSWC_V2.shp")

# frozen constants
SEED_POOL   = 20260901     # new control pool
SEED_RAND   = 2026         # random-cell-in-polygon
SEED_MIS    = 5            # misclassification simulation
SEED_BOOT   = 20260902     # crossover bootstrap
SEED_EVCELL = 11           # event x cell control draw
CELL_M      = 5000.0       # frozen event-footprint cell size
GRID_LO, GRID_HI, GRID_STEP = 5.0, 60.0, 0.1   # frozen slope grid for curve statistics
BAMBOO  = '竹林'
BROAD   = '闊葉樹林型'
MIXED   = '竹闊混淆林'
FT6     = ['待成林地', BAMBOO, MIXED, '針葉樹林型', '針闊葉樹混淆', BROAD]
COVS    = ['slope', 'north', 'east', 'planc', 'profc', 'elev']
FORM    = "case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"

# ================================================================= STAGE 0
log("STAGE 0  source inventory")
srcs = {}
for name, p in [('dem_20m.tif', DEM), ('forest4.shp', F4), ('overlay_result.csv', OVL),
                ('landslide_centroids.csv', LC), ('bg_points.csv', BG), ('step1_dataset.csv', S1),
                ('results.json', RJ)] + [(f'poly_{y}', POLY[y]) for y in sorted(POLY)]:
    srcs[name] = dict(path=p, bytes=os.path.getsize(p), sha256_16=sha256(p, cap=64 << 20))
    log(f"   {name:26s} {srcs[name]['bytes']:>12,} B  sha16={srcs[name]['sha256_16']}")
fact('S0.sources', srcs, "raw input files with size and SHA-256 prefix (first 64 MB)",
     "filesystem", "A", "sha over first 64 MB only for files >64 MB")

# ================================================================= STAGE 1
log("STAGE 1  landslide inventory")
import geopandas as gpd, pyogrio
lc = pd.read_csv(LC)
fact('S1.n_polygons_total', int(len(lc)), "records in the 2018-2025 event-based inventory",
     "landslide_centroids.csv", "A")
by_year = lc.groupby('year').size().to_dict()
trig = lc.trigger.value_counts().to_dict()
fact('S1.by_year', {int(k): int(v) for k, v in by_year.items()}, "inventory records per year",
     "landslide_centroids.csv", "A")
fact('S1.by_trigger', {k: int(v) for k, v in trig.items()}, "inventory records per trigger class",
     "landslide_centroids.csv", "A")
frames = []
for y in sorted(POLY):
    g = pyogrio.read_dataframe(POLY[y], on_invalid='ignore').set_crs(3826, allow_override=True)
    g = g.rename(columns={'BeforeImag': 'BeforeImg', 'AfterImage': 'AfterImg'})
    g['year'] = y; g['pid'] = [f"{y}_{i}" for i in range(len(g))]
    frames.append(g[['pid', 'year', 'Events', 'Area_ha', 'geometry']])
G = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=3826)
G['geometry'] = G.geometry.buffer(0)
poly_year = G.groupby('year').size().to_dict()
ok_year = all(int(poly_year[y]) == int(by_year[y]) for y in poly_year)
fact('S1.polygon_counts_match_centroids', bool(ok_year),
     "per-year polygon count equals per-year centroid count for every year 2018-2025",
     "annual shapefiles vs landslide_centroids.csv", "A",
     f"polygons per year = {({int(k):int(v) for k,v in poly_year.items()})}")
cen = G.geometry.centroid
G['cx'] = cen.x; G['cy'] = cen.y
mm = lc.merge(G[['pid', 'cx', 'cy', 'Events', 'Area_ha']], left_on='lid', right_on='pid', how='left')
d = np.hypot(mm.x - mm.cx, mm.y - mm.cy)
fact('S1.centroid_offset_m', dict(median=float(np.nanmedian(d)), p99=float(np.nanpercentile(d, 99)),
                                  max=float(np.nanmax(d))),
     "planar distance between landslide_centroids.csv coordinates and the polygon centroid",
     "annual shapefiles vs landslide_centroids.csv", "A")
fact('S1.event_label_mismatches', int((mm.event != mm.Events).sum()),
     "records whose event name differs between the centroid table and the shapefile",
     "annual shapefiles vs landslide_centroids.csv", "A")
fact('S1.area_max_abs_diff_ha', float(np.nanmax(np.abs(mm.area_ha - mm.Area_ha))),
     "largest absolute difference in mapped area", "annual shapefiles vs landslide_centroids.csv", "A")
G[['pid','year','Events','Area_ha','geometry']].to_file(ROOT+'/verify/all_polys.gpkg', layer='ls', driver='GPKG')
log(f"   polygons {len(G)}  centroid offset median {np.nanmedian(d):.4f} m  max {np.nanmax(d):.2f} m")

# ================================================================= STAGE 2
log("STAGE 2  terrain reconstruction")
import rasterio
src = rasterio.open(DEM)
Tr = src.transform; L = 20.0; NOD = src.nodata
fact('S2.dem', dict(width=src.width, height=src.height, dtype=str(src.dtypes[0]),
                    res=list(src.res), nodata=float(NOD), origin=[Tr.c, Tr.f], epsg=3826),
     "digital elevation model header", "dem_20m.tif", "A")
A = src.read(1).astype(np.float64)

def rc_from_xy(x, y):
    """nearest-cell-centre sampling (verified convention)"""
    col = np.round((np.asarray(x) - Tr.c) / Tr.a - 0.5).astype(int)
    row = np.round((np.asarray(y) - Tr.f) / Tr.e - 0.5).astype(int)
    return row, col

def terrain(row, col):
    Z1=A[row-1,col-1]; Z2=A[row-1,col]; Z3=A[row-1,col+1]
    Z4=A[row,col-1];   Z5=A[row,col];   Z6=A[row,col+1]
    Z7=A[row+1,col-1]; Z8=A[row+1,col]; Z9=A[row+1,col+1]
    gx=(Z6-Z4)/(2*L); gy=(Z8-Z2)/(2*L)
    slope=np.degrees(np.arctan(np.hypot(gx,gy))); asp=np.arctan2(gy,-gx)
    D=((Z4+Z6)/2-Z5)/L**2; E=((Z2+Z8)/2-Z5)/L**2; F=(-Z1+Z3+Z7-Z9)/(4*L**2)
    Gg=gx; H=-gy; den=Gg**2+H**2; den2=np.where(den==0, np.nan, den)
    ok=np.all(np.vstack([Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9])!=NOD, axis=0)
    horn=np.degrees(np.arctan(np.hypot(((Z3+2*Z6+Z9)-(Z1+2*Z4+Z7))/(8*L),
                                       ((Z7+2*Z8+Z9)-(Z1+2*Z2+Z3))/(8*L))))
    return dict(slope=slope, north=np.cos(asp), east=-np.sin(asp),
                planc=2*(D*H**2+E*Gg**2-F*Gg*H)/den2,
                profc=-2*(D*Gg**2+E*H**2+F*Gg*H)/den2,
                elev=Z5, ok=ok, slope_horn=horn)

d1 = pd.read_csv(S1)
ov = pd.read_csv(OVL)
m  = lc.merge(ov, on='lid', how='left')
rain = m[(m.trigger == '降雨') & (m.forest_type.isin(FT6))].reset_index(drop=True)
cs = d1[d1.case == 1].reset_index(drop=True)
fact('S2.case_selection_rule', dict(n=int(len(rain)), rule="trigger == rainfall AND forest_type in the six analysed classes"),
     "the filter that reproduces the case rows of step1_dataset.csv exactly",
     "landslide_centroids.csv + overlay_result.csv", "A")
assert len(rain) == len(cs) == 7454
fact('S2.case_ft_sequence_identical', bool((rain.forest_type.values == cs.ft.values).all()),
     "the forest-type sequence of the filtered centroid table equals that of step1_dataset cases, row for row",
     "landslide_centroids.csv + overlay_result.csv vs step1_dataset.csv", "A")
r, c = rc_from_xy(rain.x.values, rain.y.values)
t = terrain(r, c)
diffs = {k: float(np.nanmax(np.abs(t[k] - cs[k].values))) for k in COVS}
fact('S2.terrain_max_abs_diff', diffs,
     "largest absolute difference between recomputed terrain and step1_dataset, over the 7,454 cases",
     "dem_20m.tif recomputation (Zevenbergen-Thorne, nearest-cell sampling)", "A")
hd = np.abs(t['slope_horn'] - cs.slope.values)
fact('S2.slope_vs_horn', dict(median_deg=float(np.median(hd)), max_deg=float(np.max(hd)),
                              pct_gt_2deg=float(100*np.mean(hd > 2))),
     "difference between the Horn slope and the slope actually stored in step1_dataset",
     "dem_20m.tif recomputation", "A",
     "the stored slope is Zevenbergen-Thorne, not Horn")
log(f"   terrain max abs diff: " + ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()))
log(f"   Horn differs by median {np.median(hd):.2f} deg, max {np.max(hd):.2f} deg")

# ================================================================= STAGE 3
log("STAGE 3  forest type reconstruction")
finfo = pyogrio.read_info(F4)
fdf = pyogrio.read_dataframe(F4, columns=['TypeName', 'Area_Ha'], read_geometry=False)
tt = fdf.groupby('TypeName').Area_Ha.agg(['sum', 'size']).sort_values('sum', ascending=False)
fact('S3.forest_map', dict(polygons=int(finfo['features']), total_area_ha=float(fdf.Area_Ha.sum()),
                           distinct_typename_strings=int(fdf.TypeName.nunique()),
                           area_ha_by_type={k: float(v) for k, v in tt['sum'].head(8).items()},
                           polygons_by_type={k: int(v) for k, v in tt['size'].head(8).items()}),
     "Fourth Forest Resource Inventory type map summary", "forest4.shp", "A")
bg = pd.read_csv(BG)
pts = pd.concat([rain[['lid', 'x', 'y']].assign(kind='case', key=lambda z: z.lid),
                 bg[['lid', 'x', 'y']].assign(kind='bg', key=lambda z: z.lid)], ignore_index=True)
Gp = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y), crs=3826)
res = []
N = int(finfo['features']); STEP = 40000
for s0 in range(0, N, STEP):
    FT = pyogrio.read_dataframe(F4, columns=['TypeName'], skip_features=s0, max_features=STEP, on_invalid='ignore')
    FT = FT[FT.geometry.notna()].set_crs(3826, allow_override=True)
    FT['geometry'] = FT.geometry.buffer(0)
    j = gpd.sjoin(Gp, FT[['TypeName', 'geometry']], how='inner', predicate='within')
    res.append(pd.DataFrame({'key': j.key.values, 'TypeName': j.TypeName.values}))
    del FT, j; gc.collect()
R = pd.concat(res).drop_duplicates('key')
FTJ = Gp[['key', 'kind', 'x', 'y']].merge(R, on='key', how='left')
FTJ.to_csv(ROOT+'/verify/ft_join.csv', index=False)
vc = FTJ[FTJ.kind == 'case'].merge(ov, left_on='key', right_on='lid', how='left')
agree = int((vc.TypeName.fillna('__NONFOREST__') == vc.forest_type).sum())
fact('S3.case_forest_type_agreement', dict(n=int(len(vc)), agree=agree, pct=round(100*agree/len(vc), 4)),
     "forest type recomputed at the polygon CENTROID vs the published overlay_result.csv",
     "forest4.shp point-in-polygon at centroids", "A")
bgc = FTJ[FTJ.kind == 'bg'].TypeName.value_counts()
ctl = d1[d1.case == 0].ft.value_counts()
match = all(int(bgc.get(k, 0)) == int(ctl.get(k, 0)) for k in FT6)
fact('S3.control_counts_match', dict(match=bool(match),
                                     recomputed={k: int(bgc.get(k, 0)) for k in FT6},
                                     step1={k: int(ctl.get(k, 0)) for k in FT6},
                                     bg_on_forest_land=int(FTJ[FTJ.kind=='bg'].TypeName.notna().sum()),
                                     bg_in_six_classes=int(sum(int(bgc.get(k,0)) for k in FT6)),
                                     bg_excluded_rare_classes=int(FTJ[FTJ.kind=='bg'].TypeName.notna().sum()
                                                                  - sum(int(bgc.get(k,0)) for k in FT6))),
     "forest type recomputed at the 50,000 background points vs the control composition of step1_dataset",
     "forest4.shp point-in-polygon at background points", "A")
log(f"   case forest-type agreement {agree}/{len(vc)}  control counts match: {match}")

# ================================================================= STAGE 4
log("STAGE 4  control coordinate recovery")
colb, rowb = None, None
rb, cb = rc_from_xy(bg.x.values, bg.y.values)
inr = (rb >= 1) & (rb < A.shape[0]-1) & (cb >= 1) & (cb < A.shape[1]-1)
rb2 = np.clip(rb, 1, A.shape[0]-2); cb2 = np.clip(cb, 1, A.shape[1]-2)
tb = terrain(rb2, cb2)
valid = inr & tb['ok'] & np.isfinite(tb['planc']) & np.isfinite(tb['profc'])
B = pd.DataFrame({'lid': bg.lid, 'x': bg.x, 'y': bg.y, 'valid': valid,
                  **{k: tb[k] for k in COVS}})
ct = d1[d1.case == 0].reset_index(drop=True)
from collections import defaultdict
def keyset(df):
    return list(zip(*[np.round(df[k].values, 9 if k != 'elev' else 6) for k in COVS]))
idx = defaultdict(list)
for i, k in enumerate(keyset(B)):
    if valid[i]: idx[k].append(i)
hits = [idx.get(k, []) for k in keyset(ct)]
n1 = sum(1 for h in hits if len(h) == 1)
seq = [h[0] for h in hits if len(h) == 1]
mono = all(seq[i] < seq[i+1] for i in range(len(seq)-1))
fact('S4.control_match', dict(n_controls=int(len(ct)), unique_on_six_covariates=int(n1),
                              ambiguous=int(sum(1 for h in hits if len(h) > 1)),
                              unmatched=int(sum(1 for h in hits if len(h) == 0)),
                              strictly_increasing_in_bg_order=bool(mono)),
     "recovery of control coordinates by matching all six terrain covariates against the background points",
     "bg_points.csv + dem_20m.tif vs step1_dataset.csv", "A")
# monotone completion
assign = np.full(len(ct), -1, dtype=int)
for j, h in enumerate(hits):
    if len(h) == 1: assign[j] = h[0]
prev = -1
for j in range(len(ct)):
    if assign[j] >= 0: prev = assign[j]; continue
    k = j
    while k < len(ct) and assign[k] < 0: k += 1
    nxt = assign[k] if k < len(ct) else len(B)
    cand = [q for q in hits[j] if prev < q < nxt]
    if cand: assign[j] = cand[0]
    else:
        lo, hi = prev+1, nxt
        if hi > lo:
            dist = np.abs(B.slope.values[lo:hi]-ct.slope.values[j]) + np.abs(B.elev.values[lo:hi]-ct.elev.values[j])/1000.
            assign[j] = lo + int(np.nanargmin(dist))
    prev = assign[j] if assign[j] >= 0 else prev
sub = B.iloc[assign].reset_index(drop=True)
resid = {k: float(np.nanmax(np.abs(sub[k].values - ct[k].values))) for k in COVS}
fact('S4.residual_after_completion', resid,
     "largest covariate mismatch after monotone completion of the control mapping",
     "bg_points.csv + dem_20m.tif vs step1_dataset.csv", "A")
CTRL = ct.copy(); CTRL['bg_lid'] = sub.lid.values; CTRL['x'] = sub.x.values; CTRL['y'] = sub.y.values
CASE = pd.concat([cs.reset_index(drop=True),
                  rain[['lid','year','x','y','event','area_ha']].reset_index(drop=True)], axis=1)
CASE.to_csv(ROOT+'/verify/cases_xy.csv', index=False)
CTRL.to_csv(ROOT+'/verify/controls_xy.csv', index=False)
del A; gc.collect()
log(f"   controls matched uniquely {n1}/{len(ct)}, monotone {mono}, residual slope {resid['slope']:.2e}")

json.dump({'facts': OUT, 'log': LOG}, open(ROOT+'/verify/verified_facts_partial.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
log("stages 0-4 written to verified_facts_partial.json")
