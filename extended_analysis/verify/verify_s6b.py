# -*- coding: utf-8 -*-
"""Stage 6b: polygon representation, source point, failure-type screens, misclassification,
three-level model, per-event estimates, leave-one-event-out, crossover bootstrap, common support.
Frozen definitions and seeds throughout."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings,gc,glob,pickle
import numpy as np,pandas as pd,geopandas as gpd,pyogrio,rasterio
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm,chi2
from shapely.geometry import Point
from shapely.prepared import prep
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
SRC=json.load(open(ROOT+'/verify/verified_facts_s6a2.json')) if glob.glob(ROOT+'/verify/verified_facts_s6a2.json') \
    else json.load(open(ROOT+'/verify/verified_facts_s6a.json'))
OUT=SRC['facts']
def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s,flush=True); LOG.append(s)
def fact(fid,v,d,s,c,n=""):
    OUT[fid]=dict(value=v,definition=d,source=s,cls=c,note=n); return v
BAMBOO='竹林'; BROAD='闊葉樹林型'; MIXED='竹闊混淆林'
K1=[BROAD,BAMBOO]; K2=[BROAD,BAMBOO,MIXED]
COVS=['slope','north','east','planc','profc','elev']
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
SPL ="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
U=RAW
DEM=f"{U}/Dataset/01_SOURCE/Terrain_Canopy/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif"
F4=RAW+"/forest4/f4.shp"; AUX=f"{U}/_stage_polygons/auxlayers"
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')
POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
lc=pd.read_csv(ROOT+'/../data/landslide_centroids.csv')

def build(a,b,bam,keep):
    A=a[a.ft.isin(keep)][COVS+['ft']].copy(); A['case']=1
    B=b[b.ft.isin(keep)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['bamboo']=D.ft.isin(bam).astype(int); D['slope2']=D.slope**2
    return D
def inter(D):
    m=smf.logit(FORM,data=D).fit(disp=0); b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=float(2*(1-norm.cdf(abs(b/s)))),n_case=int(D.case.sum()),
                n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),
                n_ctrl=int((D.case==0).sum()),n_bamboo_ctrl=int(((D.case==0)&(D.bamboo==1)).sum()))
def curve(D,grid=GRID):
    sp=smf.logit(SPL,data=D).fit(disp=0); di=sp.model.data.orig_exog.design_info
    med={c:float(np.median(D[c])) for c in ['north','east','planc','profc','elev']}
    row=lambda b: pd.DataFrame({'slope':grid,'bamboo':b,**med})
    d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0)))
    lo=d_@sp.params.values; V=sp.cov_params().values
    se=np.sqrt(np.einsum('ij,jk,ik->i',d_,V,d_))
    return lo,se
def cross_from(lo,grid=GRID):
    s=np.sign(lo)
    for i in range(len(grid)-1):
        if s[i]<0 and s[i+1]>=0:
            return float(grid[i]-lo[i]*(grid[i+1]-grid[i])/(lo[i+1]-lo[i]))
    return None
def bands(lo,se,grid=GRID):
    hi=np.exp(lo+1.96*se); l=np.exp(lo-1.96*se)
    p=grid[hi<1]; w=grid[l>1]
    f=lambda a: None if len(a)==0 else [float(a.min()),float(a.max()),bool(np.allclose(np.diff(a),0.1,atol=1e-6))]
    return f(p),f(w)

# ---------------------------------------------------------------- 6b.1 polygon cells
log("6b.1 DEM cells inside each landslide polygon")
src=rasterio.open(DEM); Tr=src.transform; L=20.; NOD=src.nodata; A=src.read(1).astype(np.float64)
X0,Y0=Tr.c,Tr.f
G=gpd.read_file(ROOT+'/verify/all_polys.gpkg',layer='ls').merge(lc[['lid','trigger']],left_on='pid',right_on='lid')
ovl=pd.read_csv(f"{U}/Dataset/99_INBOX/Downloads_archives/overlay_result.csv")
G=G.merge(ovl,on='lid')
R=G[(G.trigger=='降雨')].reset_index(drop=True)
def terr(rr,cc):
    Z1=A[rr-1,cc-1];Z2=A[rr-1,cc];Z3=A[rr-1,cc+1];Z4=A[rr,cc-1];Z5=A[rr,cc];Z6=A[rr,cc+1]
    Z7=A[rr+1,cc-1];Z8=A[rr+1,cc];Z9=A[rr+1,cc+1]
    gx=(Z6-Z4)/(2*L); gy=(Z8-Z2)/(2*L); asp=np.arctan2(gy,-gx)
    Dc=((Z4+Z6)/2-Z5)/L**2; Ec=((Z2+Z8)/2-Z5)/L**2; Fc=(-Z1+Z3+Z7-Z9)/(4*L**2)
    Gg=gx; H=-gy; den=Gg**2+H**2; den2=np.where(den==0,np.nan,den)
    ok=np.all(np.vstack([Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9])!=NOD,axis=0)
    return dict(slope=np.degrees(np.arctan(np.hypot(gx,gy))),north=np.cos(asp),east=-np.sin(asp),
        planc=2*(Dc*H**2+Ec*Gg**2-Fc*Gg*H)/den2,profc=-2*(Dc*Gg**2+Ec*H**2+Fc*Gg*H)/den2,elev=Z5,ok=ok)
recs=[]; srcpt=[]
for geom,pid,ft,ev,ar in zip(R.geometry,R.pid,R.forest_type,R.Events,R.Area_ha):
    x0,y0,x1,y1=geom.bounds
    cs=np.arange(int(np.floor((x0-X0)/L)),int(np.floor((x1-X0)/L))+1)
    rs=np.arange(int(np.floor((Y0-y1)/L)),int(np.floor((Y0-y0)/L))+1)
    if len(cs)*len(rs)>200000: continue
    CC,RR=np.meshgrid(cs,rs); CC=CC.ravel(); RR=RR.ravel()
    px=X0+(CC+0.5)*L; py=Y0-(RR+0.5)*L
    pg=prep(geom); ins=np.fromiter((pg.contains(Point(a,b)) for a,b in zip(px,py)),bool,len(px))
    if ins.sum()==0:
        CC=np.array([int(np.floor((geom.centroid.x-X0)/L))]); RR=np.array([int(np.floor((Y0-geom.centroid.y)/L))]); ins=np.array([True])
    CC=CC[ins]; RR=RR[ins]
    m=(RR>=1)&(RR<A.shape[0]-1)&(CC>=1)&(CC<A.shape[1]-1)
    if m.sum()==0: continue
    CC=CC[m]; RR=RR[m]; t=terr(RR,CC)
    good=t['ok']&np.isfinite(t['planc'])&np.isfinite(t['profc'])
    if good.sum()==0: continue
    rec=dict(pid=pid,ft=ft,event=ev,area_ha=ar,n=int(good.sum()),
             **{k:t[k][good] for k in COVS})
    recs.append(rec)
    z=t['elev']; kk=int(np.argmax(np.where(t['ok'],z,-9e9)))
    srcpt.append((pid,ev,X0+(CC[kk]+0.5)*L,Y0-(RR[kk]+0.5)*L,int(RR[kk]),int(CC[kk])))
pickle.dump(recs,open(ROOT+'/verify/poly_cells.pkl','wb'))
SP=pd.DataFrame(srcpt,columns=['pid','event','x','y','row','col'])
log(f"   {len(recs)} polygons, {sum(r['n'] for r in recs)} cells")
fact('S6.polygon_cells',dict(polygons=len(recs),cells=int(sum(r['n'] for r in recs)),
     cells_per_polygon_median=float(np.median([r['n'] for r in recs]))),
 "20 m DEM cells whose centre falls inside each rainfall landslide polygon",
 "dem_20m.tif + annual shapefiles","A")
t=terr(SP.row.values,SP.col.values)
for k in COVS: SP[k]=t[k]
SP.to_csv(ROOT+'/verify/source_points.csv',index=False)
del A; gc.collect()

# ---------------------------------------------------------------- 6b.2 forest type at source point
log("6b.2 forest type at the highest DEM cell of each polygon")
GS=gpd.GeoDataFrame(SP,geometry=gpd.points_from_xy(SP.x,SP.y),crs=3826)
res=[]; info=pyogrio.read_info(F4); N=int(info['features'])
for s0 in range(0,N,40000):
    FT=pyogrio.read_dataframe(F4,columns=['TypeName'],skip_features=s0,max_features=40000,on_invalid='ignore')
    FT=FT[FT.geometry.notna()].set_crs(3826,allow_override=True); FT['geometry']=FT.geometry.buffer(0)
    j=gpd.sjoin(GS,FT[['TypeName','geometry']],how='inner',predicate='within')
    res.append(pd.DataFrame({'pid':j.pid.values,'TypeName':j.TypeName.values}))
    del FT,j; gc.collect()
SP=SP.merge(pd.concat(res).drop_duplicates('pid'),on='pid',how='left')
SP.to_csv(ROOT+'/verify/source_points.csv',index=False)
cnt=SP.TypeName.value_counts(dropna=False)
sa=SP[SP.TypeName.isin(K1)].rename(columns={'TypeName':'ft'})
r_src=inter(build(sa,CTRL,[BAMBOO],K1))
pid2={r['pid']:r for r in recs}
sub=CASE[CASE.lid.isin(pid2)]
hi=np.array([pid2[p]['elev'].max() for p in sub.lid]); cen=sub.elev.values
fact('S6.source_point',dict(bamboo_cases_at_source=int(cnt.get(BAMBOO,0)),
     broadleaf_cases_at_source=int(cnt.get(BROAD,0)),
     published_E5_source_area={'OR':1.0268,'n_case':209},reproduced=r_src,
     elev_gain_over_centroid_m=dict(mean=round(float(np.mean(hi-cen)),1),median=round(float(np.median(hi-cen)),1),
                                    p90=round(float(np.percentile(hi-cen,90)),1),
                                    share_identical_pct=round(float(100*np.mean(hi==cen)),1)),
     manuscript_states_60m=True),
 "forest type and terrain taken at the highest DEM cell inside each polygon (the rule the manuscript "
 "describes in section 2.3 but which the main analysis does not use)",
 "dem_20m.tif + forest4.shp","A")
log(f"   bamboo cases at source point {int(cnt.get(BAMBOO,0))}; OR {r_src['OR']}; elev gain mean {np.mean(hi-cen):.1f} m")

# ---------------------------------------------------------------- 6b.3 polygon representations
log("6b.3 polygon representation")
rep={}
rep['centroid']=inter(build(CASE,CTRL,[BAMBOO],K1))
def from_recs(sel):
    rows=[dict(ft=r['ft'],**sel(r)) for r in recs if r['ft'] in K1]
    A_=pd.DataFrame(rows); A_['case']=1
    B_=CTRL[CTRL.ft.isin(K1)][COVS+['ft']].copy(); B_['case']=0
    D=pd.concat([A_,B_],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    return D
rep['highest_cell']=inter(from_recs(lambda r:{v:float(r[v][int(np.argmax(r['elev']))]) for v in COVS}))
rep['steepest_cell']=inter(from_recs(lambda r:{v:float(r[v][int(np.argmax(r['slope']))]) for v in COVS}))
rep['lowest_cell']=inter(from_recs(lambda r:{v:float(r[v][int(np.argmin(r['elev']))]) for v in COVS}))
rep['median_cell']=inter(from_recs(lambda r:{v:float(np.median(r[v])) for v in COVS}))
rep['polygon_mean']=inter(from_recs(lambda r:{v:float(np.mean(r[v])) for v in COVS}))
def top20(r):
    thr=np.quantile(r['elev'],0.8); m=r['elev']>=thr
    if m.sum()==0: m=np.ones_like(r['elev'],bool)
    return {v:float(np.mean(r[v][m])) for v in COVS}
rep['source_area_top20pct']=inter(from_recs(top20))
allrows=[]
for r in recs:
    if r['ft'] not in K1: continue
    allrows.append(pd.DataFrame({v:r[v] for v in COVS}|{'ft':r['ft'],'w':1.0/r['n']}))
AC=pd.concat(allrows,ignore_index=True); AC['case']=1
BC=CTRL[CTRL.ft.isin(K1)][COVS+['ft']].copy(); BC['case']=0; BC['w']=1.0
import statsmodels.api as sm
def wfit(D):
    D=D.copy(); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    m=smf.glm(FORM,data=D,family=sm.families.Binomial(),freq_weights=D.w.values).fit()
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=float(2*(1-norm.cdf(abs(b/s)))),n_cells=int((D.case==1).sum()))
rep['all_cells_polygon_weight1']=wfit(pd.concat([AC,BC],ignore_index=True))
AC2=AC.copy(); AC2['w']=1.0
rep['all_cells_area_weighted']=wfit(pd.concat([AC2,BC],ignore_index=True))
rng=np.random.default_rng(2026); ors=[]
for _ in range(200):
    rows=[dict(ft=r['ft'],**{v:float(r[v][rng.integers(r['n'])]) for v in COVS}) for r in recs if r['ft'] in K1]
    A_=pd.DataFrame(rows); A_['case']=1
    D=pd.concat([A_,BC[COVS+['ft','case']]],ignore_index=True)
    D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    m=smf.logit(FORM,data=D).fit(disp=0); ors.append(math.exp(m.params['bamboo:slope']))
ors=np.array(ors)
rep['random_cell_200reps']=dict(OR_mean=round(float(ors.mean()),4),
    OR_range_mc=[round(float(np.percentile(ors,2.5)),4),round(float(np.percentile(ors,97.5)),4)],
    share_gt1=float((ors>1).mean()),percentile_of_published=float(100*(ors<1.0341).mean()),seed=2026)
slope_sd=dict(centroid=float(CASE[CASE.ft.isin(K1)].slope.std()),
              polygon_mean=float(np.std([np.mean(r['slope']) for r in recs if r['ft'] in K1])),
              within_polygon_mean_sd=float(np.mean([r['slope'].std() for r in recs if r['ft'] in K1 and r['n']>1])))
big=[r for r in recs if r['ft'] in K1 and r['n']>=5]
rows=[dict(ft=r['ft'],**{v:float(np.mean(r[v])) for v in COVS}) for r in big]
Am=pd.DataFrame(rows); Am['case']=1
Dm=pd.concat([Am,BC[COVS+['ft','case']]],ignore_index=True); Dm['bamboo']=(Dm.ft==BAMBOO).astype(int); Dm['slope2']=Dm.slope**2
rg=np.random.default_rng(7)
rows=[dict(ft=r['ft'],**{v:float(r[v][rg.integers(r['n'])]) for v in COVS}) for r in big]
Ar=pd.DataFrame(rows); Ar['case']=1
Dr=pd.concat([Ar,BC[COVS+['ft','case']]],ignore_index=True); Dr['bamboo']=(Dr.ft==BAMBOO).astype(int); Dr['slope2']=Dr.slope**2
rep['dilution_check_ge5cells']=dict(polygon_mean=inter(Dm),random_cell=inter(Dr),n_polygons=len(big))
rep['slope_sd']=slope_sd
fact('S6.polygon_representation',rep,
 "the interaction re-estimated with the landslide represented by different points or by all its cells",
 "dem_20m.tif + annual shapefiles + rebuilt control table","B",
 "random_cell uses seed 2026; all other rows are deterministic")
for k,v in rep.items():
    if isinstance(v,dict) and 'OR' in v: log(f"   {k:28s} OR {v['OR']} p {v['p']:.4g}")
log(f"   random cell (200 reps) mean {rep['random_cell_200reps']['OR_mean']}, {100*rep['random_cell_200reps']['share_gt1']:.0f}% above 1")
log(f"   dilution check (>=5 cells): mean {rep['dilution_check_ge5cells']['polygon_mean']['OR']} vs random {rep['dilution_check_ge5cells']['random_cell']['OR']}")
json.dump({'facts':OUT,'log':SRC['log']+LOG},open(ROOT+'/verify/verified_facts_s6b.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 6b part 1 written")
