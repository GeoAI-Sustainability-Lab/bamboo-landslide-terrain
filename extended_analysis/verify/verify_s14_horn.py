# -*- coding: utf-8 -*-
"""Stage 14 - slope algorithm sensitivity. The stored terrain uses the Zevenbergen-Thorne (1987) four-neighbour slope.
Here the Horn (1981) eight-neighbour slope is recomputed from the same 20 m DEM at every analysis point (cases and
controls), the differences are summarised over all points of the bamboo-versus-broadleaf sample, and the main model of
Eq. (3) is refitted with the Horn slope in the linear, quadratic and interaction terms. Deterministic (class A).
  S14.horn_vs_zt_all_points     median and maximum absolute difference, share differing by more than 2 degrees
  S14.horn_slope_refit          interaction odds ratio per degree with Horn slopes (main design, island-wide controls)
  S14.horn_slope_refit_spline   supported slope range and the slope where the odds ratio equals 1, Horn slopes
Output: verified_facts_s14.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, numpy as np, pandas as pd, rasterio, statsmodels.formula.api as smf
from scipy.stats import norm
from patsy import dmatrix
V=ROOT+'/verify/'
U=RAW
DEM=f"{U}/Dataset/01_SOURCE/Terrain_Canopy/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif"
COVS=['slope','north','east','planc','profc','elev']
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
SPL ="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
GRID=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
BAMBOO='竹林'; BROAD='闊葉樹林型'
CASE=pd.read_csv(V+'cases_xy.csv'); CTRL=pd.read_csv(V+'controls_xy.csv')
src=rasterio.open(DEM); Tr=src.transform; L=20.0; NOD=src.nodata; A=src.read(1).astype(np.float64)
def rc_from_xy(x,y):
    col=np.round((np.asarray(x)-Tr.c)/Tr.a-0.5).astype(int); row=np.round((np.asarray(y)-Tr.f)/Tr.e-0.5).astype(int); return row,col
def slopes(row,col):
    Z1=A[row-1,col-1]; Z2=A[row-1,col]; Z3=A[row-1,col+1]; Z4=A[row,col-1]; Z6=A[row,col+1]; Z7=A[row+1,col-1]; Z8=A[row+1,col]; Z9=A[row+1,col+1]
    zt=np.degrees(np.arctan(np.hypot((Z6-Z4)/(2*L),(Z8-Z2)/(2*L))))
    horn=np.degrees(np.arctan(np.hypot(((Z3+2*Z6+Z9)-(Z1+2*Z4+Z7))/(8*L),((Z7+2*Z8+Z9)-(Z1+2*Z2+Z3))/(8*L))))
    return zt,horn
out={}
def sample(df):
    d=df[df.ft.isin([BAMBOO,BROAD])].copy(); r,c=rc_from_xy(d.x.values,d.y.values); zt,h=slopes(r,c)
    assert np.nanmax(np.abs(zt-d.slope.values))<1e-6, 'stored slope is not the Zevenbergen-Thorne recomputation'
    d['slope_horn']=h; return d
a=sample(CASE); a['case']=1; b=sample(CTRL); b['case']=0
D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft==BAMBOO).astype(int)
assert len(D)==15280 and D.case.sum()==5915 and ((D.case==1)&(D.bamboo==1)).sum()==171
hd=np.abs(D.slope_horn-D.slope)
out['S14.horn_vs_zt_all_points']={'value':dict(n_points=int(len(D)),median_deg=round(float(np.median(hd)),3),p90_deg=round(float(np.percentile(hd,90)),2),max_deg=round(float(hd.max()),2),
    pct_gt_2deg=round(float(100*np.mean(hd>2)),1),median_deg_cases=round(float(np.median(hd[D.case==1])),3),pct_gt_2deg_cases=round(float(100*np.mean(hd[D.case==1]>2)),1),
    horn_minus_zt_mean_deg=round(float((D.slope_horn-D.slope).mean()),3)),
    'definition':'absolute difference between the Horn eight-neighbour slope and the Zevenbergen-Thorne four-neighbour slope at the 15,280 analysis points (5,915 cases, 9,365 controls) of the bamboo-versus-broadleaf sample',
    'source':'dem_20m.tif recomputation at cases_xy.csv and controls_xy.csv','cls':'A'}
def fit(D,col):
    E=D.copy(); E['slope']=E[col]; E['slope2']=E.slope**2
    m=smf.logit(FORM,data=E).fit(disp=0); bb,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(bb),4),ci=[round(math.exp(bb-1.96*s),4),round(math.exp(bb+1.96*s),4)],p=float(2*(1-norm.cdf(abs(bb/s)))),n=int(len(E)),n_case=int(E.case.sum()),n_bamboo_case=int(((E.case==1)&(E.bamboo==1)).sum()))
zt_fit=fit(D,'slope'); horn_fit=fit(D,'slope_horn')
assert zt_fit['OR']==1.0341 and zt_fit['ci']==[1.0123,1.0565], zt_fit
out['S14.horn_slope_refit']={'value':dict(zevenbergen_thorne=zt_fit,horn=horn_fit),
    'definition':'interaction odds ratio per degree of slope from Eq. (3) with the slope covariate (linear, quadratic and interaction terms) computed by the Zevenbergen-Thorne method (main analysis) and by the Horn method',
    'source':'same sample as S5.headline_interaction; Horn slopes from dem_20m.tif','cls':'A'}
def curve(D,col):
    E=D.copy(); E['slope']=E[col]
    sp=smf.logit(SPL,data=E).fit(disp=0); di=sp.model.data.orig_exog.design_info
    med={c:float(np.median(E[c])) for c in ['north','east','planc','profc','elev']}
    row=lambda bam: pd.DataFrame({'slope':GRID,'bamboo':bam,**med})
    d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0))); lo=d_@sp.params.values
    Vc=sp.cov_params().values; se=np.sqrt(np.einsum('ij,jk,ik->i',d_,Vc,d_)); hi=lo+1.96*se
    below=GRID[hi<0]
    # first crossing of the curve through OR = 1 (linear interpolation on the grid)
    cross=None
    for i in range(1,len(GRID)):
        if lo[i-1]<0<=lo[i]: cross=round(float(GRID[i-1]+(GRID[i]-GRID[i-1])*(-lo[i-1])/(lo[i]-lo[i-1])),2); break
    return dict(band_upper_ci_below_1=[float(below.min()),float(below.max())] if len(below) else None,crossing_deg=cross)
zt_c=curve(D,'slope'); horn_c=curve(D,'slope_horn')
assert zt_c['band_upper_ci_below_1']==[9.9,39.5] and abs(zt_c['crossing_deg']-44.06)<0.011, zt_c
out['S14.horn_slope_refit_spline']={'value':dict(zevenbergen_thorne=zt_c,horn=horn_c),
    'definition':'cubic B-spline (df = 4) bamboo-versus-broadleaf curve of Eq. (4) with Zevenbergen-Thorne and with Horn slopes: slope range over which the pointwise upper 95% limit is below 1 (grid 5-60 degrees, step 0.1) and the slope where the curve reaches 1',
    'source':'same sample; model-based covariance','cls':'A'}
json.dump({'facts':out,'log':'verify_s14_horn.py'},open(V+'verified_facts_s14.json','w'),ensure_ascii=False,indent=1)
for k,v in out.items(): print(k, json.dumps(v['value'],ensure_ascii=False)); print()
