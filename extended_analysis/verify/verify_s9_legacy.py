# -*- coding: utf-8 -*-
"""Stage 9 - recovery of the covariate extensions and inference procedures of the earlier version of the analysis that were produced by the original pipeline (legacy/程式碼/analysis_all.py) but were
not in the v51 verification chain:
  9a  topographic wetness index (240 m block-mean DEM, D8 accumulation)   -> exactly as analysis_all.py
  9b  OpenLandMap topsoil (0 cm) organic carbon, pH, clay                  -> Zenodo v0.2 layers, nearest 250 m cell
  9c  USGS ELU surface lithology, Hengl (2018) 250 m regrid                 -> Zenodo 10.5281/zenodo.1464846
  9d  interaction re-estimated with TWI / soil / lithology; within siliciclastic sedimentary only
  9e  event random-intercept GLMM (variational Bayes), controls assigned to the nearest case's event
  9f  20 km spatial-block 5-fold cross-validation: training-fold interaction, held-out AUC,
      and (new) the interaction estimated inside each held-out block
Outputs: verified_facts_s9.json, covariates_s9.csv (per point)."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, time, warnings, numpy as np, pandas as pd, rasterio
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import norm
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings('ignore'); t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}",flush=True)
OUT={}
def fact(fid,value,definition,source,cls,note=""): OUT[fid]=dict(value=value,definition=definition,source=source,cls=cls,note=note)
S0=json.load(open(ROOT+'/verify/verified_facts.json'))['facts']['S0.sources']['value']
DEM=SRC(S0['dem_20m.tif']['path'])
CASE=pd.read_csv(ROOT+'/verify/cases_annot.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_annot.csv')
COVS=['slope','north','east','planc','profc','elev']; K1=['竹林','闊葉樹林型']
# ------------------------------------------------------------------ 9a TWI (verbatim logic of analysis_all.py)
ds=rasterio.open(DEM); A=ds.read(1); ND=int(ds.nodata); t=ds.transform; CS=t.a; ox=t.c+CS/2.0; oy=t.f+t.e/2.0
f=12; H0,W0=A.shape; Hn,Wn=H0//f,W0//f
Af=A[:Hn*f,:Wn*f].astype(np.float32); Af[A[:Hn*f,:Wn*f]==ND]=np.nan
dem=np.nanmean(Af.reshape(Hn,f,Wn,f),axis=(1,3)); del Af, A
cell=CS*f; H,W=dem.shape; demp=np.pad(dem,1,constant_values=np.nan)
dirs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
best=np.full((H,W),-1.0); down=np.full((H,W),-1,dtype=np.int64)
for dr,dc in dirs:
    nb=demp[1+dr:1+dr+H,1+dc:1+dc+W]; dist=cell*(2**0.5 if dr and dc else 1)
    drop=(dem-nb)/dist; m=(drop>best)&np.isfinite(drop); best[m]=drop[m]
    rr=(np.arange(H)[:,None]+dr); cc=(np.arange(W)[None,:]+dc)
    fl=np.where((rr>=0)&(rr<H)&(cc>=0)&(cc<W),(rr*W+cc),-1); down[m]=fl[m]
down[best<=0]=-1
acc=np.where(np.isfinite(dem),1.0,0.0).ravel(); dflat=down.ravel(); ev=dem.ravel()
order=np.argsort(np.where(np.isfinite(ev),ev,-1e18))[::-1]; order=order[np.isfinite(ev[order])]
for c in order:
    d=dflat[c]
    if d>=0: acc[d]+=acc[c]
acc=acc.reshape(H,W)
gy,gx=np.gradient(dem,cell); slp=np.arctan(np.sqrt(gx**2+gy**2)); twi=np.log((acc*cell)/(np.tan(slp)+1e-3))
def sample_twi(xs,ys):
    col=np.clip(((np.asarray(xs)-ox)/cell).astype(int),0,W-1); row=np.clip(((oy-np.asarray(ys))/cell).astype(int),0,H-1)
    v=twi[row,col]; return np.where(np.isfinite(v),v,np.nan)
for D in (CASE,CTRL): D['twi']=sample_twi(D.x.values,D.y.values)
log(f"TWI grid {H}x{W} at {cell:.0f} m; cases finite {np.isfinite(CASE.twi).mean():.4f}, controls {np.isfinite(CTRL.twi).mean():.4f}")
fact('S9.twi_definition',dict(grid_m=cell,rows=H,cols=W,method='block-mean of 20 m DEM (12x12), D8 steepest descent, TWI=ln(a*cell/(tan(beta)+1e-3))',
     dem_origin_pixel_centre=[ox,oy],finite_share_cases=round(float(np.isfinite(CASE.twi).mean()),4),finite_share_controls=round(float(np.isfinite(CTRL.twi).mean()),4)),
     "topographic wetness index exactly as in the original pipeline (analysis_all.py)","dem_20m.tif","A")
# ------------------------------------------------------------------ 9b/9c soil + lithology (EPSG:4326 rasters, nearest cell)
tr=Transformer.from_crs(3826,4326,always_xy=True)
def sample_ll(path,xs,ys,nodata=None):
    lon,lat=tr.transform(np.asarray(xs,float),np.asarray(ys,float))
    with rasterio.open(path) as src:
        a=src.read(1); T=src.transform
        col=np.round((lon-(T.c+T.a/2))/T.a).astype(int); row=np.round((lat-(T.f+T.e/2))/T.e).astype(int)
        ok=(row>=0)&(row<a.shape[0])&(col>=0)&(col<a.shape[1])
        v=np.full(len(lon),np.nan); v[ok]=a[row[ok],col[ok]].astype(float)
        nd=src.nodata if nodata is None else nodata
        if nd is not None: v[v==nd]=np.nan
    return v
import os
soil_ok=all(os.path.exists(ROOT+f'/data/layers/soil_{k}_taiwan_250m.tif') for k in ['soc','ph','clay'])
for D in (CASE,CTRL):
    D['litho']=sample_ll(ROOT+'/data/layers/litho_taiwan_250m.tif',D.x.values,D.y.values,nodata=255)
    if soil_ok:
        for k in ['soc','ph','clay']: D[k]=sample_ll(ROOT+f'/data/layers/soil_{k}_taiwan_250m.tif',D.x.values,D.y.values)
LITH={1:'acid plutonics',2:'acid volcanic',3:'basic plutonics',4:'basic volcanics',5:'carbonate sedimentary rock',6:'evaporite',7:'ice and glaciers',
      8:'intermediate plutonics',9:'intermediate volcanics',10:'metamorphics',11:'mixed sedimentary rock',12:'pyroclastics',13:'siliciclastic sedimentary',
      14:'unconsolidated sediment',15:'undefined'}
for D in (CASE,CTRL): D['litho_name']=D.litho.map(LITH)
log(f"lithology classes, cases: {CASE.litho_name.value_counts().to_dict()}")
fact('S9.lithology_classes',dict(source='Hengl (2018) 10.5281/zenodo.1464846, dtm_lithology_usgs.ecotapestry_c_250m, EPSG:4326 0.002083 deg, nearest cell',
     cases=CASE.litho_name.value_counts().to_dict(),controls=CTRL.litho_name.value_counts().to_dict()),
     "surface lithology class at each analysis point","Zenodo 1464846","A")
if soil_ok:
    fact('S9.soil_layers',dict(source='OpenLandMap v0.2 (Zenodo 2525553 SOC x5 g/kg, 2525664 pH x10, 2525663 clay %), depth 0 cm, nearest 250 m cell',
         cases_finite={k:round(float(np.isfinite(CASE[k]).mean()),4) for k in ['soc','ph','clay']},
         medians_cases={k:float(np.nanmedian(CASE[k])) for k in ['soc','ph','clay']}),"topsoil properties at each analysis point","Zenodo OpenLandMap","A")
pd.concat([CASE.assign(kind='case'),CTRL.assign(kind='ctrl')],ignore_index=True).to_csv(ROOT+'/verify/covariates_s9.csv',index=False)
# ------------------------------------------------------------------ 9d interaction with the extra covariates
def build(extra_cols=()):
    a=CASE[CASE.ft.isin(K1)].copy(); a['case']=1; b=CTRL[CTRL.ft.isin(K1)].copy(); b['case']=0
    D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft=='竹林').astype(int); D['slope2']=D.slope**2
    D=D.dropna(subset=COVS+list(extra_cols)).reset_index(drop=True); return D
def inter(D,extra=""):
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev"+extra+"+bamboo+bamboo:slope",data=D).fit(disp=0,maxiter=200)
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']; b0=m.params['bamboo']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5),
                linear_crossover_deg=round(float(-b0/b),1) if b>0 else None,n=int(len(D)),n_case=int(D.case.sum()),
                n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum())),m
E=json.load(open(ROOT+'/../expected_outputs/results.json'))['E5_interaction_specs']['specs']
specs={}
specs['terrain'],_=inter(build())
specs['plus_twi'],_=inter(build(['twi']),"+twi")
if soil_ok:
    specs['plus_soil'],_=inter(build(['soc','ph','clay']),"+soc+ph+clay")
    specs['plus_twi_soil'],_=inter(build(['twi','soc','ph','clay']),"+twi+soc+ph+clay")
Dl=build(['litho']); Dl['litho_c']=Dl.litho_name.fillna('undefined')
# collapse rare classes (<50 points) into 'other' so the categorical fit is stable
vc=Dl.litho_c.value_counts(); Dl['litho_c']=np.where(Dl.litho_c.map(vc)>=50,Dl.litho_c,'other')
specs['plus_lithology'],ml=inter(Dl,"+C(litho_c, Treatment(reference='siliciclastic sedimentary'))")
lith_rr={}
for k in ml.params.index:
    if k.startswith('C(litho_c'):
        nm=k.split('[T.')[1].rstrip(']'); lith_rr[nm]=dict(OR=round(math.exp(ml.params[k]),3),p=round(float(ml.pvalues[k]),4))
specs['within_siliciclastic'],_=inter(Dl[Dl.litho_c=='siliciclastic sedimentary'])
for k in ['metamorphics','mixed sedimentary rock','unconsolidated sediment']:
    if (Dl.litho_c==k).sum()>300 and ((Dl.litho_c==k)&(Dl.bamboo==1)&(Dl.case==1)).sum()>=10:
        specs[f'within_{k}'],_=inter(Dl[Dl.litho_c==k])
comp={k:dict(reproduced=specs[k]['OR'],published=E[k]['OR']) for k in ['terrain','plus_twi','plus_soil','plus_twi_soil'] if k in specs}
fact('S9.covariate_specs',dict(specs=specs,comparison_with_results_json=comp,lithology_class_odds_vs_siliciclastic=lith_rr,
     manuscript_lithology_values=dict(interaction_plus_lithology=1.0363,within_siliciclastic=1.0345,note='earlier values, script not recovered; compare with the reproduced values here')),
     "interaction re-estimated with TWI, soil and lithology in the model (original controls)","dem_20m.tif + Zenodo layers","A" if soil_ok else "B")
log(f"covariate specs: { {k:v['OR'] for k,v in specs.items()} }")
# ------------------------------------------------------------------ 9e GLMM event random intercept (VB), exactly as analysis_all.py
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
dg=build(); cas=dg[dg.case==1]; ctl=dg[dg.case==0]
tree=cKDTree(cas[['x','y']].values); _,idx=tree.query(ctl[['x','y']].values,k=1)
dg['event']=dg['event'].astype(object); dg.loc[ctl.index,'event']=cas.iloc[idx]['event'].values
sdmap={}
for c in ['slope','slope2','north','east','planc','profc','elev']:
    mu,sdv=dg[c].mean(),dg[c].std(); dg[c+'_z']=(dg[c]-mu)/sdv; sdmap[c]=sdv
gr=BinomialBayesMixedGLM.from_formula("case ~ slope_z+slope2_z+north_z+east_z+planc_z+profc_z+elev_z+bamboo+bamboo:slope_z",{"ev":"0+C(event)"},data=dg).fit_vb(verbose=False)
names=list(gr.model.exog_names); j=names.index('bamboo:slope_z')
pm=float(gr.fe_mean[j])/sdmap['slope']; psd=float(gr.fe_sd[j])/sdmap['slope']; re_sd=float(math.exp(np.atleast_1d(gr.vcp_mean)[0]))
glmm=dict(n_events=int(dg.event.nunique()),event_re_sd=round(re_sd,3),interaction_OR=round(math.exp(pm),4),
          ci=[round(math.exp(pm-1.96*psd),4),round(math.exp(pm+1.96*psd),4)],method='statsmodels BinomialBayesMixedGLM, variational Bayes (fit_vb), z-scored predictors, back-transformed per degree',
          published=json.load(open(ROOT+'/../expected_outputs/results.json'))['GLMM_event_re'])
fact('S9.glmm_event_random_intercept',glmm,"event random-intercept mixed logistic model, controls assigned to the event of the nearest case","cases_annot/controls_annot","A")
log(f"GLMM {glmm['interaction_OR']} {glmm['ci']} re_sd {glmm['event_re_sd']}")
# ------------------------------------------------------------------ 9f spatial block CV (train-fold OR, held-out AUC, within-block OR)
d7=build(); BLK=20000.0
d7['bx']=np.floor((d7.x-d7.x.min())/BLK).astype(int); d7['by']=np.floor((d7.y-d7.y.min())/BLK).astype(int)
blocks=d7.groupby(['bx','by']).ngroup(); nfold=5
ub=pd.Series(blocks.unique()); fold_of={b:i%nfold for i,b in enumerate(ub.sample(frac=1,random_state=1))}
d7['fold']=blocks.map(fold_of)
FRM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    order=np.argsort(np.concatenate([pos,neg])); ranks=np.empty_like(order,float); ranks[order]=np.arange(1,len(order)+1)
    rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
folds=[]
for k in range(nfold):
    trn=d7[d7.fold!=k]; te=d7[d7.fold==k]
    m=smf.logit(FRM,data=trn).fit(disp=0); pr=m.predict(te)
    row=dict(fold=k,n_test=int(len(te)),n_test_bamboo_case=int(((te.bamboo==1)&(te.case==1)).sum()),
             train_interaction_OR=round(math.exp(m.params['bamboo:slope']),4),train_p=round(float(m.pvalues['bamboo:slope']),5),held_out_AUC=round(auc(te.case.values,pr.values),3))
    try:
        mt=smf.logit(FRM,data=te).fit(disp=0,maxiter=200); bt,st=mt.params['bamboo:slope'],mt.bse['bamboo:slope']
        row.update(within_block_interaction_OR=round(math.exp(bt),4),within_block_ci=[round(math.exp(bt-1.96*st),4),round(math.exp(bt+1.96*st),4)],within_block_p=round(float(2*(1-norm.cdf(abs(bt/st)))),4))
    except Exception as e:
        row.update(within_block_interaction_OR=None,within_block_error=str(e)[:80])
    folds.append(row)
E7=json.load(open(ROOT+'/../expected_outputs/results.json'))['E7_spatial_cv']
fact('S9.spatial_block_cv',dict(block_km=20,nfold=5,fold_seed=1,folds=folds,mean_held_out_AUC=round(float(np.mean([f['held_out_AUC'] for f in folds])),3),
     published_folds=E7['folds'],
     clarification='In the earlier version the per-fold interaction odds ratio was estimated on the four training folds (the held-out block contributed only the AUC). The within-block estimates are new.'),
     "20 km spatial-block five-fold cross-validation reproduced, plus the interaction estimated inside each held-out block","cases_annot/controls_annot","A")
log(f"CV folds: {[(f['fold'],f['train_interaction_OR'],f['held_out_AUC'],f['within_block_interaction_OR']) for f in folds]}")
json.dump(dict(facts=OUT),open(ROOT+'/verify/verified_facts_s9.json','w'),ensure_ascii=False,indent=1)
log("done")
