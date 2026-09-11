# -*- coding: utf-8 -*-
"""Stage 6c: failure-type screens, misclassification, three-level model, per-event,
leave-one-event-out, crossover bootstrap, common support, frozen curve statistics."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,math,time,warnings,gc,glob,pickle
import numpy as np,pandas as pd,geopandas as gpd,pyogrio
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm,chi2
warnings.filterwarnings('ignore')
T0=time.time(); LOG=[]
SRC=json.load(open(ROOT+'/verify/verified_facts_s6b.json')); OUT=SRC['facts']
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
U=RAW; AUX=f"{U}/_stage_polygons/auxlayers"
CASE=pd.read_csv(ROOT+'/verify/cases_xy.csv'); CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')
POOL=pd.read_csv(ROOT+'/verify/newctrl_pool.csv')
recs=pickle.load(open(ROOT+'/verify/poly_cells.pkl','rb'))
def build(a,b,bam,keep):
    A=a[a.ft.isin(keep)][COVS+['ft']].copy(); A['case']=1
    B=b[b.ft.isin(keep)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['bamboo']=D.ft.isin(bam).astype(int); D['slope2']=D.slope**2
    return D
def inter(D):
    m=smf.logit(FORM,data=D).fit(disp=0); b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    return dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],
                p=float(2*(1-norm.cdf(abs(b/s)))),n_case=int(D.case.sum()),
                n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),n_ctrl=int((D.case==0).sum()))
def curve_stats(D):
    sp=smf.logit(SPL,data=D).fit(disp=0); di=sp.model.data.orig_exog.design_info
    med={c:float(np.median(D[c])) for c in ['north','east','planc','profc','elev']}
    row=lambda b: pd.DataFrame({'slope':GRID,'bamboo':b,**med})
    d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0)))
    lo=d_@sp.params.values; V=sp.cov_params().values
    se=np.sqrt(np.einsum('ij,jk,ik->i',d_,V,d_))
    s=np.sign(lo); cx=None
    for i in range(len(GRID)-1):
        if s[i]<0 and s[i+1]>=0: cx=float(GRID[i]-lo[i]*(GRID[i+1]-GRID[i])/(lo[i+1]-lo[i])); break
    hi=np.exp(lo+1.96*se); l=np.exp(lo-1.96*se)
    f=lambda a: None if len(a)==0 else [float(a.min()),float(a.max()),bool(np.allclose(np.diff(a),0.1,atol=1e-6))]
    at=lambda v: dict(OR=round(float(np.exp(lo[np.argmin(np.abs(GRID-v))])),3),
                      ci=[round(float(l[np.argmin(np.abs(GRID-v))]),3),round(float(hi[np.argmin(np.abs(GRID-v))]),3)])
    return dict(crossover_deg=round(cx,2) if cx else None,protection_band=f(GRID[hi<1]),
                worse_band=f(GRID[l>1]),at={str(v):at(v) for v in (20,25,30,35,40,42,44,46,50)})

# ------------------------------------------------- 6c.1 historical scars + aux screens
log("6c.1 historical scars 2004-2017 and failure-type layers")
hs=[]
for y in range(2004,2018):
    p=glob.glob(f"{U}/_stage_polygons/{y}/*.shp")
    if not p: continue
    g=pyogrio.read_dataframe(p[0],columns=[],on_invalid='ignore')
    g=g[g.geometry.notna()].set_crs(3826,allow_override=True); hs.append(g[['geometry']])
H=gpd.GeoDataFrame(pd.concat(hs,ignore_index=True),crs=3826); H['geometry']=H.geometry.buffer(0)
fact('S6.historic_inventory',dict(years='2004-2017',polygons=int(len(H))),
 "pre-study-period event inventory used to flag reactivated scars","annual shapefiles 2004-2017","A")
ds=pyogrio.read_dataframe(f"{AUX}/debrisstream1753_20260126_twd97.shp",columns=[],on_invalid='ignore').set_crs(3826,allow_override=True)
rd=pyogrio.read_dataframe(f"{AUX}/ROAD_國省道(含快速公路)_1150409.shp",columns=[],on_invalid='ignore').set_crs(3826,allow_override=True)
rv=pyogrio.read_dataframe(f"{AUX}/riverpoly.shp",columns=[],on_invalid='ignore').set_crs(3826,allow_override=True)
lp=pyogrio.read_dataframe(f"{AUX}/115年_94處警戒發布區潛勢範圍_TWD97_UTF8.shp",columns=[],on_invalid='ignore').set_crs(3826,allow_override=True)
def annotate(df):
    Gp=gpd.GeoDataFrame(df.copy(),geometry=gpd.points_from_xy(df.x,df.y),crs=3826)
    df=df.copy()
    df['old_scar']=gpd.sjoin(Gp,H,how='left',predicate='within').groupby(level=0).index_right.apply(lambda s:s.notna().any()).values
    for nm,lay in (('d_debris',ds),('d_road',rd),('d_river',rv)):
        df[nm]=gpd.sjoin_nearest(Gp[['geometry']],lay[['geometry']],how='left',distance_col='d').groupby(level=0).d.min().values
    df['in_largels']=gpd.sjoin(Gp,lp[['geometry']],how='left',predicate='within').groupby(level=0).index_right.apply(lambda s:s.notna().any()).values
    return df
CA=annotate(CASE); CO=annotate(CTRL); PO=annotate(POOL)
CA.to_csv(ROOT+'/verify/cases_annot.csv',index=False)
CO.to_csv(ROOT+'/verify/controls_annot.csv',index=False)
PO.to_csv(ROOT+'/verify/pool_annot.csv',index=False)
react=dict(cases_pct=round(float(100*CA.old_scar.mean()),2),
           cases_bamboo_pct=round(float(100*CA[CA.ft==BAMBOO].old_scar.mean()),2),
           cases_broadleaf_pct=round(float(100*CA[CA.ft==BROAD].old_scar.mean()),2),
           controls_pct=round(float(100*CO.old_scar.mean()),2),
           new_pool_pct=round(float(100*PO.old_scar.mean()),2))
fact('S6.reactivation_rates',react,
 "share of points falling inside a 2004-2017 mapped landslide polygon","annual shapefiles","A")
log(f"   reactivation: cases {react['cases_pct']}%, controls {react['controls_pct']}%, pool {react['new_pool_pct']}%")

# shape metrics
pid2={r['pid']:r for r in recs}
CA['relief']=CA.lid.map({k:float(np.ptp(v['elev'])) for k,v in pid2.items()})
G=gpd.read_file(ROOT+'/verify/all_polys.gpkg',layer='ls').set_index('pid')
mrr=G.geometry.minimum_rotated_rectangle()
def elong(poly):
    xs,ys=poly.exterior.coords.xy; p=np.c_[xs,ys][:-1]
    dd=np.diff(np.r_[p,p[:1]],axis=0); d=np.hypot(dd[:,0],dd[:,1])
    return max(d[0],d[1])/max(min(d[0],d[1]),1e-9)
CA['elong']=CA.lid.map({pid:elong(pp) for pid,pp in mrr.items()})
fact('S6.polygon_geometry',dict(area_ha=dict(median=float(CA.area_ha.median()),p95=float(CA.area_ha.quantile(.95)),
      pct_ge_10ha=round(float(100*(CA.area_ha>=10).mean()),2),pct_ge_1ha=round(float(100*(CA.area_ha>=1).mean()),1)),
      relief_m=dict(median=float(CA.relief.median()),p95=float(CA.relief.quantile(.95))),
      elongation=dict(median=float(CA.elong.median()),p95=float(CA.elong.quantile(.95)))),
 "geometry of the 7,454 rainfall landslide polygons in forest","annual shapefiles + dem_20m.tif","A")

log("6c.2 failure-type screens")
sc={}
c1=CA[CA.ft.isin(K1)]; o1=CO[CO.ft.isin(K1)]; p1=PO[PO.ft.isin(K1)]
sc['baseline']=inter(build(c1,o1,[BAMBOO],K1))
sc['exclude_reactivation']=inter(build(c1[~c1.old_scar],o1[~o1.old_scar],[BAMBOO],K1))
sc['exclude_debris_200m']=inter(build(c1[c1.d_debris>200],o1[o1.d_debris>200],[BAMBOO],K1))
sc['exclude_debris_500m']=inter(build(c1[c1.d_debris>500],o1[o1.d_debris>500],[BAMBOO],K1))
sc['exclude_road_200m']=inter(build(c1[c1.d_road>200],o1[o1.d_road>200],[BAMBOO],K1))
sc['exclude_river_200m']=inter(build(c1[c1.d_river>200],o1[o1.d_river>200],[BAMBOO],K1))
sc['exclude_large_ls_potential']=inter(build(c1[~c1.in_largels],o1[~o1.in_largels],[BAMBOO],K1))
sc['exclude_area_ge10ha']=inter(build(c1[c1.area_ha<10],o1,[BAMBOO],K1))
sc['exclude_area_ge1ha']=inter(build(c1[c1.area_ha<1],o1,[BAMBOO],K1))
sc['relief_lt50m']=inter(build(c1[c1.relief<50],o1,[BAMBOO],K1))
sc['relief_lt30m']=inter(build(c1[c1.relief<30],o1,[BAMBOO],K1))
sc['elongation_lt3']=inter(build(c1[c1.elong<3],o1,[BAMBOO],K1))
SHc=lambda d:(d.area_ha<1)&(~d.old_scar)&(d.d_debris>200)&(~d.in_largels)
SHp=lambda d:(~d.old_scar)&(d.d_debris>200)&(~d.in_largels)
ALLc=lambda d:SHc(d)&(d.d_road>200)&(d.d_river>200)
ALLp=lambda d:SHp(d)&(d.d_road>200)&(d.d_river>200)
sc['reliably_shallow_original_controls']=inter(build(c1[SHc(c1)],o1[SHp(o1)],[BAMBOO],K1))
sc['reliably_shallow_new_pool']=inter(build(c1[SHc(c1)],p1[SHp(p1)],[BAMBOO],K1))
sc['all_screens_original_controls']=inter(build(c1[ALLc(c1)],o1[ALLp(o1)],[BAMBOO],K1))
sc['all_screens_new_pool']=inter(build(c1[ALLc(c1)],p1[ALLp(p1)],[BAMBOO],K1))
c2=CA[CA.ft.isin(K2)]; p2=PO[PO.ft.isin(K2)]
sc['pooled_all_screens_new_pool']=inter(build(c2[ALLc(c2)],p2[ALLp(p2)],[BAMBOO,MIXED],K2))
sc['pooled_reliably_shallow_new_pool']=inter(build(c2[SHc(c2)],p2[SHp(p2)],[BAMBOO,MIXED],K2))
fact('S6.screens',dict(definitions=dict(
      reliably_shallow="area < 1 ha AND not on a 2004-2017 scar AND >200 m from a debris-flow torrent AND outside large-landslide potential areas",
      all_screens="reliably_shallow AND >200 m from a national/provincial road AND >200 m from a river polygon"),**sc),
 "the interaction re-estimated after removing the failure types listed in the failure-type screen",
 "rebuilt table + official hazard layers","A")
for k,v in sc.items(): log(f"   {k:38s} OR {v['OR']} p {v['p']:.4g} (bamboo cases {v['n_bamboo_case']})")
bt=c1[c1.ft==BAMBOO]
fact('S6.relief_truncation',dict(bamboo_cases_gt40deg=dict(all=int((bt.slope>40).sum()),
      relief_lt50=int(((bt.relief<50)&(bt.slope>40)).sum()),relief_lt30=int(((bt.relief<30)&(bt.slope>40)).sum()))),
 "how many bamboo cases above 40 degrees survive each relief restriction",
 "rebuilt table","A","explains why the relief-restricted subsets lose the steep-slope leverage")

log("6c.3 forest-type misclassification simulation")
D0=build(CA,CO,[BAMBOO],K1)
rng=np.random.default_rng(5); mis={}
for rate in (0.02,0.05,0.10,0.20):
    ors=[]
    for _ in range(300):
        b=D0.bamboo.values.copy(); fl=rng.random(len(b))<rate; b[fl]=1-b[fl]
        D=D0.copy(); D['bamboo']=b
        m=smf.logit(FORM,data=D).fit(disp=0); ors.append(math.exp(m.params['bamboo:slope']))
    ors=np.array(ors)
    mis[f'{int(rate*100)}pct_both_ways']=dict(OR_mean=round(float(ors.mean()),4),
        OR_range_mc=[round(float(np.percentile(ors,2.5)),4),round(float(np.percentile(ors,97.5)),4)],
        share_gt1=float((ors>1).mean()))
    log(f"   {int(rate*100):2d}% both ways: mean {ors.mean():.4f}  >1 in {100*(ors>1).mean():.0f}%")
for rate in (0.05,0.10,0.20):
    ors=[]
    for _ in range(200):
        b=D0.bamboo.values.copy(); idx=np.where(b==1)[0]
        b[idx[rng.random(len(idx))<rate]]=0
        D=D0.copy(); D['bamboo']=b
        m=smf.logit(FORM,data=D).fit(disp=0); ors.append(math.exp(m.params['bamboo:slope']))
    ors=np.array(ors)
    mis[f'{int(rate*100)}pct_bamboo_to_broadleaf']=dict(OR_mean=round(float(ors.mean()),4),share_gt1=float((ors>1).mean()))
fact('S6.misclassification',dict(seed=5,**mis),
 "non-differential and one-directional forest-type misclassification simulated on the analysis table",
 "rebuilt table","C","bounds the effect of the gap between the 2017 forest map and the 2018-2025 inventory")

log("6c.4 three-level model and the pooling test")
def three(ctrl):
    A=CA[CA.ft.isin(K2)][COVS+['ft']].copy(); A['case']=1
    B=ctrl[ctrl.ft.isin(K2)][COVS+['ft']].copy(); B['case']=0
    D=pd.concat([A,B],ignore_index=True); D['slope2']=D.slope**2
    D['ftc']=pd.Categorical(D.ft,categories=[BROAD,MIXED,BAMBOO])
    m=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+C(ftc)+C(ftc):slope",data=D).fit(disp=0)
    g=lambda k: dict(OR=round(math.exp(m.params[k]),4),
                     ci=[round(math.exp(m.params[k]-1.96*m.bse[k]),4),round(math.exp(m.params[k]+1.96*m.bse[k]),4)],
                     p=float(2*(1-norm.cdf(abs(m.params[k]/m.bse[k])))))
    return dict(mixed_main=g(f'C(ftc)[T.{MIXED}]'),bamboo_main=g(f'C(ftc)[T.{BAMBOO}]'),
                mixed_x_slope=g(f'C(ftc)[T.{MIXED}]:slope'),bamboo_x_slope=g(f'C(ftc)[T.{BAMBOO}]:slope')),m,D
t_old,_,_=three(CO); t_new,mfull,Dn=three(PO)
Dn['bam']=Dn.ft.isin([BAMBOO,MIXED]).astype(int)
mred=smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+C(ftc)+bam:slope",data=Dn).fit(disp=0)
lr=2*(mfull.llf-mred.llf); plr=float(1-chi2.cdf(lr,1))
fact('S6.three_level',dict(original_controls=t_old,new_pool=t_new,
     LR_test_common_slope_interaction=dict(chi2=round(float(lr),3),df=1,p=plr,
       conclusion="reject a common slope interaction" if plr<0.05 else "cannot reject a common slope interaction")),
 "forest type entered as three levels (broadleaf reference, bamboo-broadleaf mixed, pure bamboo), "
 "each with its own slope interaction; the likelihood-ratio test asks whether the two bamboo-containing "
 "types share one interaction",
 "rebuilt table + new control pool","A")
log(f"   mixed x slope {t_new['mixed_x_slope']['OR']} p={t_new['mixed_x_slope']['p']:.4g}; "
    f"bamboo x slope {t_new['bamboo_x_slope']['OR']} p={t_new['bamboo_x_slope']['p']:.4g}; LR p={plr:.4f}")

log("6c.5 pooled definition, per-event, leave-one-event-out")
pooled=dict(original_controls=inter(build(CA,CO,[BAMBOO,MIXED],K2)),
            new_pool=inter(build(CA,PO,[BAMBOO,MIXED],K2)))
drop=['凱米颱風','0728豪雨']
pooled['drop2events_pure_original']=inter(build(CA[(CA.ft.isin(K1))&(~CA.event.isin(drop))],CO,[BAMBOO],K1))
pooled['drop2events_pure_newpool']=inter(build(CA[(CA.ft.isin(K1))&(~CA.event.isin(drop))],PO,[BAMBOO],K1))
pooled['drop2events_pooled_newpool']=inter(build(CA[(CA.ft.isin(K2))&(~CA.event.isin(drop))],PO,[BAMBOO,MIXED],K2))
fact('S6.bamboo_definition',pooled,"pure bamboo versus bamboo-containing stands, and the effect of "
 "removing the two events that carry most bamboo cases","rebuilt table","A")
for k,v in pooled.items(): log(f"   {k:32s} OR {v['OR']} p {v['p']:.4g} (bamboo cases {v['n_bamboo_case']})")
FP={k:set(map(tuple,v)) for k,v in json.load(open(ROOT+'/verify/event_cells_5km.json')).items()}
cell=lambda df,c=5000.: list(zip(np.floor(df.x/c).astype(int),np.floor(df.y/c).astype(int)))
PO2=PO.copy(); PO2['cell']=cell(PO2)
def per_event(keep,bam,minb):
    rows=[]
    for ev,cs in CA[CA.ft.isin(keep)].groupby('event'):
        nb=int(cs.ft.isin(bam).sum())
        if nb<minb: continue
        sel=PO2[PO2.ft.isin(keep)&PO2.cell.isin(FP.get(ev,set()))]
        if len(sel)<200: continue
        r=inter(build(cs,sel,bam,keep)); r['event']=ev; r['n_bamboo_case']=nb; rows.append(r)
    return sorted(rows,key=lambda r:-r['n_bamboo_case'])
pe1=per_event(K1,[BAMBOO],15); pe2=per_event(K2,[BAMBOO,MIXED],15)
fact('S6.per_event',dict(pure=pe1,pooled=pe2,
     rule="events with at least 15 bamboo cases and at least 200 controls inside the event footprint"),
 "event-specific interaction estimates","rebuilt table + new control pool","A")
log(f"   per-event (pure): {[(r['event'],r['OR'],round(r['p'],4)) for r in pe1]}")
Dloeo=build(CA,CO,[BAMBOO],K1)
evv=np.concatenate([CA[CA.ft.isin(K1)].event.values,np.array(['CTRL']*int((CO.ft.isin(K1)).sum()))])
Dloeo['ev']=evv
rows=[]
for ev in sorted(CA[CA.ft.isin(K1)].event.unique()):
    d=Dloeo[Dloeo.ev!=ev]
    if int(((d.case==1)&(d.bamboo==1)).sum())<10: continue
    m=smf.logit(FORM,data=d).fit(disp=0); b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    rows.append(dict(dropped=ev,n_dropped=int((CA[CA.ft.isin(K1)].event==ev).sum()),
                     OR=round(math.exp(b),4),p=float(2*(1-norm.cdf(abs(b/s))))))
L=pd.DataFrame(rows)
fact('S6.leave_one_event_out',dict(n_refits=len(L),OR_min=float(L.OR.min()),OR_max=float(L.OR.max()),
     all_above_1=bool((L.OR>1).all()),n_p_lt_05=int((L.p<0.05).sum()),
     worst=L.sort_values('p').tail(1).to_dict('records')[0],rows=L.to_dict('records')),
 "the interaction refitted 39 times, each time removing all cases of one event",
 "rebuilt table","A")
log(f"   LOEO range {L.OR.min():.4f}-{L.OR.max():.4f}, all>1 {bool((L.OR>1).all())}, p<0.05 in {int((L.p<0.05).sum())}/{len(L)}")

log("6c.6 crossover bootstrap, common support, frozen curve statistics")
D1=build(CA,CO,[BAMBOO],K1)
cs_ev=CA[CA.ft.isin(K1)].event.values
rng=np.random.default_rng(20260902)
ca_idx=np.where(D1.case.values==1)[0]; co_idx=np.where(D1.case.values==0)[0]
events=np.unique(cs_ev); by_ev={e:ca_idx[cs_ev==e] for e in events}
cx=[]
for _ in range(600):
    pick=rng.choice(events,len(events),replace=True)
    ci=np.concatenate([by_ev[e] for e in pick])
    oi=rng.choice(co_idx,len(co_idx),replace=True)
    d=D1.iloc[np.concatenate([ci,oi])]
    if int(((d.case==1)&(d.bamboo==1)).sum())<20: continue
    try:
        sp=smf.logit(SPL,data=d).fit(disp=0); di=sp.model.data.orig_exog.design_info
        med={c:float(np.median(d[c])) for c in ['north','east','planc','profc','elev']}
        row=lambda b: pd.DataFrame({'slope':GRID,'bamboo':b,**med})
        lo=(np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0))))@sp.params.values
        s=np.sign(lo); v=None
        for i in range(len(GRID)-1):
            if s[i]<0 and s[i+1]>=0: v=float(GRID[i]-lo[i]*(GRID[i+1]-GRID[i])/(lo[i+1]-lo[i])); break
        if v is not None: cx.append(v)
    except Exception: pass
cx=np.array(cx)
bp95=float(CTRL[CTRL.ft==BAMBOO].slope.quantile(0.95))
fact('S6.crossover_bootstrap',dict(seed=20260902,draws=600,valid=int(len(cx)),
     median=round(float(np.median(cx)),2),ci95=[round(float(np.percentile(cx,2.5)),2),round(float(np.percentile(cx,97.5)),2)],
     ci50=[round(float(np.percentile(cx,25)),2),round(float(np.percentile(cx,75)),2)],
     share_below_bamboo_p95=round(float(100*(cx<bp95).mean()),1),bamboo_control_p95_deg=round(bp95,2)),
 "cluster bootstrap of the crossover: events resampled with replacement (all their cases), controls "
 "resampled with replacement; crossover interpolated on the frozen grid",
 "rebuilt table","C")
log(f"   crossover bootstrap: median {np.median(cx):.2f}, 95% CI [{np.percentile(cx,2.5):.2f},{np.percentile(cx,97.5):.2f}], valid {len(cx)}/600")
cur={}
cur['pure_original_controls']=curve_stats(build(CA,CO,[BAMBOO],K1))
cur['pure_new_pool']=curve_stats(build(CA,PO,[BAMBOO],K1))
cur['pooled_original_controls']=curve_stats(build(CA,CO,[BAMBOO,MIXED],K2))
cur['pooled_new_pool']=curve_stats(build(CA,PO,[BAMBOO,MIXED],K2))
fact('S6.curve_statistics',dict(grid="5.0 to 60.0 degrees, step 0.1",**cur),
 "crossover, significant-protection band and pointwise odds ratios under the frozen slope grid",
 "rebuilt table","A")
for k,v in cur.items(): log(f"   {k:28s} crossover {v['crossover_deg']} protection {v['protection_band']}")
bins=[0,10,20,30,40,45,50,55,60,90]
def support(ctrl,bam):
    b=ctrl[ctrl.ft.isin(bam)]; r=ctrl[ctrl.ft==BROAD]
    t=pd.DataFrame({'bamboo':pd.cut(b.slope,bins).value_counts().sort_index(),
                    'broadleaf':pd.cut(r.slope,bins).value_counts().sort_index()})
    t['bamboo_share_pct']=(100*t.bamboo/(t.bamboo+t.broadleaf)).round(2)
    return {str(k):dict(bamboo=int(v.bamboo),broadleaf=int(v.broadleaf),bamboo_share_pct=float(v.bamboo_share_pct))
            for k,v in t.iterrows()}
fact('S6.common_support',dict(original_controls=support(CTRL,[BAMBOO]),new_pool=support(POOL,[BAMBOO]),
     bamboo_control_slope=dict(original=dict(median=round(float(CTRL[CTRL.ft==BAMBOO].slope.median()),1),
        p95=round(float(CTRL[CTRL.ft==BAMBOO].slope.quantile(.95)),1),max=round(float(CTRL[CTRL.ft==BAMBOO].slope.max()),1)),
        new_pool=dict(median=round(float(POOL[POOL.ft==BAMBOO].slope.median()),1),
        p95=round(float(POOL[POOL.ft==BAMBOO].slope.quantile(.95)),1),max=round(float(POOL[POOL.ft==BAMBOO].slope.max()),1))),
     bamboo_cases_gt50deg=int((CA[CA.ft==BAMBOO].slope>50).sum()),bamboo_cases_total=int((CA.ft==BAMBOO).sum())),
 "slope distribution of bamboo and broadleaf control points, by 10-degree bins",
 "rebuilt table + new control pool","A")
ev=CA[CA.ft.isin(K1)].groupby('event').agg(n_case=('lid','size'),n_bamboo=('ft',lambda s:(s==BAMBOO).sum()))
ev=ev.sort_values('n_case',ascending=False)
fact('S6.event_structure',dict(n_events=int(len(ev)),events_lt30_cases=int((ev.n_case<30).sum()),
     events_zero_bamboo=int((ev.n_bamboo==0).sum()),
     top2_bamboo_share_pct=round(float(100*ev.n_bamboo.nlargest(2).sum()/ev.n_bamboo.sum()),1),
     per_event={k:dict(n_case=int(v.n_case),n_bamboo=int(v.n_bamboo)) for k,v in ev.iterrows()}),
 "cases per rainfall event in the bamboo-versus-broadleaf subset","rebuilt table","A")
json.dump({'facts':OUT,'log':SRC['log']+LOG},open(ROOT+'/verify/verified_facts.json','w'),
          ensure_ascii=False,indent=1,default=str)
log("stage 6c written -> verified_facts.json")
