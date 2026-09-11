# -*- coding: utf-8 -*-
"""Build Table S1 and Figure S1 directly from verified_facts.json so they can never
drift from the register."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json,numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
F=json.load(open('verified_facts.json'))['facts']
S9=json.load(open('verified_facts_s9.json'))['facts']; S9B=json.load(open('verified_facts_s9b.json'))['facts']
S10=json.load(open('verified_facts_s10.json'))['facts']; S10B=json.load(open('verified_facts_s10b.json'))['facts']; S11=json.load(open('verified_facts_s11.json'))['facts']
AC=json.load(open('allcells_cluster.json')); CIP=json.load(open('controls_inside_polygons.json')); RF=json.load(open('regional_form_check.json'))
rows=[]
def add(gid,sid,label,OR,lo,hi,p,cls,note=''):
    rows.append(dict(id=f"{gid}{sid}",group=gid,specification=label,OR_per_degree=OR,
                     CI_low=lo,CI_high=hi,p=p,stability=cls,note=note))
h=F['S5.headline_interaction']['value']['from_published_table']
add('A',1,'published specification (centroid, unmatched)',h['OR'],*h['ci'],h['p'],'A')
fe=F['S6.event_fixed_effects']['value']
add('A',2,'event fixed effects, 5 km footprint',fe['fe']['OR'],*fe['fe']['ci'],fe['fe']['p'],'B')
add('A',3,'event FE, SE clustered by point',fe['cluster_by_point']['OR'],*fe['cluster_by_point']['ci'],fe['cluster_by_point']['p'],'B')
add('A',4,'event FE, SE clustered by event',fe['cluster_by_event']['OR'],*fe['cluster_by_event']['ci'],fe['cluster_by_event']['p'],'B')
o=F['S6.each_control_once_mc']['value']
add('A',5,'event FE, each control used once (25 draws)',o['OR_mean'],*o['OR_range_mc'],o['p_median'],'C','MC range, not a CI')
fp=F['S6.footprint_sensitivity']['value']
for i,(k,lab) in enumerate([('2km','footprint 2 km grid'),('10km','footprint 10 km grid'),
                            ('20km','footprint 20 km grid'),('buffer2km','footprint 2 km buffer union'),
                            ('buffer5km','footprint 5 km buffer union')],start=6):
    add('A',i,lab,fp[k]['OR'],*fp[k]['ci'],fp[k]['p'],'B')
np_=F['S6.new_pool_models_mc']['value']
for i,(k,lab) in enumerate([('5to1','new control pool, 5:1 (25 draws)'),('10to1','new control pool, 10:1 (25 draws)'),
                            ('20to1','new control pool, 20:1 (25 draws)')],start=11):
    add('A',i,lab,np_[k]['OR_mean'],*np_[k]['OR_range_mc'],np_[k]['p_median'],'C','MC range, not a CI')
u=F['S6.new_pool_models_mc']  # unmatched lives in curve/screens; take from screens baseline new pool
ex=F['S6.event_by_block_clogit_exact']['value']
for i,(k,lab) in enumerate([('20km','conditional logit, event x 20 km block (exact, 10 draws)'),
                            ('10km','conditional logit, event x 10 km block (exact, 10 draws)'),
                            ('5km','conditional logit, event x 5 km block (exact, 10 draws)')],start=14):
    add('A',i,lab,ex[k]['OR_mean'],*ex[k]['OR_range_mc'],ex[k]['p_median'],'C','MC range, not a CI')
pr=F['S6.polygon_representation']['value']
lab={'centroid':'centroid (published)','highest_cell':'highest cell (the rule section 2.3 describes)',
     'steepest_cell':'steepest cell','lowest_cell':'lowest cell','median_cell':'median cell',
     'polygon_mean':'polygon-mean terrain','source_area_top20pct':'source area, top 20% elevation'}
i=1
for k,l in lab.items():
    v=pr[k]; add('B',i,l,v['OR'],*v['ci'],v['p'],'B'); i+=1
for k,l in [('all_cells_area_weighted','all 20 m cells, SE clustered by polygon'),('all_cells_polygon_weight1','all cells, polygon weight 1, SE clustered by polygon (approx.)')]:
    v=AC[k]; add('B',i,l,v['OR'],*v['ci'],v['p'],'B'); i+=1
rc=pr['random_cell_200reps']
add('B',i,'random cell in polygon (200 draws)',rc['OR_mean'],*rc['OR_range_mc'],None,'C',
    f"{100*rc['share_gt1']:.0f}% of draws above 1"); i+=1
sp=F['S6.source_point']['value']['reproduced']
add('B',i,'source point = highest DEM cell, full re-run',sp['OR'],*sp['ci'],sp['p'],'B')
sc=F['S6.screens']['value']
order=[('exclude_reactivation','exclude pre-2018 scar reactivations'),
 ('exclude_debris_200m','exclude <200 m from a debris-flow torrent'),
 ('exclude_debris_500m','exclude <500 m from a debris-flow torrent'),
 ('exclude_road_200m','exclude <200 m from a national or provincial road'),
 ('exclude_river_200m','exclude <200 m from a river polygon'),
 ('exclude_large_ls_potential','exclude large-landslide potential areas'),
 ('exclude_area_ge10ha','exclude area >= 10 ha (official large-scale criterion)'),
 ('exclude_area_ge1ha','exclude area >= 1 ha'),
 ('elongation_lt3','elongation < 3'),('relief_lt50m','within-polygon relief < 50 m'),
 ('relief_lt30m','within-polygon relief < 30 m'),
 ('reliably_shallow_original_controls','proxy-screened subset (area <1 ha, no 2004-2017 scar, >200 m from torrents, outside potential areas), original controls'),
 ('reliably_shallow_new_pool','proxy-screened subset, new control pool'),
 ('all_screens_original_controls','all screens combined, original controls'),
 ('all_screens_new_pool','all screens combined, new control pool')]
for i,(k,l) in enumerate(order,start=1):
    v=sc[k]; add('C',i,l,v['OR'],*v['ci'],v['p'],'A')
bd=F['S6.bamboo_definition']['value']
add('D',1,'bamboo-containing stands pooled, original controls',bd['original_controls']['OR'],*bd['original_controls']['ci'],bd['original_controls']['p'],'A')
add('D',2,'bamboo-containing stands pooled, new control pool',bd['new_pool']['OR'],*bd['new_pool']['ci'],bd['new_pool']['p'],'A')
add('D',3,'drop the two dominant events, pure bamboo',bd['drop2events_pure_original']['OR'],*bd['drop2events_pure_original']['ci'],bd['drop2events_pure_original']['p'],'A')
add('D',4,'drop the two dominant events, pure bamboo, new pool',bd['drop2events_pure_newpool']['OR'],*bd['drop2events_pure_newpool']['ci'],bd['drop2events_pure_newpool']['p'],'A')
add('D',5,'drop the two dominant events, pooled, new pool',bd['drop2events_pooled_newpool']['OR'],*bd['drop2events_pooled_newpool']['ci'],bd['drop2events_pooled_newpool']['p'],'A')
add('D',6,'pooled, all screens, new pool',sc['pooled_all_screens_new_pool']['OR'],*sc['pooled_all_screens_new_pool']['ci'],sc['pooled_all_screens_new_pool']['p'],'A')
add('D',7,'pooled, proxy-screened subset, new pool',sc['pooled_reliably_shallow_new_pool']['OR'],*sc['pooled_reliably_shallow_new_pool']['ci'],sc['pooled_reliably_shallow_new_pool']['p'],'A')
tl=F['S6.three_level']['value']['new_pool']
add('D',8,'three-level model: pure bamboo term',tl['bamboo_x_slope']['OR'],*tl['bamboo_x_slope']['ci'],tl['bamboo_x_slope']['p'],'A')
add('D',9,'three-level model: bamboo-broadleaf mixed term',tl['mixed_x_slope']['OR'],*tl['mixed_x_slope']['ci'],tl['mixed_x_slope']['p'],'A')
for i,r in enumerate(F['S6.per_event']['value']['pure'],start=10):
    EN={'0728豪雨':'0728 rainstorm 2025','凱米颱風':'Typhoon Gaemi 2024'}
    add('D',i,f"event-specific: {EN.get(r['event'],r['event'])}",r['OR'],*r['ci'],r['p'],'A')
ms=F['S6.misclassification']['value']
for i,(k,l) in enumerate([('5pct_both_ways','5% forest-type misclassification (300 draws)'),
                          ('10pct_both_ways','10% forest-type misclassification (300 draws)'),
                          ('20pct_both_ways','20% forest-type misclassification (300 draws)')],start=1):
    v=ms[k]; add('E',i,l,v['OR_mean'],*v['OR_range_mc'],None,'C',f"{100*v['share_gt1']:.1f}% of draws above 1")
# ---- R1 additions: event-structure inference, covariates, screens on controls, growth form, within-period repeats
ecr=F['S5.event_cluster_robust']['value']['reproduced']; add('A',17,'headline, event-cluster robust SE (39 events + controls)',ecr['OR'],*ecr['ci'],ecr['p'],'A')
gl=S9['S9.glmm_event_random_intercept']['value']; add('A',18,'mixed logistic model, event random intercept (VB)',gl['interaction_OR'],*gl['ci'],None,'B','posterior 95% interval')
fr=S10['S10.event_fe_with_rainfall']['value']
add('A',19,'event FE + log max 24-h IMERG rainfall',fr['fe_plus_max24']['OR'],*fr['fe_plus_max24']['ci'],fr['fe_plus_max24']['p'],'B')
frc=S11['S11.fe_rain_spline_bands']['value']['linear_interaction']['control_point_clustered']; add('A',20,'event FE + log max 24-h rainfall, SE clustered by control point',frc['OR'],*frc['ci'],frc['p'],'B')
add('A',21,'event FE + log max 24-h + log total rainfall',fr['fe_plus_max24_total']['OR'],*fr['fe_plus_max24_total']['ci'],fr['fe_plus_max24_total']['p'],'B')
add('A',22,'event FE + log max 72-h rainfall',fr['fe_plus_max72']['OR'],*fr['fe_plus_max72']['ci'],fr['fe_plus_max72']['p'],'B')
nr=S10['S10.new_pool_5to1_with_rainfall']['value']['plus_max24']; add('A',23,'new control pool 5:1 + log max 24-h rainfall (25 draws)',nr['OR_mean'],*nr['OR_range_mc'],nr['p_median'],'C','MC range, not a CI')
cl=S10['S10.island_wide_with_climatology']['value']['plus_mean_annual_precip']; add('A',24,'island-wide + CHIRPS mean annual precipitation',cl['OR'],*cl['ci'],cl['p'],'B')
cs=S9['S9.covariate_specs']['value']['specs']
for i,(k,l) in enumerate([('plus_twi','+ topographic wetness index'),('plus_soil','+ topsoil organic carbon, pH, clay'),('plus_twi_soil','+ wetness index and soil'),('plus_lithology','+ surface lithology classes'),('within_siliciclastic','within siliciclastic sedimentary lithology only')],start=1):
    v=cs[k]; add('F',i,l,v['OR'],*v['ci'],v['p'],'A')
ch=S9B['S9.canopy_height_checks']['value']
chk=[k for k in ch if isinstance(ch[k],dict) and 'OR' in ch[k]]
if 'plus_canopy_height' in ch: v=ch['plus_canopy_height']; add('F',6,'+ canopy height',v['OR'],*v['ci'],v['p'],'A')
else:
    v=ch[chk[0]]; add('F',6,'+ canopy height',v['OR'],*v['ci'],v['p'],'A')
gr=S10B['S10.pre_event_greenness']['value']['like_for_like']
for i,(k,l) in enumerate([('plus_ndvi','+ pre-event NDVI (like-for-like window)'),('plus_ndvi_ndmi','+ pre-event NDVI and NDMI'),('plus_nbr','+ pre-event NBR')],start=7):
    v=gr[k]; add('F',i,l,v['OR'],*v['ci'],v['p'],'A')
add('F',10,'greenness subsample without indices (5,393 cases, 9,148 controls)',gr['base_on_covered_subset']['OR'],*gr['base_on_covered_subset']['ci'],gr['base_on_covered_subset']['p'],'A')
grm=S10B['S10.pre_event_greenness']['value'].get('multiyear_controls')
if grm:
    add('F',11,'+ pre-event NDVI, controls as 2019-2024 median (earlier definition)',grm['plus_ndvi']['OR'],*grm['plus_ndvi']['ci'],grm['plus_ndvi']['p'],'A')
    add('F',12,'+ pre-event NDVI and NDMI, controls as 2019-2024 median',grm['plus_ndvi_ndmi']['OR'],*grm['plus_ndvi_ndmi']['ci'],grm['plus_ndvi_ndmi']['p'],'A')
ci_=CIP['excluding_inside']; add('C',16,'exclude the 42 controls inside 2018-2025 landslide polygons',ci_['OR'],*ci_['ci'],ci_['p'],'A')
wr=S11['S11.within_period_reactivation']['value']
add('C',17,'exclude cases inside an earlier 2018-2025 polygon (within-period repeat)',wr['rule_earlier_polygon']['refit']['OR'],*wr['rule_earlier_polygon']['refit']['ci'],wr['rule_earlier_polygon']['refit']['p'],'A')
add('C',18,'exclude cases inside any other-event 2018-2025 polygon',wr['rule_any_other_event']['refit']['OR'],*wr['rule_any_other_event']['refit']['ci'],wr['rule_any_other_event']['refit']['p'],'A')
bf=RF['by_form_original_controls']
add('D',12,'running (monopodial) bamboo only vs broadleaf',bf['單桿狀竹(running)']['OR'],*bf['單桿狀竹(running)']['ci'],bf['單桿狀竹(running)']['p'],'A')
add('D',13,'clumping (sympodial) bamboo only vs broadleaf',bf['叢生狀竹(clumping)']['OR'],*bf['叢生狀竹(clumping)']['ci'],bf['叢生狀竹(clumping)']['p'],'A')
env=S11['S11.envelope_exact']['value']
df=pd.DataFrame(rows)
df['significant']=df.apply(lambda r: 'MC' if r.p is None or (isinstance(r.p,float) and np.isnan(r.p)) else ('yes' if r.p<0.05 else 'no'),axis=1)
df.to_csv(ROOT+'/tables/TableS1_specification_curve.csv',index=False,encoding='utf-8-sig')
print("rows",len(df),"| OR>1:",int((df.OR_per_degree>1).sum()),
      "| p<0.05:",int((df.p.astype(float)<0.05).sum()),"of",int(df.p.notna().sum()))
print("OR range %.4f - %.4f"%(df.OR_per_degree.min(),df.OR_per_degree.max()))
print("non-significant:",list(df[(df.p.notna())&(df.p.astype(float)>=0.05)].specification))
d=df.sort_values('OR_per_degree').reset_index(drop=True)
grp={'A':'#1f6f8b','B':'#b3541e','C':'#3d6b35','D':'#6b3d8b','E':'#8b6b1e','F':'#3b3b8b'}
fig,ax=plt.subplots(figsize=(7.8,13.6))
for i,r in d.iterrows():
    c=grp[r.group]; filled = (r.significant!='no')
    ax.plot([r.CI_low,r.CI_high],[i,i],color=c,lw=1.35,alpha=.85,zorder=2)
    ax.plot(r.OR_per_degree,i,'o',ms=4.7,color=c,mfc=c if filled else 'white',mec=c,mew=1.2,zorder=3)
ax.axvline(1,color='0.35',lw=1,ls='--',zorder=1); ax.axvline(1.0341,color='k',lw=1,ls=':',zorder=1)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([t if len(t)<=70 else t[:67]+'...' for t in d.specification],fontsize=6.2)
ax.set_xlabel('Bamboo × slope interaction odds ratio per degree',fontsize=9)
ax.set_xlim(0.962,1.082); ax.set_ylim(-1.2,len(d)+9.8); ax.tick_params(axis='x',labelsize=8)
cnt=d.group.value_counts()
h=[Line2D([],[],color=grp[g],lw=1.35,marker='o',ms=4.7,label=f"{g}  {t} ({cnt.get(g,0)})") for g,t in
   [('A','Event exposure, rainfall and inference'),('B','Polygon and source-point representation'),
    ('C','Failure-type and proxy screens'),('D','Stand definition, growth form and event structure'),
    ('E','Forest-map misclassification'),('F','Additional covariates')]]
h+= [Line2D([],[],color='0.3',lw=0,marker='o',ms=4.7,mfc='white',mec='0.3',label='open marker: p ≥ 0.05'),
     Line2D([],[],color='k',lw=1,ls=':',label='published estimate, 1.0341')]
ax.legend(handles=h,fontsize=6.6,loc='upper left',bbox_to_anchor=(0.004,0.999),frameon=False,handlelength=1.8)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_title(f'Specification curve: {len(d)} alternative analyses',fontsize=9.8,pad=10,loc='left')
plt.tight_layout(); plt.savefig(ROOT+'/tables/FigS1_specification_curve.png',dpi=400,bbox_inches='tight')
print("figure written")
