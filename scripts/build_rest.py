# -*- coding: utf-8 -*-
"""F1 density, F3 OR-by-type dumbbell, F5 robustness forest plot, F6 CV + size sensitivity. Palette D, neutral-ink labels, no dual-axis."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import palette as P
R=json.load(open('../expected_outputs/results.json'))
P.set_fonts(9.5)
OK,STROKE,EN,INK,GREY=P.OK,P.STROKE,P.EN,P.INK,P.GREY

# F1
E1=R['E1_density']; rows=E1['rows']; mean=E1['mean_density_per_km2']; y=np.arange(len(rows))[::-1]
fig,ax=plt.subplots(figsize=(7.6,4.0))
for yi,r in zip(y,rows):
    ax.barh(yi,r['density_per_km2'],color=OK[r['type']],edgecolor=STROKE[r['type']],lw=0.8,height=0.66,zorder=3)
    ax.text(r['density_per_km2']+0.03,yi,f"{r['density_per_km2']:.2f}   (RR {r['RR_vs_mean']:.2f}, n={r['n_rain']})",va='center',ha='left',fontsize=9.2,color=INK,zorder=4)
ax.axvline(mean,color=GREY,ls='--',lw=1.2,zorder=2); ax.text(mean,len(rows)-0.32,'  forest-wide mean',color=GREY,fontsize=8.6,va='top')
ax.set_yticks(y); ax.set_yticklabels([EN[r['type']] for r in rows],fontsize=9.6)
ax.set_xlim(0,2.35); ax.set_xlabel('Rainfall-triggered landslide density  (events per km² of forest type)')

for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('../figures/F1_density.png',dpi=400,bbox_inches='tight'); plt.close(); print('F1 done')

# F3
e3=R['E3_or_by_type']['rows']
ordmap={'竹林':0,'竹闊混淆林':1,'針闊葉樹混淆':2,'針葉樹林型':3,'待成林地':4}   # same order as Fig 2
e3=sorted(e3,key=lambda r:ordmap.get(r['type'],9))
fig,ax=plt.subplots(figsize=(7.4,4.0)); yy=np.arange(len(e3))[::-1]
off=0.17
for yi,r in zip(yy,e3):
    k=r['type']; col,cold=OK[k],STROKE[k]; yr,ya=yi+off,yi-off
    ax.plot([r['adj_ci'][0],r['adj_ci'][1]],[ya,ya],color=cold,lw=2.4,alpha=0.5,zorder=2)          # adjusted 95% CI
    ax.plot([r['OR_raw'],r['OR_adj']],[yr,ya],color=GREY,lw=1.2,zorder=2)                            # shift connector
    ax.plot(r['OR_raw'],yr,'o',mfc='white',mec=cold,mew=1.6,ms=8,zorder=4)                           # raw (hollow, upper)
    ax.plot(r['OR_adj'],ya,'o',mfc=col,mec=cold,mew=1.0,ms=9,zorder=5)                               # adjusted (filled, lower)
    lx=max(r['OR_raw'],r['adj_ci'][1])*1.06
    ax.text(lx,yr,f"raw {r['OR_raw']:.2f}",va='center',ha='left',fontsize=7.8,color=GREY,zorder=6,bbox=dict(fc='white',ec='none',pad=0.15,alpha=0.9))
    ax.text(lx,ya,f"adj {r['OR_adj']:.2f}",va='center',ha='left',fontsize=8.4,color=INK,fontweight='bold',zorder=6,bbox=dict(fc='white',ec='none',pad=0.15,alpha=0.9))
ax.axvline(1.0,color=STROKE['闊葉樹林型'],ls='--',lw=1.3,zorder=1)
ax.text(1.04,0.12,'broadleaf = 1 (reference)',color=STROKE['闊葉樹林型'],fontsize=8.6,va='bottom',ha='left')
ax.set_xscale('log'); ax.set_xticks([0.3,0.5,1,2,4]); ax.set_xticklabels(['0.3','0.5','1','2','4'])
ax.set_yticks(yy); ax.set_yticklabels([f"{EN[r['type']]}\n(n={r['n_case']})" for r in e3],fontsize=8.8)
ax.set_xlim(0.24,6.8); ax.set_xlabel('Odds ratio vs broadleaf')

for sp in ('top','right'): ax.spines[sp].set_visible(False)
leg3=[Line2D([],[],marker='o',mfc='white',mec='#444',mew=1.6,ls='',ms=8,label='Unadjusted'),
      Line2D([],[],marker='o',mfc='#777',mec='#444',mew=1.0,ls='',ms=9,label='Terrain-adjusted'),
      Line2D([],[],color='#999',lw=2.4,alpha=0.7,label='95% CI, adjusted')]
ax.legend(handles=leg3,loc='upper right',fontsize=8.2,frameon=True,framealpha=0.96,edgecolor='#cccccc',borderpad=0.55,labelspacing=0.5)
plt.tight_layout(); plt.savefig('../figures/F3_or.png',dpi=400,bbox_inches='tight'); plt.close(); print('F3 done')

# F5
G,GD=OK['竹林'],STROKE['竹林']
items=[('Terrain (slope, aspect, curvature, elevation)',1.034,[1.012,1.056],0),('+ TWI + aspect',1.036,[1.014,1.058],0),
 ('+ soil group',1.039,[1.016,1.062],0),('+ TWI + soil',1.040,[1.017,1.063],0),('Source-area attribution',1.027,[1.006,1.049],0),
 ('Pooled bamboo (bamboo + bamboo-broadleaf)',1.024,[1.006,1.043],0),('+ lithology (USGS ELU)',1.036,[1.014,1.059],1),
 ('Within Western Foothills only',1.034,[1.011,1.059],1),('Event-cluster-robust SE',1.034,[1.013,1.056],1),('Event random-intercept GLMM',1.038,[1.017,1.059],1)]
fig,ax=plt.subplots(figsize=(7.5,4.5)); yy=np.arange(len(items))[::-1]
for idx,(yi,(lab,orr,ci,hl)) in enumerate(zip(yy,items)):
    key=(idx==0)                                   # first row = headline specification
    col=GD if (hl or key) else G
    ax.plot(ci,[yi,yi],color=col,lw=2.7 if key else 2.0,zorder=2)
    ax.plot(orr,yi,'s',mfc=(G if key else 'white'),mec=col,mew=1.8,ms=9.5 if key else 8,zorder=3)
    if key:
        ax.text(1.067,yi,f"{orr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  p = 0.002",va='center',ha='left',fontsize=9.2,color=INK,fontweight='bold')
    else:
        ax.text(1.067,yi,f"{orr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",va='center',ha='left',fontsize=8.2,color=INK)
ax.axvline(1.0,color=GREY,ls='--',lw=1.1); ax.set_yticks(yy); _lab=ax.set_yticklabels([it[0] for it in items],fontsize=8.8); _lab[0].set_fontweight('bold')
ax.set_xlim(0.998,1.093); ax.set_xlabel('Interaction odds ratio per degree, bamboo vs broadleaf')

for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('../figures/F5_robust_litho.png',dpi=400,bbox_inches='tight'); plt.close(); print('F5 done')

# F6
cv=R['E7_spatial_cv']; folds=cv['folds']; E8=R['E8_size_sensitivity']['rows']
fig,(axL,axR)=plt.subplots(1,2,figsize=(7.6,3.9)); x=np.arange(len(folds))
axL.bar(x,[f['interaction_OR'] for f in folds],color=OK['竹林'],edgecolor=STROKE['竹林'],lw=0.7,width=0.62,zorder=3)
axL.axhline(1.0,color=GREY,ls='--',lw=1.1,zorder=2)
for i,f in zip(x,folds): axL.text(i,f['interaction_OR']+0.002,f"{f['interaction_OR']:.3f}",ha='center',va='bottom',fontsize=7.6,color=INK)
axL.set_xticks(x); axL.set_xticklabels([f"F{f['fold']}\nn={f['n_test_bamboo_ls']}" for f in folds],fontsize=8.0)
axL.set_ylim(0.99,1.06); axL.set_ylabel('Interaction OR (held-out 20 km block)')
axL.set_title(f"(a) 20 km spatial-block CV  ·  mean AUC {cv['mean_AUC']:.2f}",loc='left',fontsize=9.6)
for sp in ('top','right'): axL.spines[sp].set_visible(False)
lab8=['all','≥0.1 ha','≥0.5 ha','≥1.0 ha']; xx=np.arange(len(E8))
for i,r in zip(xx,E8):
    axR.plot([i,i],r['ci'],color=STROKE['竹闊混淆林'],lw=2.0,zorder=2)
    axR.plot(i,r['OR'],'o',mfc=OK['竹闊混淆林'],mec=STROKE['竹闊混淆林'],mew=1.0,ms=9,zorder=3)
    axR.text(i,r['ci'][1]+0.004,f"n={r['n_bamboo_ls']}",ha='center',va='bottom',fontsize=7.8,color=INK)
axR.axhline(1.0,color=GREY,ls='--',lw=1.1,zorder=1); axR.set_xticks(xx); axR.set_xticklabels(lab8[:len(E8)],fontsize=8.4)
axR.set_xlim(-0.5,len(E8)-0.5); axR.set_ylabel('Interaction OR per degree'); axR.set_title('(b) Size / detection-bias sensitivity',loc='left',fontsize=9.6)
for sp in ('top','right'): axR.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('../figures/F6_cv_size.png',dpi=400,bbox_inches='tight'); plt.close(); print('F6 done')
