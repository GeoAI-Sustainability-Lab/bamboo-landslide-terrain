# -*- coding: utf-8 -*-
"""Fig 3 (slope niche) and Fig 5 (slope-signature spline), palette D.
Fig 3: violins + equal-spacing beeswarm (one global spacing, width = count, comparable across types),
open IQR box, whisker to p5/p95 with caps, filled median dot. No in-figure title (caption carries it).
Fig 5: main OR(slope) panel plus a separate bottom count panel sharing the X axis (histogram + rug).
Scenario dots at the three real case slopes 13, 29, 42. No text touches curves."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import collections
import palette as P
R = json.load(open('../expected_outputs/results.json'))
P.set_fonts(9.5)
OK, STROKE, EN, INK, GREY = P.OK, P.STROKE, P.EN, P.INK, P.GREY

# ---- Fig 3 (niche) ----
bt = R['E2_slope_niche']['byType']
order = [k for k in ['竹林','竹闊混淆林','闊葉樹林型','針闊葉樹混淆','針葉樹林型','待成林地'] if k in bt]
data = [np.array(bt[k]['slope_samples'], float) for k in order]
fig, ax = plt.subplots(figsize=(8.8, 4.6))
parts = ax.violinplot(data, positions=np.arange(len(order)), widths=0.9, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(OK[order[i]]); pc.set_alpha(0.13); pc.set_edgecolor(STROKE[order[i]]); pc.set_linewidth(1.3)

# equal-spacing beeswarm: ONE global spacing for all types, so width = count everywhere
BW_BIN, HALFW = 1.6, 0.40
binned = [np.floor(v/BW_BIN).astype(int) for v in data]
maxc = max(max(collections.Counter(b).values()) for b in binned)
STEP = 2*HALFW/maxc
def bee(bins):
    off = np.zeros(len(bins))
    for bb in collections.Counter(bins):
        idx = np.where(bins == bb)[0]
        k = np.arange(len(idx)); off[idx] = ((-1)**k)*((k+1)//2)
    return off*STEP
wcap = 0.11
for i, (k, v, bins) in enumerate(zip(order, data, binned)):
    edge = STROKE[k]
    ax.scatter(i+bee(bins), v, s=4.0, color=edge, alpha=0.16, edgecolors='none', zorder=2)
    p5, q1, med, q3, p95 = np.percentile(v, [5,25,50,75,95])
    ax.plot([i,i], [p5,p95], color=edge, lw=1.0, zorder=3)
    ax.add_patch(Rectangle((i-0.17, q1), 0.34, q3-q1, fc='none', ec=edge, lw=1.6, zorder=4))
    ax.plot([i-wcap,i+wcap], [p95,p95], color=edge, lw=1.6, zorder=4, solid_capstyle='round')
    ax.plot([i-wcap,i+wcap], [p5,p5],  color=edge, lw=1.6, zorder=4, solid_capstyle='round')
    ax.text(i, p95+1.6, f'{p95:.0f}', ha='center', va='bottom', fontsize=8.2, color=INK)
    ax.text(i, p5-1.6,  f'{p5:.0f}',  ha='center', va='top',    fontsize=8.2, color=INK)
    ax.plot(i, med, 'o', mfc=OK[k], mec=edge, mew=1.2, ms=8.6, zorder=5)
    ax.text(i+0.23, med, f'{med:.0f}', va='center', ha='left', fontsize=9.2, color=INK, fontweight='bold', zorder=6)
    ax.text(i, -8.4, f'n={len(v)}', ha='center', va='top', fontsize=8.0, color=INK)
leg2 = [Patch(fc='#cfcfcf', alpha=0.6, ec='#8a8a8a', label='Slope distribution (violin)'),
        Line2D([],[], marker='o', mfc='#9a9a9a', mec='#444', mew=1.2, ls='', ms=8, label='Median (filled, with value)'),
        Patch(fc='none', ec='#444', lw=1.6, label='Interquartile range (open box)'),
        Line2D([],[], color='#444', lw=1.0, marker='_', ms=9, mew=1.6, label='Whisker to 5th / 95th percentile'),
        Line2D([],[], marker='o', mfc='#888', mec='none', ls='', ms=4, label='Sample (fixed spacing, width = count)')]
ax.legend(handles=leg2, loc='upper left', fontsize=7.6, frameon=True, framealpha=0.96, edgecolor='#cccccc', handlelength=1.8, borderpad=0.55, labelspacing=0.45)
ax.set_xticks(np.arange(len(order))); ax.set_xticklabels([EN[k] for k in order], rotation=16, ha='right', fontsize=9)
ax.set_ylabel('Slope (degrees)'); ax.set_ylim(-12, 92); ax.set_xlim(-0.6, len(order)-0.3)
plt.tight_layout(); plt.savefig('../figures/F2_niche.png', bbox_inches='tight'); plt.close(); print('F2 done')

# ---- Fig 5 (spline + marginal counts) ----
S = R['E4_spline']
grid=np.array(S['grid'],float); orr=np.array(S['or'],float); lo=np.array(S['ci_lo'],float); hi=np.array(S['ci_hi'],float)
bslopes=np.array(S['bamboo_case_slopes'],float); niche=float(S['bamboo_niche_p95']); cross=float(S['crossover_slope'])
GREEN=OK['竹林']; GREEN_D=STROKE['竹林']; BROWN=STROKE['闊葉樹林型']; OLIVE_D=STROKE['針闊葉樹混淆']; RUST_D=STROKE['待成林地']; TAN=STROKE['闊葉樹林型']
m=(grid>=6)&(grid<=64); g,o,l,h=grid[m],orr[m],lo[m],hi[m]
fig = plt.figure(figsize=(8.8, 5.8))
ax  = fig.add_axes([0.085, 0.315, 0.895, 0.640])
axh = fig.add_axes([0.085, 0.095, 0.895, 0.180])
ax.axvspan(niche,64,color=TAN,alpha=0.08,lw=0,zorder=0)
ax.fill_between(g,l,h,color=GREEN,alpha=0.14,lw=0,zorder=1)
ax.plot(g,o,color=GREEN,lw=2.9,zorder=6)
ax.axhline(1.0,color=BROWN,ls='--',lw=1.4,zorder=4); ax.text(63.6,1.04,'broadleaf = 1',color=BROWN,ha='right',va='bottom',fontsize=9.0)
ax.axvline(niche,color=GREEN,ls=':',lw=1.2,alpha=0.8,zorder=3)
ax.text(niche-0.8,1.42,'Protection',color=GREEN_D,fontsize=10.5,rotation=90,ha='right',va='center',zorder=7)
ax.text(niche+0.8,1.42,'Protection lost',color=RUST_D,fontsize=10.5,rotation=90,ha='left',va='center',zorder=7)
ax.text(niche-1.2,0.335,f'bamboo niche edge\n(p95 = {niche:.0f}°)',color=GREEN_D,ha='right',va='center',fontsize=9.2,zorder=7)
ax.text(56.5,0.45,'extrapolation\nbeyond niche',color=BROWN,ha='center',va='center',fontsize=8.2,zorder=7)
ax.plot([cross,cross],[1.0,2.22],color=GREY,ls=':',lw=1.3,zorder=5); ax.plot(cross,1.0,'o',mfc='white',mec=INK,mew=1.9,ms=9.5,zorder=8)
ax.text(cross,2.26,f'crossover ≈ {cross:.0f}°',color=INK,ha='center',va='bottom',fontsize=11.0,fontweight='bold',zorder=8)
for xv,lab,col,dx,dy,ha,va in [(13,'Gentle',GREEN_D,0,0.13,'center','bottom'),(29,'Moderate',OLIVE_D,0,0.13,'center','bottom'),(42,'Steep',RUST_D,3.6,-0.07,'left','center')]:
    yv=np.interp(xv,g,o); ax.plot(xv,yv,'o',mfc='white',mec=col,mew=2.0,ms=8.6,zorder=9)
    ax.text(xv+dx,yv+dy,f'{lab} {xv}°',color=col,ha=ha,va=va,fontsize=9.4,fontweight='bold',zorder=9)
leg4=[Line2D([],[],color=GREEN,lw=2.9,label='Odds ratio, bamboo vs broadleaf (spline)'),
      Patch(fc=GREEN,alpha=0.14,ec='none',label='95% confidence band'),
      Line2D([],[],color=BROWN,lw=1.4,ls='--',label='Broadleaf reference, OR = 1'),
      Patch(fc=TAN,alpha=0.10,ec='none',label='Extrapolation beyond niche (> p95)')]
ax.legend(handles=leg4,loc='upper left',fontsize=8.0,frameon=True,framealpha=0.96,edgecolor='#cccccc',handlelength=1.8,borderpad=0.6,labelspacing=0.5)
ax.set_xlim(6,64); ax.set_ylim(0.25,2.5); ax.set_ylabel('Odds ratio, bamboo vs broadleaf (terrain-adjusted)',fontsize=9.0)
ax.tick_params(labelbottom=False)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
# bottom panel: observed bamboo landslide slopes, counts, shared X
bins=np.arange(6,66,2)
axh.hist(bslopes,bins=bins,color=GREEN,alpha=0.55,edgecolor=GREEN_D,lw=0.6,zorder=3)
for s in bslopes: axh.plot([s,s],[0,0.9],color=GREEN_D,lw=1.0,alpha=0.8,zorder=4)
axh.axvspan(niche,64,color=TAN,alpha=0.08,lw=0,zorder=0)
axh.axvline(niche,color=GREEN,ls=':',lw=1.2,alpha=0.8,zorder=2)
axh.set_xlim(6,64); axh.set_xlabel('Hillslope steepness (degrees)')
axh.set_ylabel(f'Count\n(n={len(bslopes)})',fontsize=8.4)
for sp in ('top','right'): axh.spines[sp].set_visible(False)
plt.savefig('../figures/F4_reversal.png',bbox_inches='tight'); plt.close(); print('F4 done')
