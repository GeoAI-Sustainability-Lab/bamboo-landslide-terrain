# -*- coding: utf-8 -*-
"""Fig 7 (fused, side-by-side): main panel shows the bamboo-vs-broadleaf interaction effect across slope
for the base model and the model with pre-event greenness and canopy height added (near-coincident curves);
right panel ranks median canopy height by forest type. No overlap between panels. Sans font."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import palette as P
P.set_fonts(9.5)
OK, STROKE, INK, GREY = P.OK, P.STROKE, P.INK, P.GREY

# real per-degree interaction OR (CHM/NDVI robustness refits)
r_base, r_both = 1.0318, 1.0337

fig = plt.figure(figsize=(7.8, 3.9))
ax  = fig.add_axes([0.085, 0.155, 0.545, 0.680])
axb = fig.add_axes([0.740, 0.155, 0.240, 0.680])
fig.text(0.085, 0.955, 'Interaction effect is nearly unchanged when greenness and canopy are added',
         fontsize=11, fontweight='bold', color='#222', ha='left', va='top')

s = np.linspace(6, 46, 160); d0 = s - 6
yb, yo = r_base**d0, r_both**d0
ax.fill_between(s, yb, yo, color=GREY, alpha=0.18, lw=0, zorder=2)
ax.plot(s, yb, color=STROKE['竹林'], lw=3.0, ls='-', zorder=4)
ax.plot(s, yo, color=INK, lw=2.2, ls=(0, (5, 2.5)), zorder=5)
ax.axhline(1.0, color=GREY, ls=':', lw=1.0, zorder=1)
ax.text(45.4, 1.04, 'gentlest slope = 1 (reference)', fontsize=7.8, color=GREY, va='bottom', ha='right')
ax.set_xlim(6, 46); ax.set_ylim(0.85, 4.2)
ax.set_xlabel('Hillslope steepness (degrees)')
ax.set_ylabel('Relative interaction effect\n(bamboo vs broadleaf)')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
leg = [Line2D([], [], color=STROKE['竹林'], lw=3.0, label='Terrain only (base)  ·  1.032 per degree'),
       Line2D([], [], color=INK, lw=2.2, ls=(0, (5, 2.5)), label='+ greenness and canopy  ·  1.034 per degree'),
       Line2D([], [], color=GREY, lw=8, alpha=0.18, label='Difference between the two models')]
ax.legend(handles=leg, loc='upper left', fontsize=8.0, frameon=True, framealpha=0.96,
          edgecolor='#cccccc', borderpad=0.55, handlelength=2.0)

d = pd.read_csv('../data/combined_chm.csv'); d = d[d.chm >= 0]; bg = d[d.kind == 'bg']
order = [('竹林','Bamboo'),('竹闊混淆林','Bamboo–broadleaf'),('闊葉樹林型','Broadleaf'),
         ('針闊葉樹混淆','Conifer–broadleaf'),('針葉樹林型','Conifer')]
meds = [(en, float(np.median(bg[bg.forest_type==k].chm)), k) for k, en in order]
yy = np.arange(len(meds))[::-1]
for y, (en, mv, k) in zip(yy, meds):
    axb.barh(y, mv, color=OK[k], edgecolor=STROKE[k], lw=0.8, height=0.62, zorder=3)
    axb.text(mv+0.5, y, f'{mv:.0f}', va='center', ha='left', fontsize=8.0, color=INK,
             fontweight=('bold' if k == '竹林' else 'normal'))
axb.set_yticks(yy); axb.set_yticklabels([en for en, _, _ in meds], fontsize=7.8)
axb.set_xlim(0, 20); axb.set_xlabel('Median canopy height\n(m, CHMv2)', fontsize=8.0)
axb.tick_params(labelsize=7.4)
for sp in ('top', 'right'): axb.spines[sp].set_visible(False)

plt.savefig('../figures/F8_chm_validity.png', dpi=400, bbox_inches='tight')
print('Fig7 side-by-side done')
