# -*- coding: utf-8 -*-
"""Fig 1 — research architecture and workflow (steps only, no results).
Academic flowchart conventions: process = rounded rectangle, decision = diamond,
data input = parallelogram. All process boxes the SAME size (equal importance),
four aligned columns, bottom row of four equally spaced steps aligned to the columns,
generous margins to the dashed group borders, right-angle arrows that never cross text."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Patch
plt.rcParams.update({'font.family':'DejaVu Sans'})

TEAL,  TEAL_F  = '#2C8C86', '#E4F2F0'   # data input
BLUE,  BLUE_F  = '#2C6FB0', '#E4EDF6'   # model step
PINK,  PINK_F  = '#C0567E', '#F8DCE6'   # key test
ORNG,  ORNG_F  = '#D07A28', '#FBEBDC'   # decision
GREEN, GREEN_F = '#2E8B37', '#E6F2E7'   # outcome step
GREY,  GREY_F  = '#6C7A86', '#EFF1F3'   # baseline comparison
INK='#222b31'; GRP='#8a949c'

fig=plt.figure(figsize=(11.6,7.8))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,68); ax.axis('off')

BW, BH = 19.0, 6.2                     # ONE size for every process box
C1, C2, C3, C4 = 13.0, 39.0, 65.5, 88.0   # four shared column centres
R1, R2, R3, R4 = 56.6, 49.2, 41.8, 34.4   # row centres in the top panels
RD = 18.2                                  # bottom-row centre

def rect(cx,cy,txt,ec,fc,tc=None,bold=False):
    ax.add_patch(FancyBboxPatch((cx-BW/2,cy-BH/2),BW,BH,boxstyle='round,pad=0.12,rounding_size=1.1',
                 fc=fc,ec=ec,lw=1.9,zorder=4))
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=10.8,color=(tc or INK),
            fontweight='normal',zorder=5)
def para(cx,cy,txt,ec,fc):
    sk=1.7
    p=[(cx-BW/2+sk,cy+BH/2),(cx+BW/2+sk,cy+BH/2),(cx+BW/2-sk,cy-BH/2),(cx-BW/2-sk,cy-BH/2)]
    ax.add_patch(Polygon(p,closed=True,fc=fc,ec=ec,lw=1.9,joinstyle='round',zorder=4))
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=10.6,color=INK,zorder=5)
def diamond(cx,cy,txt,ec,fc,W=21.0,H=11.0):
    p=[(cx,cy+H/2),(cx+W/2,cy),(cx,cy-H/2),(cx-W/2,cy)]
    ax.add_patch(Polygon(p,closed=True,fc=fc,ec=ec,lw=1.9,joinstyle='round',zorder=4))
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=10.4,color=INK,zorder=5)
def elbow(pts,color,lw=2.0,lab=None,labxy=None):
    for i in range(len(pts)-2):
        ax.plot([pts[i][0],pts[i+1][0]],[pts[i][1],pts[i+1][1]],color=color,lw=lw,
                solid_capstyle='round',zorder=2)
    ax.annotate('',xy=pts[-1],xytext=pts[-2],
                arrowprops=dict(arrowstyle='-|>',mutation_scale=14,color=color,lw=lw,shrinkA=0,shrinkB=0),zorder=3)
    if lab: ax.text(labxy[0],labxy[1],lab,fontsize=9.2,color=color,zorder=6,ha='center')
def group(x0,y0,x1,y1,label):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle='round,pad=0,rounding_size=1.2',
                 fc='none',ec=GRP,lw=1.3,ls=(0,(6,4)),zorder=1))
    ax.text(x0+1.5,y1-1.1,label,ha='left',va='top',fontsize=10.8,color='#4a565f',zorder=2)

# ---------------- A  data inputs (parallelograms) ----------------
group(0.8,29.0,25.0,63.4,'A  Data inputs')
Ain=[('Landslide inventory\n2018 to 2025',R1),('Forest type map\n4th inventory',R2),
     ('Terrain layers\n20 m DEM',R3),('Soil, greenness,\ncanopy height',R4)]
for t,cy in Ain: para(C1,cy,t,TEAL,TEAL_F)

# ---------------- B  design and model ----------------
group(26.5,29.0,51.5,63.4,'B  Design and model')
rect(C2,R1,'Naive density\ncomparison',GREY,GREY_F)
rect(C2,R2,'Case and control\nsampling',BLUE,BLUE_F)
rect(C2,R3,'Terrain-adjusted\nlogistic model',BLUE,BLUE_F)
rect(C2,R4,'Slope interaction\ntest',PINK,PINK_F,tc=PINK,bold=True)
for _,cy in Ain: ax.plot([C1+BW/2-0.3,25.6],[cy,cy],color=TEAL,lw=1.6,zorder=2)
ax.plot([25.6,25.6],[R4,R1],color=TEAL,lw=1.6,solid_capstyle='round',zorder=2)
elbow([(25.6,R1),(C2-BW/2-0.2,R1)],TEAL,lw=1.6)
for ya,yb in [(R1-BH/2,R2+BH/2),(R2-BH/2,R3+BH/2),(R3-BH/2,R4+BH/2)]:
    elbow([(C2,ya),(C2,yb)],BLUE)

# ---------------- C  inference gate (diamond decision) ----------------
group(54.0,29.0,99.0,63.4,'C  Inference gate')
diamond(C3,R2,'Robust and\nsignificant?',ORNG,ORNG_F)
rect(C4,R1,'Retain effect',GREEN,GREEN_F,bold=True)
rect(C3,R4,'Revise model',ORNG,ORNG_F)
elbow([(C2+BW/2,R4),(52.7,R4),(52.7,R2),(C3-10.5,R2)],ORNG,lab='test',labxy=(52.7,R2+1.3))
elbow([(C3+10.5,R2),(76.9,R2),(76.9,R1),(C4-BW/2,R1)],GREEN,lab='YES',labxy=(76.4,R1+0.9))
elbow([(C3,R2-5.5),(C3,R4+BH/2)],ORNG,lab='NO',labxy=(C3+1.9,(R2-5.5+R4+BH/2)/2))
# retained effect continues in row D, routed around all boxes and text
elbow([(C4,R1-BH/2),(C4,26.6),(0.4,26.6),(0.4,RD),(C1-BW/2-0.1,RD)],GREEN,
      lab='within niche',labxy=(55.0,25.6))

# ---------------- D  scope and reporting (four aligned steps) ----------------
group(0.8,12.6,99.0,24.8,'D  Scope and reporting')
rect(C1,RD,'Inference within\nbamboo niche',BLUE,BLUE_F)
rect(C2,RD,'Steep slopes as\nextrapolation',BLUE,BLUE_F)
rect(C3,RD,'Effect\ninterpretation',GREEN,GREEN_F)
rect(C4,RD,'Results\nreporting',GREEN,GREEN_F,bold=True)
elbow([(C1+BW/2,RD),(C2-BW/2-0.2,RD)],BLUE)
elbow([(C2+BW/2,RD),(C3-BW/2-0.2,RD)],GREEN)
elbow([(C3+BW/2,RD),(C4-BW/2-0.2,RD)],GREEN)

# ---------------- legend: shape and color = function ----------------
leg=[Patch(fc=TEAL_F,ec=TEAL,lw=1.6,label='Data input (parallelogram)'),
     Patch(fc=GREY_F,ec=GREY,lw=1.6,label='Baseline comparison'),
     Patch(fc=BLUE_F,ec=BLUE,lw=1.6,label='Model step'),
     Patch(fc=PINK_F,ec=PINK,lw=1.6,label='Key test'),
     Patch(fc=ORNG_F,ec=ORNG,lw=1.6,label='Decision (diamond)'),
     Patch(fc=GREEN_F,ec=GREEN,lw=1.6,label='Outcome step')]
fig.legend(handles=leg,loc='lower center',ncol=6,fontsize=8.8,frameon=False,bbox_to_anchor=(0.5,0.14),
           handlelength=1.5,columnspacing=1.2,handletextpad=0.5)

plt.savefig('../figures/F11_flow.png',dpi=350,bbox_inches='tight',facecolor='white')
print('[saved] F11_flow.png (diamond, parallelograms, 4-column alignment)')
