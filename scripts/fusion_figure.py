# -*- coding: utf-8 -*-
"""F10 method-fusion figure (polished, self-contained: palette inlined so it survives container resets).
Real Taiwan case-control data (step1_dataset.csv). Panel (a) tree-ensemble factor importance reproduces
'slope dominates' as in prior susceptibility studies; panel (b) recovers the bamboo x slope interaction
two independent ways (model-free ensemble + parametric logistic) on the same real data.
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# ---- inlined palette D ----
BAMBOO='#2E8B37'; BAMBOO_D='#1F6E29'; BROWN_D='#6E3D18'; INK='#111111'; GREY='#555555'; SLATE='#4B5A62'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.linewidth':0.9,'figure.dpi':400})

d = pd.read_csv('../data/step1_dataset.csv').dropna()
FT=['闊葉樹林型','針葉樹林型','針闊葉樹混淆','竹闊混淆林','竹林','待成林地']
code={k:i for i,k in enumerate(FT)}
d=d[d.ft.isin(FT)].copy(); d['ft_code']=d.ft.map(code)
feats=['slope','elev','north','east','planc','profc','ft_code']
X,y=d[feats].values, d['case'].values
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
clf=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_depth=3,l2_regularization=1.0,random_state=0).fit(Xtr,ytr)
auc=roc_auc_score(yte,clf.predict_proba(Xte)[:,1])
pi=permutation_importance(clf,Xte,yte,n_repeats=10,random_state=0,scoring='roc_auc')
lab={'slope':'Slope','elev':'Elevation','north':'Aspect (N)','east':'Aspect (E)','planc':'Plan curvature','profc':'Profile curvature','ft_code':'Forest type'}
imp=sorted(zip(feats,pi.importances_mean),key=lambda t:-t[1])

med={c:float(np.median(d[c])) for c in ['elev','north','east','planc','profc']}
gs=np.linspace(8,60,80)
def pred(ft):
    R=pd.DataFrame({'slope':gs,'elev':med['elev'],'north':med['north'],'east':med['east'],'planc':med['planc'],'profc':med['profc'],'ft_code':code[ft]})
    return clf.predict_proba(R[feats].values)[:,1]
pB,pO=pred('竹林'),pred('闊葉樹林型'); rr_ml=(pB/(1-pB))/(pO/(1-pO))

sub=d[d.ft.isin(['竹林','闊葉樹林型'])].copy(); sub['bamboo']=(sub.ft=='竹林').astype(float)
Xs=sm.add_constant(pd.DataFrame({'bamboo':sub.bamboo,'slope':sub.slope,'slope2':sub.slope**2,'bamboo_slope':sub.bamboo*sub.slope,'elev':sub.elev,'north':sub.north,'east':sub.east}))
ml=sm.Logit(sub.case.astype(float),Xs).fit(disp=0)
ior=float(np.exp(ml.params['bamboo_slope'])); ise=ml.bse['bamboo_slope']
ici=(float(np.exp(ml.params['bamboo_slope']-1.96*ise)),float(np.exp(ml.params['bamboo_slope']+1.96*ise)))
par=np.exp(ml.params['bamboo']+ml.params['bamboo_slope']*gs)
cross=float(-ml.params['bamboo']/ml.params['bamboo_slope'])   # slope where parametric OR crosses 1

fig,(axA,axB)=plt.subplots(1,2,figsize=(7.5,4.1))
names=[lab[k] for k,_ in imp][::-1]; vals=[v for _,v in imp][::-1]
cols=['#5A6B75' for n in names]
axA.barh(range(len(names)),vals,color=cols,edgecolor='#333',lw=0.6)
axA.set_yticks(range(len(names))); axA.set_yticklabels(names,fontsize=9)
axA.set_xlabel('Permutation importance (AUC drop)')
axA.set_title('(a) Factor importance',loc='left',fontsize=10)
for sp in ('top','right'): axA.spines[sp].set_visible(False)

axB.axhline(1.0,color=BROWN_D,ls='--',lw=1.3); axB.text(59.5,1.04,'broadleaf = 1',color=BROWN_D,ha='right',va='bottom',fontsize=8.6)
axB.plot(gs,rr_ml,color=BAMBOO,lw=2.9,zorder=5)
axB.plot(gs,par,color=BAMBOO_D,lw=1.7,ls=(0,(4,2)),zorder=4)
axB.set_xlabel('Hillslope steepness (degrees)'); axB.set_ylabel('Risk ratio, bamboo vs broadleaf')
axB.set_title('(b) Bamboo and slope interaction, two estimators',loc='left',fontsize=10)
axB.set_ylim(0,max(1.7,np.nanmax(par)*1.02)); axB.set_xlim(8,60)
fig.legend(handles=[Line2D([],[],color=BAMBOO,lw=2.9,label='Tree ensemble (model-free)'),
                    Line2D([],[],color=BAMBOO_D,lw=1.7,ls=(0,(4,2)),label='Parametric logistic interaction')],
           fontsize=8.4,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,0.005))
for sp in ('top','right'): axB.spines[sp].set_visible(False)
plt.tight_layout(rect=[0,0.07,1,1]); plt.savefig('../figures/F10_fusion.png',dpi=400,bbox_inches='tight')
json.dump({'auc':auc,'importance':[(lab[k],v) for k,v in imp],'interaction_or':ior,'interaction_ci':ici,'parametric_crossover':cross},
          open('../figures/fusion_result_rerun.json','w'),ensure_ascii=False,indent=1)
print(f'[saved] F10 AUC={auc:.3f} interOR={ior:.3f}{ici} crossover={cross:.1f}')
print('importance:',[(lab[k],round(v,4)) for k,v in imp])
