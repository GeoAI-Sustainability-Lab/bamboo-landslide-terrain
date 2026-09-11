# -*- coding: utf-8 -*-
"""Stage 11f - tree-ensemble comparison, numbers only (the figure version lives in r1figs/build_r1_figS3_tree.py).
Purpose: check whether a flexible susceptibility model (gradient-boosted trees) agrees with the logistic models on the
direction of the slope dependence of the bamboo-versus-broadleaf odds ratio, on the SAME sample (5,915 cases, 9,365
controls), with the SAME covariates (slope, northness, eastness, plan and profile curvature, elevation, bamboo
indicator) and the SAME 20 km spatial-block five-fold split (block seed 1) as the spatial cross-validation.
Outputs: held-out AUC of the ensemble and of the two logistic models per fold; permutation importance on the held-out
blocks; predicted odds ratio of bamboo versus broadleaf at median covariates on the 5-60 degree grid.
Registers S11.tree_ensemble_fair in verified_facts_s11.json (asserted against the previous value when present).
Deterministic (class B); a few minutes."""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, math, numpy as np, pandas as pd, statsmodels.formula.api as smf
from patsy import dmatrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
V=ROOT+'/verify/'
CASE=pd.read_csv(V+'cases_annot.csv'); CTRL=pd.read_csv(V+'controls_annot.csv')
COVS=['slope','north','east','planc','profc','elev']; K1=['竹林','闊葉樹林型']
a=CASE[CASE.ft.isin(K1)].copy(); a['case']=1; b=CTRL[CTRL.ft.isin(K1)].copy(); b['case']=0
D=pd.concat([a,b],ignore_index=True); D['bamboo']=(D.ft=='竹林').astype(int); D['slope2']=D.slope**2
BLK=20000.0
D['bx']=np.floor((D.x-D.x.min())/BLK).astype(int); D['by']=np.floor((D.y-D.y.min())/BLK).astype(int)
blocks=D.groupby(['bx','by']).ngroup(); nfold=5
ub=pd.Series(blocks.unique()); fold_of={bk:i%nfold for i,bk in enumerate(ub.sample(frac=1,random_state=1))}
D['fold']=blocks.map(fold_of)
FEATS=COVS+['bamboo']
FRM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
SPL="case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev"
SPL_CV="case ~ bs(slope, knots=(35.43232547287678,), degree=3, lower_bound=0.0, upper_bound=79.36055186759805)*bamboo + north+east+planc+profc+elev"
def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    order=np.argsort(np.concatenate([pos,neg])); ranks=np.empty_like(order,float); ranks[order]=np.arange(1,len(order)+1)
    rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def hgb(): return HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_depth=3,l2_regularization=1.0,random_state=0)
folds=[]; imps=[]
for k in range(nfold):
    tr=D[D.fold!=k]; te=D[D.fold==k]
    clf=hgb().fit(tr[FEATS].values,tr.case.values); p_t=clf.predict_proba(te[FEATS].values)[:,1]
    ml=smf.logit(FRM,data=tr).fit(disp=0); p_l=ml.predict(te)
    ms=smf.logit(SPL_CV,data=tr).fit(disp=0); p_s=ms.predict(te)
    pi=permutation_importance(clf,te[FEATS].values,te.case.values,n_repeats=10,random_state=0,scoring='roc_auc')
    imps.append(pi.importances_mean)
    folds.append(dict(fold=k,n_test=int(len(te)),n_test_bamboo_case=int(((te.bamboo==1)&(te.case==1)).sum()),
                      auc_tree=round(auc(te.case,p_t),3),auc_logistic_linear=round(auc(te.case,p_l),3),auc_logistic_spline=round(auc(te.case,p_s),3)))
    print('fold',k,folds[-1],flush=True)
imps=np.array(imps); imp_mean=imps.mean(0); imp_sd=imps.std(0); order=np.argsort(imp_mean)[::-1]
grid=np.round(np.arange(5.0,60.0+1e-9,0.1),4)
med={c:float(np.median(D[c])) for c in ['north','east','planc','profc','elev']}
clf=hgb().fit(D[FEATS].values,D.case.values)
def pred(bam):
    R=pd.DataFrame({'slope':grid,**med,'bamboo':bam}); return clf.predict_proba(R[FEATS].values)[:,1]
pB,pO=pred(1),pred(0); or_tree=(pB/(1-pB))/(pO/(1-pO))
ml=smf.logit(FRM,data=D).fit(disp=0); or_lin=np.exp(ml.params['bamboo']+ml.params['bamboo:slope']*grid)
ms=smf.logit(SPL,data=D).fit(disp=0); di=ms.model.data.orig_exog.design_info
row=lambda bam: pd.DataFrame({'slope':grid,'bamboo':bam,**med})
d_=np.asarray(dmatrix(di,row(1)))-np.asarray(dmatrix(di,row(0))); or_spl=np.exp(d_@ms.params.values)
p95=float(CTRL[CTRL.ft=='竹林'].slope.quantile(0.95))
at=lambda arr,v: round(float(arr[np.argmin(np.abs(grid-v))]),3)
out=dict(sample=dict(n=int(len(D)),n_case=int(D.case.sum()),n_bamboo_case=int(((D.case==1)&(D.bamboo==1)).sum()),n_ctrl=int((D.case==0).sum())),
         features=FEATS,tree=dict(model='HistGradientBoostingClassifier',max_iter=400,learning_rate=0.06,max_depth=3,l2_regularization=1.0,random_state=0),
         cv=dict(block_km=20,nfold=5,fold_seed=1,folds=folds,
                 mean_auc_tree=round(float(np.mean([f['auc_tree'] for f in folds])),3),mean_auc_logistic_linear=round(float(np.mean([f['auc_logistic_linear'] for f in folds])),3),
                 mean_auc_logistic_spline=round(float(np.mean([f['auc_logistic_spline'] for f in folds])),3)),
         permutation_importance_heldout_mean={FEATS[i]:round(float(imp_mean[i]),4) for i in order},
         permutation_importance_heldout_sd={FEATS[i]:round(float(imp_sd[i]),4) for i in order},
         importance_rank=[FEATS[i] for i in order],
         predicted_or_at={str(v):dict(tree=at(or_tree,v),spline=at(or_spl,v),linear=at(or_lin,v)) for v in (10,20,25,30,35,40,45,50,55,60)},
         tree_or_min=dict(slope=float(grid[int(np.argmin(or_tree))]),OR=round(float(or_tree.min()),3)),
         tree_or_max_below_p95=round(float(or_tree[grid<=p95].max()),3),tree_or_at_p95=at(or_tree,p95),
         tree_crosses_1_within_grid=bool((or_tree>=1).any()),tree_first_slope_ge_1=(float(grid[np.argmax(or_tree>=1)]) if (or_tree>=1).any() else None),
         note='predicted odds ratio at median covariates; the ensemble prediction is piecewise-constant in slope')
F=json.load(open(V+'verified_facts_s11.json'))
old=F['facts'].get('S11.tree_ensemble_fair',{}).get('value')
if old:
    assert old['cv']['folds']==out['cv']['folds'] and old['predicted_or_at']==out['predicted_or_at'] and old['importance_rank']==out['importance_rank'], 'tree comparison differs from the registered value'
F['facts']['S11.tree_ensemble_fair']={'value':out,'definition':'gradient-boosted tree ensemble versus logistic models on the same bamboo-versus-broadleaf sample, covariates and 20 km spatial-block five-fold split: held-out AUC, permutation importance, predicted odds ratio of bamboo versus broadleaf at median covariates','source':'cases_annot.csv + controls_annot.csv','cls':'B'}
json.dump(F,open(V+'verified_facts_s11.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out['cv'],indent=None)[:400]); print(out['importance_rank']); print('registered' + (' (identical to the previous value)' if old else ''))
