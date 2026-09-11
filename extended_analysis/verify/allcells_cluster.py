# -*- coding: utf-8 -*-
"""All-cells polygon representation with polygon-clustered standard errors (class B, deterministic).
Complements S6.polygon_representation, whose all-cells rows used model-based SEs that treat cells
of one polygon as independent.  Output: allcells_cluster.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import pickle, pandas as pd, numpy as np, math, json, warnings
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings('ignore')
recs=pickle.load(open(ROOT+'/verify/poly_cells.pkl','rb'))
COVS=['slope','north','east','planc','profc','elev']; K1=['竹林','闊葉樹林型']; BAMBOO='竹林'
FORM="case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope"
CTRL=pd.read_csv(ROOT+'/verify/controls_xy.csv')
rows=[]
for r in recs:
    if r['ft'] not in K1: continue
    rows.append(pd.DataFrame({v:r[v] for v in COVS}|{'ft':r['ft'],'w':1.0/r['n'],'grp':r['pid']}))
AC=pd.concat(rows,ignore_index=True); AC['case']=1
BC=CTRL[CTRL.ft.isin(K1)][COVS+['ft']].copy(); BC['case']=0; BC['w']=1.0; BC['grp']=['c'+str(i) for i in range(len(BC))]
out={'definition':'every 20 m DEM cell whose centre lies inside a rainfall landslide polygon is a case row; controls as in the published design; '
                  'standard errors clustered by landslide polygon (each control its own cluster)'}
for key,w in [('all_cells_polygon_weight1',AC.w.values),('all_cells_area_weighted',np.ones(len(AC)))]:
    A=AC.copy(); A['w']=w
    D=pd.concat([A,BC],ignore_index=True).reset_index(drop=True); D['bamboo']=(D.ft==BAMBOO).astype(int); D['slope2']=D.slope**2
    g=pd.factorize(D.grp)[0]
    m=smf.glm(FORM,data=D,family=sm.families.Binomial(),freq_weights=D.w.values).fit(cov_type='cluster',cov_kwds={'groups':g})
    b,s=m.params['bamboo:slope'],m.bse['bamboo:slope']
    out[key]=dict(OR=round(math.exp(b),4),ci=[round(math.exp(b-1.96*s),4),round(math.exp(b+1.96*s),4)],p=round(float(2*(1-norm.cdf(abs(b/s)))),5),
                  n_cells=int((D.case==1).sum()),n_polygons=int(D[D.case==1].grp.nunique()),n_ctrl=int((D.case==0).sum()),
                  note=('weights 1/n per cell so each polygon counts once; cluster covariance with fractional frequency weights is approximate' if key.endswith('weight1') else 'each cell weight 1; exact cluster covariance'))
json.dump(out,open(ROOT+'/verify/allcells_cluster.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
