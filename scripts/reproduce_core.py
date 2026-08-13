# -*- coding: utf-8 -*-
"""Reproduce the headline statistics of the study from the packaged model-input data.

Run from the scripts/ directory:  python reproduce_core.py

Refits, from data/step1_dataset.csv alone
  1. the terrain-adjusted interaction between the bamboo indicator and slope
     (the headline per-degree interaction odds ratio), and
  2. the cubic B-spline odds ratio curve of bamboo versus broadleaf over slope,
     with the slope at which the curve crosses 1,
and compares both against expected_outputs/results.json.
"""
import json, math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrix

d = pd.read_csv("../data/step1_dataset.csv").dropna()
d = d[d.ft.isin(["闊葉樹林型", "竹林"])].copy()
d["bamboo"] = (d.ft == "竹林").astype(int)
d["slope2"] = d.slope ** 2
E = json.load(open("../expected_outputs/results.json"))

# 1. headline interaction, identical specification to the published pipeline
m = smf.logit("case ~ slope+slope2+north+east+planc+profc+elev+bamboo+bamboo:slope", data=d).fit(disp=0)
b, se = m.params["bamboo:slope"], m.bse["bamboo:slope"]
got = {"OR": round(math.exp(b), 4),
       "ci": [round(math.exp(b - 1.96 * se), 4), round(math.exp(b + 1.96 * se), 4)],
       "p": float(m.pvalues["bamboo:slope"])}
exp_t = E["E5_interaction_specs"]["specs"]["terrain"]
print("1. Interaction odds ratio per degree, bamboo vs broadleaf (terrain specification)")
print(f"   reproduced  OR {got['OR']}  95% CI {got['ci']}  p {got['p']:.4f}")
print(f"   expected    OR {exp_t['OR']}  95% CI {exp_t['ci']}  p {exp_t['p']:.4f}")
ok1 = abs(got["OR"] - exp_t["OR"]) < 1e-3

# 2. cubic B-spline odds ratio curve and its crossover, identical specification
sp = smf.logit("case ~ bs(slope, df=4)*bamboo + north+east+planc+profc+elev", data=d).fit(disp=0)
di = sp.model.data.design_info
med = {c: float(np.median(d[c])) for c in ["north", "east", "planc", "profc", "elev"]}
grid = np.linspace(float(d.slope.quantile(0.02)), float(d.slope.quantile(0.98)), 200)
def row(bam):
    return pd.DataFrame({"slope": grid, "bamboo": bam, **{k: v for k, v in med.items()}})
X1 = np.asarray(dmatrix(di, row(1))); X0 = np.asarray(dmatrix(di, row(0)))
diff = X1 - X0
lo = diff @ sp.params.values
orc = np.exp(lo)
cross = float(grid[np.argmin(np.abs(orc - 1))])
exp_cross = float(E["E4_spline"]["crossover_slope"])
print("2. Spline odds ratio curve, bamboo vs broadleaf")
print(f"   reproduced  crossover of OR = 1 at {cross:.1f} degrees")
print(f"   expected    crossover of OR = 1 at {exp_cross:.1f} degrees")
ok2 = abs(cross - exp_cross) < 1.0

# 3. bamboo slope niche summary from the background points
bg = pd.read_csv("../data/step1_dataset.csv").dropna()
bb = bg[(bg.ft == "竹林") & (bg.case == 0)].slope
print("3. Bamboo slope niche from background points")
print(f"   reproduced  median {bb.median():.1f}  p95 {bb.quantile(0.95):.1f} degrees")
print(f"   expected    median 25.8  p95 41.5 degrees (manuscript values)")

print()
print("PASS" if (ok1 and ok2) else "CHECK DIFFERENCES ABOVE")
