# -*- coding: utf-8 -*-
"""Slope range over which the pointwise upper 95% limit is below 1, and the slope at which the fitted odds ratio equals 1, under
alternative inference procedures and designs (the values quoted in Sections 3.2 and 3.3 of the text). Built from
verified_facts_s11.json so it cannot drift from the register. Output: tables/supported_slope_range_by_inference.csv"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, csv
S=json.load(open(ROOT+'/verify/verified_facts_s11.json'))['facts']
ps=S['S11.pooled_spline_bands']['value']; fe=S['S11.fe_rain_spline_bands']['value']; bt=S['S11.registered_bootstrap_curves']['value']
rows=[]
def add(design,inference,band,cross,note=''):
    rows.append(dict(design=design,inference=inference,lower_deg=band[0],upper_deg=band[1],crossing_deg=cross,note=note))
add('island-wide case-control, cubic B-spline (Fig. 6)','model-based covariance, pointwise delta method',ps['ordinary']['supported_band'],ps['ordinary']['crossover_deg'],'main design, model-based covariance')
add('island-wide case-control, cubic B-spline (Fig. 6)','cluster-robust covariance, rainfall events as clusters and the controls as one further cluster, pointwise',ps['event_cluster_robust']['supported_band'],ps['event_cluster_robust']['crossover_deg'],'same fitted curve; only the covariance differs')
add('island-wide case-control, cubic B-spline (Fig. 6)',f"cluster bootstrap by event, 600 draws, pointwise 2.5 to 97.5 percentiles of the refitted curves",bt['band_q975_below_1'],bt['crossover']['median'],f"median over the {bt['crossings_found']} draws whose fitted odds ratio equals 1 inside 5 to 60 degrees; 95% interval {bt['crossover']['ci95'][0]} to {bt['crossover']['ci95'][1]}; {bt['no_crossing']} draws stay below 1 throughout")
add('event fixed effects + log max 24-h IMERG rainfall, same spline knots','model-based covariance, pointwise',fe['spline_ordinary']['supported_band'],fe['spline_ordinary']['crossover_deg'],f"{fe['linear_interaction']['n_rows']:,} rows, {fe['linear_interaction']['n_distinct_control_points']:,} distinct control points")
add('event fixed effects + log max 24-h IMERG rainfall, same spline knots','covariance clustered by control point, pointwise',fe['spline_control_point_clustered']['supported_band'],fe['spline_control_point_clustered']['crossover_deg'],'')
with open(ROOT+'/tables/supported_slope_range_by_inference.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
