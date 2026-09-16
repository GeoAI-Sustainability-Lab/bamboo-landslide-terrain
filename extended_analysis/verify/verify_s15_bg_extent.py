# -*- coding: utf-8 -*-
"""Stage 15 - extent of the island-wide control sample. Reads data/bg_points.csv (the 50,000 candidate points distributed
with the repository) and registers its bounding coordinates, so that Text S1 can state the sampling rectangle from the
file rather than from memory. Deterministic (class A). Output: verified_facts_s15.json"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, pandas as pd
V=ROOT+'/verify/'
b=pd.read_csv(ROOT+'/../data/bg_points.csv')
ext=dict(n_points=int(len(b)),x_min=float(b.x.min()),x_max=float(b.x.max()),y_min=float(b.y.min()),y_max=float(b.y.max()))
rect=dict(x0=int(round(ext['x_min'],-3)),x1=int(round(ext['x_max'],-3)),y0=int(round(ext['y_min'],-3)),y1=int(round(ext['y_max'],-3)))
out={'S15.bg_points_extent':{'value':dict(**ext,rectangle_rounded_to_1km=rect,crs='EPSG:3826 (TWD97 TM2)'),
     'definition':'bounding coordinates of the 50,000 candidate control points and the sampling rectangle they imply (rounded to 1 km)',
     'source':'bg_points.csv','cls':'A'}}
json.dump({'facts':out,'log':'verify_s15_bg_extent.py'},open(V+'verified_facts_s15.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False))
