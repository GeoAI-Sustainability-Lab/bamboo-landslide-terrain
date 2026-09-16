# -*- coding: utf-8 -*-
"""Tables S2 and S3 of the supplementary material, written from the registers so they cannot drift from them.
  Table S2  full coefficients of the main model (Eq. 3) and variance inflation factors before and after centring slope
  Table S3  rainfall events of the inventory: rainfall window, cases and bamboo cases of the bamboo-versus-broadleaf
            comparison, and the leave-one-event-out estimate of the interaction after removing the event
Output: tables/TableS2_coefficients_vif.csv, tables/TableS3_events_cases_loeo.csv"""
import os as _os
ROOT=_os.environ.get('BAMBOO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # the extended_analysis/ folder
RAW=_os.environ.get('BAMBOO_RAW', _os.path.join(ROOT,'raw'))  # raw source layers, obtained from their providers (not redistributed)
def SRC(p):  # resolve a source path recorded in the registers to this checkout
    p=p.replace('/mnt/user-data/uploads/文章發想與實踐',RAW).replace('/home/claude/forest4',RAW+'/forest4').replace('/home/claude/results.json',ROOT+'/../expected_outputs/results.json').replace('/home/claude/verify/',ROOT+'/verify/')
    return p.replace('/home/claude/',ROOT+'/../data/')
import json, csv
F=json.load(open('verified_facts.json'))['facts']; S10=json.load(open('verified_facts_s10.json'))['facts']
OUT=ROOT+'/tables/'
# ---- Table S2
coef=F['S5.full_coefficients']['value']; vif=F['S5.vif']['value']
TERM={'Intercept':'Intercept','slope':'slope (deg)','slope2':'slope squared','north':'northness','east':'eastness','planc':'plan curvature','profc':'profile curvature','elev':'elevation (m)','bamboo':'bamboo indicator','bamboo:slope':'bamboo indicator by slope interaction'}
VKEY={'slope':'slope','slope2':'slope2','north':'north','east':'east','planc':'planc','profc':'profc','elev':'elev','bamboo':'bamboo','bamboo:slope':'bam_slope'}
rows=[]
for k,v in coef.items():
    vk=VKEY.get(k)
    rows.append(dict(term=TERM[k],coefficient=v['beta'],se=v['se'],z=v['z'],p=v['p'],odds_ratio=v['OR'],ci_low=v['lo'],ci_high=v['hi'],
                     vif_uncentred=(vif['raw'].get(vk,'') if vk else ''),vif_slope_centred=(vif['centred'].get(vk if vk in vif['centred'] else vk+'_c','') if vk else '')))
with open(OUT+'TableS2_coefficients_vif.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(rows)
# ---- Table S3
W=S10['S10.rainfall_windows']['value']; ev=F['S6.event_structure']['value']['per_event']; lo={r['dropped']:r for r in F['S6.leave_one_event_out']['value']['rows']}
rows=[]
for k in sorted(W, key=lambda k: W[k]['window'][0]):
    e=ev.get(k); r=lo.get(k)
    rows.append(dict(inventory_event_label=k,window_start=W[k]['window'][0],window_end=W[k]['window'][1],
                     cases=(e['n_case'] if e else ''),bamboo_cases=(e['n_bamboo'] if e else ''),
                     OR_per_degree_without_event=(r['OR'] if r else ''),p_without_event=(round(r['p'],4) if r else '')))
assert len(rows)==40 and sum(1 for r in rows if r['cases']!='')==39
with open(OUT+'TableS3_events_cases_loeo.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(rows)
print('Table S2 rows',len(coef),'| Table S3 rows',len(rows))
