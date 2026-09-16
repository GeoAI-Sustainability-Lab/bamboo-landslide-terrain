# Extended analysis and verification chain (version 1.1)

This folder contains the data tables, the scripts and the result registers of the extended
analysis of bamboo forests and rainfall-triggered landslides in Taiwan. It is written so that a
reader can (i) recompute every reported quantity from the tables in this folder, step by step,
and see what each step is for; and (ii) with the raw source layers obtained from their
providers, rebuild the tables themselves. No rendered figure is stored here; the numbers are
the product.

Contents of this README

1. The question and the design in one page
2. Folder layout
3. Installation and the two root paths
4. Quick start: recompute the main quantities from the frozen tables
5. Step-by-step guide: every script, its purpose, inputs, outputs and expected values
6. How to read the registers (`verified_facts*.json`)
7. Stages that need the raw source layers
8. Data dictionary of the analysis tables
9. Expected values at a glance
10. Earth Engine scripts
11. License and attribution

## 1. The question and the design in one page

The question is whether the association between bamboo forests and rainfall-triggered
landslides changes with hillslope steepness once terrain is held constant. Forest types are not
placed at random across terrain, so raw landslide rates by forest type confound the type with
the slopes it occupies. The design is case-control logistic regression: cases are
rainfall-triggered landslide polygons of the 2018 to 2025 event-based inventory, each read at
its centroid; controls are points sampled uniformly at random within forest. The base model
regresses case status on the terrain covariates (slope, slope squared, aspect northness and
eastness, plan and profile curvature, elevation) and a forest-type indicator; the key model adds
the interaction between the bamboo indicator and slope, whose exponentiated coefficient is the
factor by which the bamboo-versus-broadleaf odds ratio changes per degree of slope ("interaction
odds ratio"). A cubic B-spline in slope (four degrees of freedom, one interior knot at the sample
median slope of 35.4°, boundary knots at 0° and 79.4°) describes the same odds ratio as a smooth
function of slope; the slope range over which its pointwise upper 95% confidence limit is below
1, and the slope at which the fitted odds ratio equals 1, are the two summaries of that curve.

Because island-wide controls describe terrain availability rather than exposure to the rainfall
that triggered each event, a second family of designs draws the controls from the affected area
of each rainfall event (event fixed effects, repeated draws from an event-matched control pool,
exact conditional logistic regression stratified by event and spatial block), with point-level
event rainfall from GPM IMERG as a covariate. Around these two families the chain runs failure-
type screens, alternative polygon representations, covariate extensions (wetness index, soil,
lithology, canopy height, pre-event vegetation indices), leave-one-event-out and event-specific
estimates, latitude-band subsets, a cluster bootstrap of the spline curve, growth-form subsets
and a tree-ensemble comparison. Every quantity is written to a register with its definition and
provenance, so that the text of the study can be checked against the code one number at a time.

## 2. Folder layout

```
extended_analysis/
  README.md                  this file
  requirements.txt           Python packages (requirements-gee.txt: the two extra packages of the Earth Engine scripts)
  verify/                    the chain: stage scripts (verify_*.py, make_*.py, *_check.py),
                             the frozen analysis tables they read, and the registers they write
    cases_annot.csv, controls_annot.csv      analysis table (7,454 cases, 13,131 controls), see section 8
    cases_xy.csv, controls_xy.csv            the same points before the screen attributes were attached
    event_cells_5km.json                     5 km grid cells of the affected area of each of the 40 events
    newctrl_pool.csv, pool_annot.csv         event-matched control pool (167,924 forest points) and its attributes
    source_points.csv                        highest DEM cell of every polygon (alternative representation)
    poly_cells.pkl                           every 20 m DEM cell of every rainfall polygon (alternative representation)
    bamboo_points_form.csv                   growth form (running / clumping) of every bamboo-containing point
    ft_join.csv                              forest type at every case centroid and every candidate control point (overlay on the type map)
    covariates_s9.csv                        wetness index, soil, lithology per point
    event_image_dates.csv                    pre- and post-event image dates per event
    rain_exposure_table.csv                  IMERG event rainfall and CHIRPS climatology per bamboo or broadleaf case (5,915 rows)
    DM_event_matched.pkl                     cached design matrix of the event fixed-effects model
    s11b_boot_curves.npy                     the 600 cluster-bootstrap spline curves (seed 20260902)
    verified_facts*.json                     registers (section 6; verified_facts_partial.json = stages 1-4); allcells_cluster.json, bamboo_form.json,
                                             controls_inside_polygons.json, regional_form_check.json: outputs of the scripts of the same name
    *.log                                    console logs of the runs that produced the registers
  tables/                    TableS1_specification_curve.csv (82 specifications), TableS2_coefficients_vif.csv,
                             TableS3_events_cases_loeo.csv, supported_slope_range_by_inference.csv
  gee/                       Earth Engine scripts and their outputs (section 10)
  data/layers/               public 250 m OpenLandMap soil and USGS lithology layers clipped to Taiwan; fetch_soil.py
```

## 3. Installation and the two root paths

Python 3.10 or later. Install the packages with `pip install -r requirements.txt`
(numpy, pandas, scipy, statsmodels, patsy, scikit-learn, geopandas, pyogrio, shapely, rasterio,
pyproj; matplotlib is used only by `make_tableS1.py` for its by-product figure). The chain was
run with Python 3.11, statsmodels 0.14 and scikit-learn 1.4.

Every script resolves its files from two roots, defined in a short header at the top of the
script (after the docstring):

- `ROOT` is this `extended_analysis/` folder. It is derived from the script's own location, so
  nothing needs to be set when the repository is used as checked out; set the environment
  variable `BAMBOO_ROOT` to point elsewhere.
- `RAW` is the folder holding the raw source layers that are not redistributed (section 7).
  Default `ROOT/raw`; set `BAMBOO_RAW` to point elsewhere. Only the stages of section 7 read it.

The tables of the core analysis (version 1.0) are read from the repository's top-level `data/`
folder and `expected_outputs/results.json`; keep the repository structure as is.

Run every script from inside its folder, for example `cd extended_analysis/verify && python
regional_form_check.py`. Each prints what it computes and rewrites its register; the registers
shipped here are the reference values, so a rerun can be compared with them (for example with
`git diff`, since the JSON files are indented and key-ordered).

## 4. Quick start: recompute the main quantities from the frozen tables

About seven minutes on a laptop; no raw layer needed.

```bash
cd extended_analysis/verify
python verify_models.py              # core estimates: interaction 1.0341, spline curve, cluster-robust errors, VIF (2 min)
python regional_form_check.py        # latitude bands and growth forms
python allcells_cluster.py           # all-cells polygon representation, polygon-clustered errors
python verify_s9b_chm.py             # canopy height as a covariate
python verify_s10b_vi.py             # pre-event vegetation indices as covariates
python verify_s11_audit.py           # spline bands under two covariances; event FE + rainfall spline; curve bootstrap
python verify_s11b_boot.py           # the registered cluster bootstrap (600 draws), curves saved
python verify_s11c_fe_cluster.py     # control-point-clustered errors of the event fixed-effects design
python verify_s11e_envelope_lr.py    # odds ratio within the bamboo range; latitude-band likelihood-ratio tests
python verify_s11f_tree_ensemble.py  # gradient-boosted trees versus the logistic models on the same split
python verify_s13_reported_quantities.py   # densities, slope distributions, odds ratios by type, common support
python verify_s15_bg_extent.py       # extent of the 50,000 candidate control points (data/bg_points.csv)
python make_supported_range_table.py # slope range with the upper 95% limit below 1, by procedure and design
python make_tableS2_S3.py            # coefficient and VIF table; rainfall events, cases and leave-one-event-out table
python make_tableS1.py               # the 82 alternative specifications
```

Four more stages also run from the frozen tables: `verify_s10_rain.py` (event rainfall
covariates, 1 min), `verify_crosscheck.py` (independent re-implementations of the main path,
1 min), `verify_s6a2.py` (25 repeated draws from the event-matched control pool, 10 min) and
`verify_s6d.py` (exact conditional logistic regression, about 70 min). The remaining stages,
including `verify_s14_horn.py` (Horn slopes recomputed from the DEM), read the raw source layers
(section 7).

What to look for in the output (literal fragments; some scripts print their whole register as
indented JSON, others print one summary line per quantity):

```
verify_models.py                  published 1.0341 | from step1 1.0341 | from rebuilt 1.0341
                                  spline max |OR| diff vs results.json = 4.92e-05; crossover 44.06 deg; protection band [9.9, 39.5, True]
regional_form_check.py            JSON: "by_region" -> "south": "OR": 1.0506, "ci": [1.0215, 1.0805]; "central": 1.0408; "north": 0.9741
                                        "by_form_original_controls" -> running 1.0472 [1.008, 1.0879]; clumping 1.0303 [1.004, 1.0573]
allcells_cluster.py               JSON: "all_cells_area_weighted" -> "OR": 1.0272, "ci": [1.004, 1.051], "p": 0.02149, "n_cells": 94901
verify_s9b_chm.py                 JSON: "plus_canopy_height" -> "OR": 1.0338 (base 1.0341); "coverage_share_all_points": 0.9999
verify_s10b_vi.py                 JSON: "like_for_like" -> "base_on_covered_subset" 1.0324, "plus_ndvi" 1.0354, "plus_ndmi" 1.0411, "plus_ndvi_ndmi" 1.0406
verify_s11_audit.py               pooled spline: ordinary band [9.9, 39.5] crossover 44.06; event-cluster band [9.3, 31.4] (clusters 40)
                                  FE+rain spline: ordinary band [10.2, 30.5] crossover 41.99; point-clustered band [11.1, 29.7]
verify_s11b_boot.py               registered bootstrap: crossings 584/600, median 44.71 95% [34.5, 55.78] 50% [41.5, 48.14]; no crossing 16 ...; band q97.5<1 [12.2, 35.8]
verify_s11c_fe_cluster.py         FE (no rain) linear: ordinary {'OR': 1.0315, 'ci': [1.0114, 1.0521], 'p': 0.00206} point-clustered {'OR': 1.0315, 'ci': [1.0054, 1.0583], 'p': 0.01779}
verify_s11e_envelope_lr.py        envelope {... "OR": 0.612, "ci": [0.496, 0.755], ... "n_case": 2613, "n_bamboo_case": 126 ...}
                                  LR tests {"slope_only": {"LR": 7.266, "df": 2, "p": 0.0264 ...}, "joint": {"LR": 49.607, "df": 4 ...}}
verify_s11f_tree_ensemble.py      fold 0 ... 'auc_tree': 0.729, 'auc_logistic_linear': 0.725 ... fold 4 ... 'auc_tree': 0.73, 'auc_logistic_linear': 0.72
                                  ['slope', 'north', 'east', 'planc', 'elev', 'profc', 'bamboo']   (permutation-importance rank)
                                  registered (identical to the previous value)
verify_s13_reported_quantities.py densities and odds ratios by type agree with expected_outputs/results.json
```

## 5. Step-by-step guide

The table lists every script in the order the chain was built. "Frozen tables" means the script
reads only files in this folder (plus the top-level `data/` and `expected_outputs/`); "raw" means
it also reads the raw source layers of section 7. Class A quantities are rebuilt from the raw
layers, class B are deterministic recomputations from the frozen tables, class C are Monte Carlo
with fixed seeds (mean and 2.5 to 97.5 percentile range over repeated control draws).

| Script | Purpose (what question it answers) | Reads | Writes | Needs | Time |
| --- | --- | --- | --- | --- | --- |
| `verify_all.py` | Stages 1-4. Rebuilds the analysis table from the raw layers (polygon centroids, terrain by the Zevenbergen-Thorne formulation, forest type by point-in-polygon overlay, control coordinates recovered from `data/bg_points.csv`, the 50,000 candidate points drawn with seed 20260723 as recorded in `expected_outputs/results.json`) and checks it against the version 1.0 table `data/step1_dataset.csv` row by row. Answers: are the tables reproducible from the official layers? | raw layers, `data/*` | `verified_facts_partial.json`, `all_polys.gpkg`, `cases_xy.csv`, `controls_xy.csv`, `ft_join.csv` | raw | 10-30 min |
| `verify_models.py` | Stages 5-7. Reproduces the core estimates (interaction 1.0341, spline curve, cluster-robust errors, size sensitivity, full coefficients, variance inflation factors) and compares them with `expected_outputs/results.json`. | tables, `results.json`, `verified_facts_partial.json` (stages 1-4 register, shipped) | `verified_facts_s5.json` | frozen | 2 min |
| `verify_s6a.py` | Stage 6a. Defines the affected area of each event (5 km cells intersecting the bounding box of any polygon of the event; also 2, 10, 20 km cells and 2, 5 km buffers), fits the event fixed-effects model with island-wide controls inside the affected area, and samples the event-matched control pool (260 points per 5 km cell, seed 20260901). Answers: does the slope dependence survive when controls share the event's rainfall? | raw layers, tables | `verified_facts_s6a.json`, `event_cells_5km.json`, `newctrl_pool.csv` | raw | 10 min |
| `verify_s6a2.py` | Stage 6a-bis. Replaces every single-draw estimate on the new control pool by the mean over 25 independent draws (5, 10, 20 controls per case; each control assigned to one event only). | tables | `verified_facts_s6a2.json` | frozen | 10 min |
| `verify_s6b.py` | Stage 6b. Alternative representations of each landslide polygon (highest DEM cell, steepest cell, upper 20% of cells, polygon mean, every cell, random cell), the elevation gain of the highest cell over the centroid, and the failure-type screen attributes. Answers: does the result depend on where inside the polygon the landslide is read? | raw layers, tables | `verified_facts_s6b.json`, `poly_cells.pkl`, `source_points.csv` | raw | 20 min |
| `verify_s6c.py` | Stage 6c. Failure-type screens (2004-2017 scars, debris-flow torrents, roads, rivers, large-landslide potential areas, area and relief limits), forest-type misclassification simulation (300 draws), model with pure and mixed bamboo as separate levels, event-specific estimates, leave-one-event-out (39 refits), cluster bootstrap of the slope at which the odds ratio equals 1 (600 draws, seed 20260902), common support by slope band, frozen curve statistics. Writes the annotated tables used by all later stages. | raw layers, tables | `verified_facts.json`, `cases_annot.csv`, `controls_annot.csv`, `pool_annot.csv` | raw | 30 min |
| `verify_s6d.py` | Stage 6d. Exact conditional logistic regression (elementary symmetric polynomials, `clogit_exact.py`) with strata = event x 20, 10 or 5 km block, averaged over 10 control draws; replaces the approximate stratified-Cox version. | tables | `verified_facts.json` (updated) | frozen | 70 min |
| `verify_crosscheck.py` | Stage 7. Independent re-implementations of the main path: Newton-Raphson logistic fit, the whole spline path rebuilt without patsy (scipy basis, hand-built design matrix and contrast), delta-method band versus a 20,000-draw simulation, the exact conditional-likelihood implementation versus statsmodels ConditionalLogit and versus Breslow-tied stratified Cox. Answers: could the main numbers be an artefact of one library? | tables | `verified_facts.json` (updated: `S7.*`) | frozen | 3 min |
| `verify_s9_legacy.py` | Stage 9. Covariate extensions: topographic wetness index (240 m block mean of the DEM, D8), OpenLandMap topsoil organic carbon, pH and clay, USGS lithology classes; the interaction with each added and within the siliciclastic sedimentary lithology; logistic GLMM with a random intercept for event (variational Bayes); 20 km spatial-block five-fold cross-validation with the interaction estimated on the training folds and re-estimated inside each held-out fold. | raw DEM, `data/layers`, tables | `verified_facts_s9.json`, `covariates_s9.csv` | raw (DEM only) | 10 min |
| `verify_s9b_chm.py` | Stage 9b. Canopy height (Meta/WRI CHM v2) as a covariate; median canopy height by forest type; coverage. | `data/combined_chm.csv`, tables | `verified_facts_s9b.json` | frozen | 5 s |
| `verify_s10_rain.py` | Stage 10. Event rainfall as a covariate: reads the per-event IMERG rasters (total, maximum 24 h, maximum 72 h over the event window) at every case and control, adds log(1 + mm) to the event fixed-effects model and to the new-pool design; adds the CHIRPS climatology to the island-wide design. Answers: is the slope dependence a rainfall artefact? | `gee/rain/*.tif`, tables | `verified_facts_s10.json`, `rain_exposure_table.csv` | frozen | 1 min |
| `verify_s10b_vi.py` | Stage 10b. Pre-event Sentinel-2 NDVI, NDMI and NBR (80 to 5 days before each polygon's pre-event image; controls in the same window of the nearest case, and alternatively the 2019-2024 median) added to the island-wide model. Answers: is the slope dependence explained by vegetation condition before the event? | `gee/s2_*.csv`, tables | `verified_facts_s10b.json` | frozen | 5 s |
| `verify_s11_audit.py` | Stage 11. The spline band under model-based and under cluster-robust (clusters = events) covariance; the same spline in the event fixed-effects plus rainfall design; a cluster bootstrap of the whole curve. Answers: how far does the range with the upper 95% limit below 1 depend on the inference procedure? | tables | `verified_facts_s11.json` (merged) | frozen | 45 s |
| `verify_s11b_boot.py` | Stage 11b. Re-runs the registered cluster bootstrap (600 draws, seed 20260902) while storing every curve, so that the crossing statistics and the pointwise percentile band come from the same draws; counts draws whose curve never reaches 1 inside 5-60°. | tables | `verified_facts_s11.json`, `s11b_boot_curves.npy` | frozen | 45 s |
| `verify_s11c_fe_cluster.py` | Stage 11c. Standard errors of the event fixed-effects model clustered by control point (the 29,277 rows hold 7,178 distinct control points). | tables | `verified_facts_s11.json` | frozen | 2 s |
| `verify_s11d_reactivation.py` | Stage 11d. Cases whose centroid lies inside a polygon of an earlier 2018-2025 event (repeat or reactivation within the study period) and the interaction without them. | `all_polys.gpkg` (from `verify_all.py`), tables | `verified_facts_s11.json` | raw | 1 min |
| `verify_s11e_envelope_lr.py` | Stage 11e. Odds ratio of bamboo versus broadleaf within the observed range of bamboo (5th to 95th percentiles of slope and elevation of the bamboo controls, unrounded); likelihood-ratio tests of latitude-band heterogeneity. Asserts equality with the registered values. | tables | `verified_facts_s11.json` | frozen | 10 s |
| `verify_s11f_tree_ensemble.py` | Stage 11f. Gradient-boosted trees and the logistic models fitted to the same sample, covariates and 20 km spatial-block five-fold split: held-out AUC, permutation importance, predicted odds ratio of bamboo versus broadleaf at median covariates. Answers: does a flexible model agree on the direction of the slope dependence? | tables | `verified_facts_s11.json` | frozen | 10 s |
| `verify_s12_consistency.py` | Stage 12. Distance from each control to the nearest landslide polygon; agreement between the forest type at the centroid and the area-majority type of the polygon; pure-bamboo area share of bamboo cases. | `all_polys.gpkg`, forest type map, tables | `verified_facts_s12.json` | raw | 1 min |
| `verify_s13_reported_quantities.py` | Stage 13. Descriptive quantities: landslide density per km² by forest type, slope distribution of the controls by type, odds ratio of each type relative to broadleaf (each type against broadleaf in a two-type sample, unadjusted and terrain-adjusted), bamboo share of the bamboo and broadleaf controls in 5° bands. Asserts agreement with `expected_outputs/results.json`. | tables, `results.json` | `verified_facts_s13.json` | frozen | 10 s |
| `allcells_cluster.py` | Every 20 m cell of every polygon as a case row, standard errors clustered by polygon (complements the all-cells rows of stage 6b). | `poly_cells.pkl`, tables | `allcells_cluster.json` | frozen | 10 s |
| `controls_inside_polygons.py` | Controls that fall inside a landslide polygon of the study period (42 of 13,131), by forest type, and the interaction with them removed. | `all_polys.gpkg`, tables | `controls_inside_polygons.json` | raw | 20 s |
| `bamboo_form.py` | Growth form (running C800 / clumping C700) of every bamboo-containing point from the type-map codes; area shares. | forest type map, tables | `bamboo_form.json`, `bamboo_points_form.csv` | raw | 5 min |
| `regional_form_check.py` | Latitude-band subsets (TWD97 northings 2,720,000 and 2,600,000 m) and growth-form subsets of the interaction; pooled heterogeneity test. | tables | `regional_form_check.json` | frozen | 3 s |
| `verify_s14_horn.py` | Stage 14. Slope algorithm sensitivity: Horn (1981) eight-neighbour slopes recomputed from the DEM at the 15,280 analysis points, their difference from the stored Zevenbergen-Thorne slopes, and the main model and the spline refitted with Horn slopes (interaction 1.0363, 95% CI 1.0138 to 1.0592). | raw DEM, `cases_xy.csv`, `controls_xy.csv` | `verified_facts_s14.json` | raw | 30 s |
| `verify_s15_bg_extent.py` | Stage 15. Bounding coordinates of the 50,000 candidate control points and the sampling rectangle they imply. | `data/bg_points.csv` | `verified_facts_s15.json` | frozen | 1 s |
| `make_supported_range_table.py` | Table of the slope range over which the upper 95% limit is below 1, by inference procedure and design, from the stage-11 register. | registers | `tables/supported_slope_range_by_inference.csv` | frozen | 1 s |
| `make_tableS2_S3.py` | Coefficient table of the main model with variance inflation factors before and after centring slope (from the stage-5 register), and the table of the 40 rainfall events with their rainfall windows, cases, bamboo cases and leave-one-event-out estimates (from the stage-6 and stage-10 registers). | registers | `tables/TableS2_coefficients_vif.csv`, `tables/TableS3_events_cases_loeo.csv` | frozen | 1 s |
| `make_tableS1.py` | Table of the 82 alternative specifications (specification curve) from the registers, including the stage-14 Horn refit; also draws the specification-curve figure as a by-product. | registers | `tables/TableS1_specification_curve.csv` (+ a PNG) | frozen | 2 s |
| `clogit_exact.py` | Library: exact conditional logistic likelihood via elementary symmetric polynomials (imported by stage 6d). | - | - | - | - |

Order for a complete rebuild from the raw layers: `verify_all` -> `verify_models` -> `verify_s6a`
-> `verify_s6a2` -> `verify_s6b` -> `verify_s6c` -> `verify_s6d` -> `verify_crosscheck` ->
`verify_s9_legacy` -> `verify_s9b_chm` -> `verify_s10_rain` -> `verify_s10b_vi` ->
`verify_s11_audit` -> `verify_s11b_boot` -> `verify_s11c_fe_cluster` -> `verify_s11d_reactivation`
-> `verify_s11e_envelope_lr` -> `verify_s11f_tree_ensemble` -> `verify_s12_consistency` ->
`verify_s13_reported_quantities` -> `verify_s14_horn` -> `verify_s15_bg_extent`, then `allcells_cluster`, `controls_inside_polygons`, `bamboo_form`,
`regional_form_check`, `make_supported_range_table`, `make_tableS2_S3`, `make_tableS1`. Each stage reads the register of the previous one; the stage-11
scripts append to `verified_facts_s11.json` rather than overwrite it.

## 6. How to read the registers

`verified_facts*.json` has the form `{"facts": {<id>: {...}}, "log": ...}`. Each entry has

- `value`: the number, or a small dictionary of numbers (for example `OR`, `ci`, `p`, `n_case`,
  `n_bamboo_case`; for Monte Carlo quantities `OR_mean`, `OR_range_mc`, `share_p_lt_05`);
- `definition`: what the quantity is, in words;
- `source`: the files it was computed from;
- `cls`: reproducibility class A, B or C as defined in section 5.

Identifiers follow the stage: `S0.*` to `S4.*` in `verified_facts_partial.json` (sources and the table rebuild), `S1.*` to `S7.*` in `verified_facts.json`, `S9.*`, `S10.*`,
`S11.*`, `S12.*`, `S13.*`, `S14.*`, `S15.*` in the files of the same suffix. For example
`verified_facts.json` -> `S6.event_fixed_effects` -> `fe` holds the event fixed-effects
interaction 1.0315 (95% CI 1.0114 to 1.0521) and `S6.crossover_bootstrap` the bootstrap of the
slope at which the odds ratio equals 1 (median 44.71°, 95% interval 34.5° to 55.78°).

## 7. Stages that need the raw source layers

The raw layers are obtained from the providers listed in the top-level README and placed under
`RAW` in the layout recorded in `verified_facts.json` under `S0.sources`:

```
RAW/
  Dataset/01_SOURCE/Terrain_Canopy/Taiwan_DEM_20m/不分幅_全台及澎湖DEM/dem_20m.tif     20 m DEM (EPSG:3826)
  _stage_polygons/<year>/*.shp                     event-based landslide inventories 2018 to 2023 (one folder per year)
  Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113/Event_Inventory_2024_ARDSWC_V7.shp
  Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114/.../Event_Inventory_2025_ARDSWC_V2.shp
  _stage_polygons/auxlayers/                       failure-type screen layers (scars 2004-2017, torrents, roads, rivers, potential areas)
  forest4/f4.shp                                   Fourth Forest Resource Inventory type map
  Dataset/99_INBOX/Downloads_archives/overlay_result.csv   forest type per landslide of the core analysis
```

Set `BAMBOO_RAW` to that folder (or place it at `extended_analysis/raw`). The stages marked
"raw" in section 5 then run; everything else runs from the frozen tables. `verify_all.py`
also writes `all_polys.gpkg` (all 13,900 inventory polygons in one file, not redistributed), which
stages 11d and 12 and `controls_inside_polygons.py` read.

## 8. Data dictionary of the analysis tables

`cases_annot.csv` (7,454 rows) and `controls_annot.csv` (13,131 rows); coordinates in TWD97 TM2
(EPSG:3826).

| Column | Meaning |
| --- | --- |
| `case` | 1 = landslide, 0 = control |
| `ft` | forest type of the type map at the point (闊葉樹林型 broadleaf, 針葉樹林型 conifer, 針闊葉樹混淆 conifer-broadleaf, 竹林 bamboo, 竹闊混淆林 bamboo-broadleaf, 待成林地 immature stand) |
| `slope`, `slope2` | slope in degrees (Zevenbergen-Thorne, 20 m DEM) and its square |
| `north`, `east` | cosine and sine of aspect |
| `planc`, `profc` | plan and profile curvature |
| `elev` | elevation, m |
| `lid` / `bg_lid` | landslide polygon identifier (`<year>_<index>`) / control identifier |
| `year`, `event`, `area_ha` | (cases) inventory year, name of the triggering event, mapped area |
| `x`, `y` | coordinates of the centroid (cases) or of the sampled point (controls) |
| `old_scar` | inside a landslide polygon of the 2004-2017 inventories |
| `d_debris`, `d_road`, `d_river` | distance, m, to the nearest debris-flow torrent, national or provincial road, river polygon |
| `in_largels` | inside a large-landslide potential area |

The bamboo-versus-broadleaf analysis uses the rows with `ft` in {竹林, 闊葉樹林型}: 5,915 cases
(171 bamboo) and 9,365 controls (663 bamboo). `covariates_s9.csv` adds `twi` (wetness index),
`litho`/`litho_name` (lithology class), `soc`, `ph`, `clay` (topsoil). `rain_exposure_table.csv`
gives, per case, the IMERG event totals `total`, `max24`, `max72` (mm) and the CHIRPS
climatology `map_mm`, `annmax_mm`. `gee/s2_cases.csv` and `gee/s2_controls.csv` give NDVI, NDMI,
NBR, the number of cloud-free observations and the pre-event image date used.

## 9. Expected values at a glance

| Quantity | Value | Register |
| --- | --- | --- |
| Interaction odds ratio per degree, island-wide design | 1.0341 (1.0123 to 1.0565), p = 0.002 | `S5.headline_interaction` |
| Same, cluster-robust by event (40 clusters) | 1.0128 to 1.0559 | `S5.event_cluster_robust` |
| Logistic GLMM, random intercept for event | 1.038 (1.017 to 1.059) | `S9.glmm_event_random_intercept` |
| Event fixed effects, controls inside 5 km affected cells | 1.0315 (1.0114 to 1.0521); control-point-clustered 1.0054 to 1.0583 | `S6.event_fixed_effects`, `S11.fe_linear_point_clustered` |
| Event fixed effects + maximum 24 h rainfall | 1.0316 (1.0116 to 1.0520) | `S10.event_fe_with_rainfall` |
| New control pool, 5/10/20 per case, 25 draws | 1.027 / 1.029 / 1.029 | `S6.new_pool_models_mc` |
| Exact conditional logistic, event x 20/10/5 km block | 1.037 / 1.029 / 1.031 | `S6.event_by_block_clogit_exact` |
| Spline: upper 95% limit below 1 | 9.9-39.5° (model-based), 9.3-31.4° (cluster-robust), 12.2-35.8° (bootstrap) | `S11.pooled_spline_bands`, `S11.registered_bootstrap_curves` |
| Slope at which the fitted odds ratio equals 1 | 44.1°; bootstrap median 44.7°, 95% 34.5-55.8° | `S6.curve_statistics`, `S6.crossover_bootstrap` |
| Odds ratio within the bamboo range (8.3-41.5°, 61-1,362 m) | 0.612 (0.496 to 0.755), 2,613 cases | `S11.envelope_exact` |
| Leave-one-event-out, 39 refits | 1.0245 to 1.0388, 38 of 39 with p < 0.05 | `S6.leave_one_event_out` |
| Latitude bands south / central / north | 1.051 / 1.041 / 0.974 | `regional_form_check.json` |
| Growth forms running / clumping | 1.047 / 1.030 | `regional_form_check.json` |
| 20 km spatial-block CV, held-out AUC | 0.68 to 0.75, mean 0.72 | `S9.spatial_block_cv` |
| Tree ensemble, mean held-out AUC | 0.72 (logistic 0.72) | `S11.tree_ensemble_fair` |

## 10. Earth Engine scripts

`gee/gee_rain_windows.py` and `gee/gee_rain_rasters.py` build, for each rainfall event, a
window from the dates in the event name or the typhoon warning period padded by one day
(`event_windows_curated.json`) and export the IMERG V07 total, maximum 24 h and maximum 72 h
accumulation over that window as rasters (`gee/rain/*.tif`, 0.1°) together with the CHIRPS
2018-2025 climatology. `gee/gee_s2_vi2.py`, `gee_s2_ctrl.py` and `gee_s2_ctrl_multiyear.py`
extract the Sentinel-2 Level-2A cloud-free median NDVI, NDMI and NBR at every case and control
(cases: 80 to 5 days before the pre-event image date of their polygon; controls: the same window
of the nearest case, and alternatively the 2019-2024 median). Their outputs are stored in
`gee/`, so the analysis does not require re-running them (`gee_s2_vi.py` is the first, per-case
version of the extraction that `gee_s2_vi2.py` replaced). To re-run them you need your own Google
Earth Engine project and service-account key: install `requirements-gee.txt`, then set
`GEE_SERVICE_ACCOUNT_KEY` (path to the key file) and `GEE_PROJECT` (project id). No credential is
stored in this repository.

## 11. License and attribution

Code: MIT License (see the repository LICENSE). The derived tables are provided for research
reproducibility; the source layers are credited in the top-level README and remain subject to
the terms of their providers. Contact: Yu-Chun Hsu, Department of Forestry, National Chung Hsing
University (ORCID 0000-0002-6616-6906).
