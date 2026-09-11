# Terrain-controlled analysis of bamboo forests and rainfall-triggered landslides in Taiwan

Data and code for a multi-event, terrain-controlled analysis of how the association between
bamboo forests and rainfall-triggered landslides varies with hillslope steepness in Taiwan
(event-based landslide inventory 2018 to 2025, national forest type map, 20 m digital elevation
model). The repository lets a reader recompute every reported quantity from the data tables it
contains, and, with the raw source layers obtained from their providers, rebuild those tables
from scratch.

Author: Yu-Chun Hsu, Department of Forestry, National Chung Hsing University, Taichung, Taiwan
(ORCID https://orcid.org/0000-0002-6616-6906). Citation metadata is in `CITATION.cff` and
`.zenodo.json`.

Archived on Zenodo: https://doi.org/10.5281/zenodo.22702156 (concept DOI; it always resolves to
the latest version, and each release has its own version DOI on that page). Cite as: Hsu, Y.-C.
(2026). Terrain-controlled analysis of bamboo forests and rainfall-triggered landslides in
Taiwan: data, code and verification chain (Version 1.1.1) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.22702156

## What the analysis does

Landslide rates cannot be compared across forest types directly, because each forest type
occupies different terrain. The analysis therefore uses a case-control design: rainfall-triggered
landslides (cases, read at the polygon centroid) are compared with random points within forest
(controls) in a logistic regression that holds terrain constant (slope, its square, aspect, plan
and profile curvature, elevation), and the model includes an interaction between forest type and
slope so that the forest-type association is allowed to change with steepness. A cubic B-spline
describes the bamboo-versus-broadleaf odds ratio as a smooth function of slope. Alternative
designs draw the controls from the affected area of each rainfall event, and a long list of
alternative specifications, screens and inference procedures tests whether the result depends
on any single choice.

Main quantities reproduced by this repository (all from the frozen tables, see
`extended_analysis/README.md` for the script behind each):

- interaction odds ratio per degree of slope, bamboo versus broadleaf, island-wide design:
  1.0341 (95% CI 1.0123 to 1.0565); with cluster-robust standard errors by rainfall event
  1.0128 to 1.0559;
- the terrain-adjusted bamboo-versus-broadleaf odds ratio is below 1 with the upper 95% limit
  below 1 from 9.9° to 39.5° (model-based standard errors), 9.3° to 31.4° (cluster-robust) and
  12.2° to 35.8° (cluster bootstrap); the fitted odds ratio equals 1 at 44.1° (bootstrap median
  44.7°, 95% interval 34.5° to 55.8°);
- with controls drawn from the affected area of each rainfall event (event fixed effects):
  1.0315 (1.0114 to 1.0521); with point-level event rainfall added: 1.0316;
- within the observed slope and elevation range of bamboo: odds ratio 0.612 (0.496 to 0.755).

## Repository layout

```
data/                    core analysis, version 1.0: derived point samples
  step1_dataset.csv        20,585 case and control points with terrain covariates
                           (case, ft, slope, north, east, planc, profc, elev, slope2)
  combined_chm.csv         the same points joined with canopy height (Meta/WRI CHM v2)
  bg_points.csv            control points within forest
  landslide_centroids.csv  landslide centroids with year, trigger and area
  bamboo_clip_typed.json   bamboo and mixed-stand polygons clipped for the case study site
expected_outputs/        canonical outputs of the core analysis
  results.json             density by type, slope distributions, odds ratios by type, spline,
                           interaction specifications, cluster-robust and mixed models,
                           spatial cross-validation, size sensitivity
  fusion_result.json       tree-ensemble cross-check of version 1.0
scripts/                 core analysis code, version 1.0 (run from inside scripts/): reproduce_core.py
                           refits the main models from data/ and checks them against expected_outputs/
                           (prints PASS); the build_*.py and fusion_figure.py scripts draw the version 1.0
                           figures into figures/ and are not needed for the numbers
figures/                 output folder of the version 1.0 figure scripts
requirements.txt         packages of the version 1.0 scripts (extended_analysis/requirements.txt is a superset)
extended_analysis/       version 1.1: the extended analysis and the verification chain — frozen
                           analysis tables, stage scripts, JSON registers of every reported quantity,
                           Earth Engine extraction, public soil and lithology layers, result tables.
                           Its README explains the purpose, inputs, outputs and expected values of
                           every step.
```

## Quick start

```bash
pip install -r extended_analysis/requirements.txt
# core analysis (version 1.0)
cd scripts && python reproduce_core.py && cd ..
# extended analysis: the stages that run from the frozen tables (about seven minutes in total)
cd extended_analysis/verify
python verify_models.py
python regional_form_check.py && python allcells_cluster.py && python verify_s9b_chm.py && python verify_s10b_vi.py
python verify_s11_audit.py && python verify_s11b_boot.py && python verify_s11c_fe_cluster.py
python verify_s11e_envelope_lr.py && python verify_s11f_tree_ensemble.py && python verify_s13_reported_quantities.py
python make_tableS9.py && python make_tableS1.py
```

Each script prints the quantities it computes and writes them to a JSON register in
`extended_analysis/verify/`; the registers shipped in the repository are the reference values,
so a rerun can be compared with them line by line. Stages that read the raw source layers
(landslide inventory polygons, forest type map, DEM) are listed separately in
`extended_analysis/README.md` with the folder layout they expect.

## Data provenance and acknowledgements

The files in `data/` and `extended_analysis/verify/` are derived point samples, clipped
geometries and per-point attributes produced for this analysis. The underlying source layers
are credited to, and remain subject to the terms of, their original providers:

- Event-based landslide inventory (2018 to 2025): Agency of Rural Development and Soil and
  Water Conservation (ARDSWC), Ministry of Agriculture, Taiwan, with image interpretation by
  National Cheng Kung University.
- Forest type map: Fourth Forest Resource Inventory, Forestry and Nature Conservation Agency,
  Ministry of Agriculture, Taiwan.
- 20 m digital elevation model: Ministry of the Interior, Taiwan.
- Soil properties: OpenLandMap (Hengl, 2018), 250 m, public Zenodo records.
- Surface lithology: USGS Global Ecological Land Units (Sayre et al., 2014), 250 m regrid by
  Hengl (2018), public Zenodo record.
- Event rainfall: NASA GPM IMERG V07; rainfall climatology: CHIRPS.
- Pre-event vegetation indices: Copernicus Sentinel-2, European Space Agency.
- Canopy height: Meta and World Resources Institute global canopy height map (CHM v2).
- Failure-type screen layers: ARDSWC debris-flow torrent and large-landslide potential layers,
  national and provincial road layer, Water Resources Agency river polygons, ARDSWC event
  inventories 2004 to 2017.

This repository does not redistribute any raw raster, polygon map or imagery from these
providers, other than the public 250 m OpenLandMap soil and USGS lithology layers clipped to
Taiwan (`extended_analysis/data/layers/`, with the download script) and the per-event IMERG
rainfall summaries derived for this study (`extended_analysis/gee/rain/`).

## Funding

National Science and Technology Council, Taiwan: Grant NSTC 114-2634-F-005-002 (Smart
Sustainable New Agriculture Research Center, SMARTer) and Grant NSTC 114-2121-M-005-007-MY2.

## License

Code is released under the MIT License (see LICENSE). The derived data files are provided for
research reproducibility with attribution to the providers listed above.
