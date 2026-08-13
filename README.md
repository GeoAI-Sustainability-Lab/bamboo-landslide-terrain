# Terrain-controlled analysis of bamboo forests and rainfall-triggered landslides in Taiwan

Data and code to reproduce the statistics and figures of a multi-event, terrain-controlled
analysis of how the association between bamboo forests and rainfall-triggered landslides
varies with hillslope steepness in Taiwan (2018 to 2025).

Headline result reproduced by this repository: with terrain held constant, the
bamboo-versus-broadleaf interaction odds ratio is 1.034 per degree of slope
(95% CI 1.012 to 1.056), and the spline odds ratio curve crosses 1 near 44 degrees,
just beyond the bamboo distribution edge (95th percentile 41.5 degrees).

## Repository contents

```
data/                  final model-input datasets (derived point samples, see below)
  step1_dataset.csv      20,585 case and background points with terrain covariates
                         (case, ft, slope, north, east, planc, profc, elev, slope2)
  combined_chm.csv       cases and background points joined with canopy height (CHMv2)
  bg_points.csv          background sample points within forest
  landslide_centroids.csv  landslide centroids with year, trigger and area
  bamboo_clip_typed.json   bamboo and mixed-stand polygons clipped for the case study site
expected_outputs/      canonical statistical outputs of the published pipeline
  results.json           all reported estimates (density, niche, odds ratios, spline,
                         interaction specifications, cluster-robust and mixed models,
                         spatial cross-validation, size sensitivity)
  fusion_result.json     tree-ensemble cross-check outputs
scripts/               analysis and figure code (run from inside scripts/)
figures/               figure outputs are written here
```

## Reproduce

```bash
pip install -r requirements.txt
cd scripts
python reproduce_core.py     # refits the headline models from data/ and compares
                             # against expected_outputs/results.json (prints PASS)
python build_f2_f4.py        # slope niche and spline reversal figures
python build_rest.py         # density, odds ratio, robustness, cross-validation figures
python build_chm_fig.py      # greenness and canopy robustness figure
python fusion_figure.py      # tree-ensemble cross-check (fits models, a few minutes)
python build_flow.py         # workflow diagram
```

`reproduce_core.py` refits, from `data/step1_dataset.csv` alone, the terrain-adjusted
interaction between the bamboo indicator and slope (expected 1.0341, 95% CI 1.0123 to
1.0565, p = 0.0021) and the cubic B-spline odds ratio curve with its crossover
(expected about 44 degrees), and checks them against the canonical outputs.

Estimates that require covariates beyond this repository (topographic wetness index,
soil, lithology, pre-event greenness) are recorded with their specifications in
`expected_outputs/results.json`. Figures that draw on restricted source rasters
(the island-wide map panel, the site figure, and the mechanism figure with
orthoimagery) are not rebuilt here; their inputs are available from the providers
listed below.

## Data provenance and acknowledgements

The files in `data/` are derived point samples and clipped geometries produced for this
analysis. The underlying source layers are credited to, and remain subject to the terms
of, their original providers:

- Event-based landslide inventory (2018 to 2025): Agency of Rural Development and
  Soil and Water Conservation (ARDSWC), Ministry of Agriculture, Taiwan, with image
  interpretation by National Cheng Kung University.
- Forest type map: Fourth Forest Resource Inventory, Forestry and Nature Conservation
  Agency, Ministry of Agriculture, Taiwan.
- 20 m digital elevation model: Ministry of the Interior, Taiwan.
- Soil properties: OpenLandMap.
- Surface lithology: USGS Global Ecological Land Units (Sayre et al., 2014),
  250 m regrid by Hengl (2018).
- Pre-event greenness: Copernicus Sentinel-2, European Space Agency.
- Canopy height: Meta and World Resources Institute global canopy height map (CHMv2).
- Orthoimagery for the site figure: National Land Surveying and Mapping Center,
  Ministry of the Interior, Taiwan.

This repository does not redistribute any raw raster, polygon map, or imagery from
these providers.

## Funding

This work was supported by the National Science and Technology Council, Taiwan, under
Grant NSTC 114-2634-F-005-002 for the Smart Sustainable New Agriculture Research Center
(SMARTer) and Grant NSTC 114-2121-M-005-007-MY2.

## Author and citation

Yu-Chun Hsu, Department of Forestry, National Chung Hsing University
(ORCID https://orcid.org/0000-0002-6616-6906).

If you use this repository, please cite it by its URL. A journal article describing
the analysis is in preparation.

## License

Code is released under the MIT License (see LICENSE). The derived data files in
`data/` are provided for research reproducibility with attribution to the providers
listed above.
