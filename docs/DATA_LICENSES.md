# Data License Audit (2026-09-17)

Everything under `data/` is gitignored — datasets are local-only artifacts. This file records redistribution terms for anything that would ship or be published with the project. "Verified" means a license file or PROVENANCE.json exists on disk; "flagged" means terms are uncertain and must be resolved before public release.

## Verified

| Dataset | Terms | Source |
|---|---|---|
| Pleiades DEMs, South Lhonak (pre/post 2023 GLOF) | CC BY 4.0 | Zenodo 10.5281/zenodo.13124662 (Gascoin & Cook) |
| Global glacial lake bathymetry compilation | CC BY 4.0 | Zenodo 10.5281/zenodo.10201073 |
| Bathymetry, 16 greater-Himalaya lakes (Zhang 2023) | CC BY 4.0 | PROVENANCE.json |
| In-situ bathymetry, 4 western-Himalaya lakes (Das 2025) | CC BY 4.0 | PROVENANCE.json |
| Copernicus DEM GLO-30 tiles | Copernicus Data Licence (free & open, attribution) | PROVENANCE.json |
| HMAGLOFDB (GLOF database, High Mountain Asia) | CC BY 4.0 | Bundled README + LICENSE; ESSD 10.5194/essd-15-3941-2023 |
| S1GFloods (+ `_extracted` chips) | CC BY 4.0 | Bundled README frontmatter |
| Kuro Siwo (S1 GRD/SLC flood dataset) | Annotations CC BY; Sentinel imagery free & open; benchmark code MIT | github.com/Orion-AI-Lab/KuroSiwo |
| Sentinel-1/2 SAFE archives (raw + datasets) | Copernicus free & open | CDSE/ESA |
| SRTM 30 m | US Government work (public domain) | USGS/NASA |

## Flagged — resolve before public release

| Dataset | Concern |
|---|---|
| RGI2000-v7.0 zips (G-13/14/15) | RGI license permits scientific/educational use with citation; redistribution of derived or raw data needs a terms check (rgi.readthedocs.io license page) |
| `glacial_lake_2022-2024/` shapefiles | No provenance file; likely ICIMOD inventory — verify source and terms |
| `Hi-MAG database.zip` | ICIMOD Hi-MAG — terms on icimod.org/rds, not recorded |
| `GLOF database of High Mountain Asia.zip` | Probably HMAGLOFDB duplicate — dedupe |
| `FloodPlanet.zip` | Research dataset (S1/S2/L8/PlanetScope + labels); license not recorded on disk — check the publication/repo terms. **PlanetScope scenes are commercial data** — redistribution likely restricted even if labels are open |
| `Shisper_Glacier_lake_data.zip` | Provenance unknown |
| `nepal-*.gpkg.zip`, `north-eastern-zone-*.gpkg.zip` | OSM-derived (Geofabrik-style naming) — ODbL: attribution + share-alike required if published |
| `models/checkpoints/` | Checkpoints trained on Kuro Siwo inherit CC BY annotation terms — attribution required, no redistribution blocker found |

## Himalayan SAR label gap — checked 2026-09-17

The plan was to look for high-altitude labelled SAR water/flood data on disk before acquiring new labels. Result: **none exists**.

- **S1GFloods**: 5,360 A/B/Label chip triplets (256×256 PNG), but **no geolocation metadata** — chips are `image_N.png` with no coordinates, and the RGB channel encoding is undocumented. Cannot select in-domain chips; unusable as-is.
- **FloodPlanet/Nepal**: 9 S1 tiles (2-band VV/VH, EPSG:4326) + labels — but all at **26.5–26.9°N, 83.3–83.6°E (Terai plains, ~100–200 m elevation)**. Same lowland OOD regime as Kuro Siwo; does not cover glacier/snow terrain.
- Everything else with labels is the same story: flood datasets are calibrated on lowland riverine events.

**Remaining options** (in rough order of feasibility):
1. Weak-label domain adaptation: use the verified Imja masks (`obs-*_expansion_mask.tif` union, SCL-gated optical evidence) as pseudo-labels over our own descending S1 pair — circular vs the deterministic path but legitimate for backbone domain adaptation.
2. Hand-label a small set of Imja-region SAR chips (water vs glacier/moraine classes) against the clear Nov S2 scene.
3. Acquire a labelled high-altitude SAR dataset if one exists (none found in this audit).
