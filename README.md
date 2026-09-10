# Intertidal DTM from satellite waterlines

Sentinel-2 waterlines, a water level for every vertex, and a sequential per-cell
assimilation that turns the two into a digital terrain model of the intertidal zone.

| file | what it does |
|---|---|
| `waterline.py` | one scene to one waterline: NDWI, Otsu, quality gates, contour |
| `water_level.py` | the level each vertex carries: tide + non-tidal residual + wave setup |
| `assimilate.py` | dated waterlines to a DTM: forecast, Huber update, observation screening |
| `run.py` | the three chained, from scene files to the DTM |

## Run

    python run.py --check
    python run.py --scenes S2_*.nc --levels levels_*.csv --aoi aoi.json \
                  --slope-ref dem.tif [--exclude mask.json] [--out output]
    python run.py --levels levels_*.csv --slope-ref dem.tif --reuse

`--check` runs the modules' self-checks, which need no data. `--reuse`
re-assimilates from `<out>/waterlines.csv` without extracting again.

Writes to `--out`:

| file | |
|---|---|
| `scene_quality.csv` | every scene, its diagnostics and why it was rejected |
| `waterlines.csv` | date, easting, northing of every accepted waterline |
| `intertidal_dtm.tif` | the reported DTM, 10 m, in the scenes' projection |
| `intertidal_dtm.pdf` | the waterlines and the DTM; gaps enclosed by reported cells are linearly interpolated for display only |

## Inputs

| argument | format |
|---|---|
| `--scenes` | netCDF, one or more: Sentinel-2 L2A `B03`, `B08`, `SCL` on `x`, `y`, `t`, with a `crs` variable carrying `crs_wkt` |
| `--levels` | CSV, one or more, a row per scene: `date` (YYYY-MM-DD), `tide_m`, `residual_m`, `wave_factor_m` = sqrt(H0 L0) in m (empty: no setup) |
| `--aoi` | JSON `{"coordinates": [[[lon, lat], ...]]}` |
| `--slope-ref` | GeoTIFF DEM in the scenes' projection, read only for the foreshore slope in the setup |
| `--exclude` | optional JSON `{"polygons": [[[x, y], ...], ...]}` in projected metres, over ground that is not shore |

The tide and residual are expected on local mean sea level, e.g. from TCARI
(Hess 2003); a scene with no row in `--levels` is not assimilated.

## Parameters

ALL_CAPS constants at the top of each module; the main ones:

| constant | value | |
|---|---|---|
| `INVALID_SCL` | 0, 1, 3, 8, 9 | SCL classes masked; thin cirrus (10) is kept |
| `CELL_M` | 10 m | grid spacing |
| `R_OBS_M2` | 0.09 m² | observation variance |
| `Q_RATE_M2_PER_YEAR` | 0.09 m² yr⁻¹ | process noise |
| `P0_M2` | 9 m² | prior variance |
| `HUBER_C` | 1.345 | Huber threshold, in units of the innovation's scale |
| `DIFFUSIVITY_M2_PER_YEAR` | 50 m² yr⁻¹ | forecast diffusion (regularisation) |
| `MAX_ANALYSIS_SD_M` | 1.0 m | a cell is reported where sqrt(P) is below it |
| `REJECT_DIFF_PCT` | 99 | percentile of the run's innovations setting delta_max |
| `REJECT_DIST_PCT` | 99 | percentile of the run's new-cell distances setting L_max |

## Installation

    conda env create -f environment.yml
    conda activate intertidal-dtm

Python 3.11 with numpy, scipy, xarray, netCDF4, pyproj, rasterio and matplotlib,
at the versions pinned in `environment.yml`.

## References

## License

MIT, see `LICENSE`.
