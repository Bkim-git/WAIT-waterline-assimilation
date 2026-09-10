"""Sentinel-2 waterlines to an intertidal DTM: extraction, water levels, assimilation.

    python run.py --scenes S2_2024.nc S2_2025.nc --levels levels_2024.csv \\
                  levels_2025.csv --aoi aoi.json --slope-ref dem.tif \\
                  [--exclude mask.json] [--out output]
    python run.py --levels ... --slope-ref ... --reuse    from <out>/waterlines.csv
    python run.py --check                                 the modules' self-checks
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# without `conda activate`, Library/bin is missing from PATH and matplotlib fails
_lib = Path(sys.executable).resolve().parent / "Library" / "bin"
if _lib.is_dir():
    os.environ["PATH"] = str(_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_lib))

import numpy as np

import assimilate as A
import water_level as WL
import waterline as W

plt_rc = {"font.family": "Times New Roman", "font.size": 16,
          "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 14,
          "ytick.labelsize": 14, "legend.fontsize": 13,
          "mathtext.fontset": "stix", "mathtext.default": "it",
          "figure.dpi": 150, "savefig.dpi": 300}


def raster_epsg(path):
    import rasterio

    with rasterio.open(path) as d:
        return d.crs.to_epsg()


def extract(scenes, aoi, exclude, out, epsg):
    lines, quality = [], []
    for f in scenes:
        got, q, e = W.extract_file(f, aoi, exclude)
        if e != epsg:
            raise SystemExit(f"{f} is EPSG:{e}, the slope reference EPSG:{epsg}")
        lines += got
        quality += q
    with open(out / "scene_quality.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "cloud_frac", "valid_frac", "threshold",
                    "separability", "water_frac", "length_m", "reason"])
        for d in quality:
            w.writerow([d["date"]] + [f"{d[k]:.4f}" for k in
                                      ("cloud_frac", "valid_frac", "threshold",
                                       "separability", "water_frac", "length_m")]
                       + [d["reason"]])
    with open(out / "waterlines.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "x_utm", "y_utm"])
        for e in lines:
            for x, y in e["xy"]:
                w.writerow([e["date"], f"{x:.2f}", f"{y:.2f}"])

    n = len(quality)
    print(f"\nextraction: {len(lines)} of {n} scenes accepted "
          f"({100 * len(lines) / n:.0f}%)")
    for r in sorted({d["reason"] for d in quality}):
        c = sum(d["reason"] == r for d in quality)
        print(f"  {r:24s} {c:3d}")
    return lines


def read_waterlines(out):
    per = {}
    with open(out / "waterlines.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per.setdefault(r["date"], []).append((float(r["x_utm"]),
                                                  float(r["y_utm"])))
    return [{"date": d, "xy": np.asarray(per[d])} for d in sorted(per)]


def write_geotiff(path, g, grid, epsg):
    import rasterio
    from rasterio.transform import from_origin

    x0, _, _, y1, nx, ny = grid
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="float32", crs=f"EPSG:{epsg}", nodata=np.nan,
                       transform=from_origin(x0, y1, A.CELL_M, A.CELL_M)) as d:
        d.write(g.astype("float32"), 1)


def fill_holes(dem):
    """(DTM with interior gaps linearly interpolated, cells filled); display only."""
    from scipy.interpolate import griddata
    from scipy.ndimage import binary_fill_holes

    ok = np.isfinite(dem)
    gap = binary_fill_holes(ok) & ~ok
    if not gap.any():
        return dem, 0
    i, j = np.nonzero(ok)
    gi, gj = np.nonzero(gap)
    out = dem.copy()
    out[gi, gj] = griddata((i, j), dem[i, j], (gi, gj), method="linear")
    return out, int(np.isfinite(out[gi, gj]).sum())


def figure(path, lines, dem, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    plt.rcParams.update(plt_rc)
    x0, x1, y0, y1, _, _ = grid
    extent = (x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8.8), constrained_layout=True)

    t = np.array([date2num(np.datetime64(e["date"]).astype("O")) for e in lines])
    norm = plt.Normalize(t.min(), t.max())
    cm = plt.get_cmap("viridis")
    for e, ti in zip(lines, t):
        ax[0].plot(e["xy"][:, 0] / 1e3, e["xy"][:, 1] / 1e3, ".", markersize=0.4,
                   color=cm(norm(ti)))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, ax=ax[0], shrink=0.85)
    cb.set_label("Acquisition date", size=16)
    cb.ax.set_yticks(np.linspace(t.min(), t.max(), 4))
    cb.ax.set_yticklabels([matplotlib.dates.num2date(v).strftime("%Y-%m")
                           for v in np.linspace(t.min(), t.max(), 4)], size=14)
    ax[0].set_title("(a) Extracted waterlines", loc="left")

    shown, n_filled = fill_holes(dem)
    print(f"figure: {n_filled:,} interior gap cells interpolated for display "
          f"(the GeoTIFF is not)")
    im = ax[1].imshow(shown, extent=extent, origin="upper", cmap="terrain",
                      interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[1], shrink=0.85)
    cb.set_label("Elevation (m, MSL)", size=16)
    cb.ax.tick_params(labelsize=14)
    ax[1].set_title("(b) Reconstructed intertidal DTM", loc="left")

    for a in ax:
        a.set_xlim(extent[0], extent[1])
        a.set_ylim(extent[2], extent[3])
        a.set_aspect("equal")
        a.set_xlabel("Easting (km)")
        a.grid(True, linewidth=0.5, alpha=0.5, color="lightgray")
    ax[0].set_ylabel("Northing (km)")
    fig.savefig(path)
    plt.close(fig)


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    epsg = raster_epsg(a.slope_ref)
    lines = (read_waterlines(a.out) if a.reuse
             else extract(a.scenes, a.aoi, a.exclude, a.out, epsg))
    if not lines:
        raise SystemExit("no scene passed the quality gates")

    levels = {}
    for p in a.levels:
        levels.update(WL.read_levels(p))
    lines = WL.attach_levels(lines, levels, WL.slope_sampler(a.slope_ref))
    allz = np.concatenate([e["y"] for e in lines])
    print(f"levels: {len(lines)} waterlines, {len(allz):,} vertices, "
          f"{allz.min():+.2f} to {allz.max():+.2f} m")

    grid = A.build_grid(lines)
    active, _, obs = A.build_domain(lines, grid)
    prior = float(np.median(allz))

    dmax, lmax = A.screening_thresholds(lines, obs, active, prior)
    zs, Ps = A.assimilate(lines, obs, active, prior, delta_max=dmax, l_max=lmax)

    dem = A.reported_dem(zs[-1], Ps[-1], active)
    n = int(np.isfinite(dem).sum())
    print(f"reported DTM at {lines[-1]['date']}: {n:,} cells "
          f"({n * A.CELL_M ** 2 / 1e4:.1f} ha), "
          f"{np.nanmin(dem):+.2f} to {np.nanmax(dem):+.2f} m")

    write_geotiff(a.out / "intertidal_dtm.tif", dem, grid, epsg)
    figure(a.out / "intertidal_dtm.pdf", lines, dem, grid)
    print(f"wrote {a.out / 'intertidal_dtm.tif'} and {a.out / 'intertidal_dtm.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path,
                    help="Sentinel-2 L2A netCDF files with B03, B08 and SCL")
    ap.add_argument("--levels", nargs="+", type=Path,
                    help="per-scene water-level tables")
    ap.add_argument("--aoi", type=Path, help="area of interest, lon/lat polygon")
    ap.add_argument("--slope-ref", type=Path,
                    help="frozen DEM for the foreshore slope, in the scenes' projection")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="polygons over ground that is not shore, projected metres")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--reuse", action="store_true",
                    help="read <out>/waterlines.csv instead of extracting")
    ap.add_argument("--check", action="store_true",
                    help="run the three modules' self-checks and stop")
    a = ap.parse_args()
    if a.check:
        W._check()
        WL._check()
        A._check()
    elif not (a.levels and a.slope_ref and (a.reuse or (a.scenes and a.aoi))):
        ap.error("need --levels and --slope-ref, and --scenes and --aoi unless --reuse")
    else:
        main(a)
