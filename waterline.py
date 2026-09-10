"""Waterline extraction from Sentinel-2 L2A scenes.

One scene gives at most one waterline: the contour of NDWI = (B03 - B08) /
(B03 + B08) at the scene's own Otsu threshold, kept where land lies beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# L2A scene classes masked: 0 no data, 1 saturated, 3 cloud shadow, 8-9 cloud.
# 10 (thin cirrus) is kept: masking it discarded scenes whose shore is visible.
INVALID_SCL = (0, 1, 3, 8, 9)

MIN_DENOM = 100.0            # B03 + B08 floor in L2A DN; below it NDWI blows up
NDWI_CLIP = 1.0              # |NDWI| <= 1 by definition

MAX_AOI_CLOUD = 0.10         # fraction of AOI pixels flagged unusable
MIN_VALID_FRAC = 0.80        # fraction of AOI pixels left after masking
MIN_SEPARABILITY = 0.70      # Otsu between-class / total variance
THR_RANGE = (-0.40, 0.40)    # plausible water/land threshold
WATER_FRAC_RANGE = (0.05, 0.95)   # both classes must be present
MIN_SHORELINE_M = 1500.0     # summed over the scene, not per segment
MIN_SEGMENT_M = 100.0        # shortest piece worth recording

# Water over bright sand pulls NDWI towards zero, so a contour must have land
# (NDWI <= SHORE_LAND_MAX) at one of these distances, not merely water.
SHORE_ADJACENT_M = (20.0, 40.0)
SHORE_LAND_MAX = -0.10


def otsu(values, nbins=256):
    """(threshold, separability) of a 1-D sample; separability 1 means bimodal."""
    hist, edges = np.histogram(values, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float) / hist.sum()

    w0 = np.cumsum(w)
    m = np.cumsum(w * centres)
    mt = m[-1]
    denom = w0 * (1.0 - w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mt * w0 - m) ** 2 / denom, 0.0)

    k = int(np.nanargmax(sigma_b))
    sigma_t = float(np.sum(w * (centres - mt) ** 2))
    sep = float(sigma_b[k] / sigma_t) if sigma_t > 0 else 0.0
    return float(centres[k]), sep


def polyline_length(seg):
    return float(np.hypot(*np.diff(seg, axis=0).T).sum())


def polygon_mask(x, y, ring_xy):
    """Boolean (ny, nx) mask of pixel centres inside a polygon ring."""
    from matplotlib.path import Path as MplPath

    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return MplPath(ring_xy).contains_points(pts).reshape(xx.shape)


def exclude_polygons(path):
    """Hand-drawn polygons over ground that is not shore, in projected metres."""
    from matplotlib.path import Path as MplPath

    if path is None or not Path(path).exists():
        return []
    polys = json.loads(Path(path).read_text())["polygons"]
    return [MplPath(np.asarray(q)) for q in polys]


def _reaches_land(seg, ndwi, x, y):
    """Lowest median NDWI either side of `seg`, along its normal."""
    dx = abs(x[1] - x[0])
    ny, nx = ndwi.shape
    t = np.gradient(seg, axis=0)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    ln = np.hypot(n[:, 0], n[:, 1])
    if not (ln > 0).any():
        return np.inf
    n = n[ln > 0] / ln[ln > 0, None]
    p = seg[ln > 0]
    best = np.inf
    for side in (+1.0, -1.0):
        for d in SHORE_ADJACENT_M:
            q = p + side * d * n
            j = np.clip(((q[:, 0] - x[0]) / dx).round().astype(int), 0, nx - 1)
            i = np.clip(((y[0] - q[:, 1]) / dx).round().astype(int), 0, ny - 1)
            v = ndwi[i, j]
            v = v[np.isfinite(v)]
            if len(v):
                best = min(best, float(np.median(v)))
    return best


def _split_outside(seg, masks):
    """`seg` cut into its runs of two or more vertices outside `masks`."""
    if not masks:
        return [seg]
    inside = np.zeros(len(seg), bool)
    for q in masks:
        inside |= q.contains_points(seg)
    if not inside.any():
        return [seg]
    out, i = [], 0
    while i < len(seg):
        if inside[i]:
            i += 1
            continue
        j = i
        while j < len(seg) and not inside[j]:
            j += 1
        if j - i >= 2:
            out.append(seg[i:j])
        i = j
    return out


def _contours(x, y, field, level, ndwi, masks):
    """Iso-`level` polylines surviving the mask, the length and the land test."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    fig = plt.figure()
    try:
        cs = fig.gca().contour(x, y, field, levels=[level])
        segs = [s for s in cs.allsegs[0] if len(s) >= 2]
    finally:
        plt.close(fig)
    segs = [t for s in segs for t in _split_outside(s, masks)]
    segs = [s for s in segs if polyline_length(s) >= MIN_SEGMENT_M]
    segs = [s for s in segs if _reaches_land(s, ndwi, x, y) <= SHORE_LAND_MAX]
    return sorted(segs, key=polyline_length, reverse=True)


def extract_scene(b03, b08, scl, inside, x, y, masks=()):
    """(waterline vertices or None, diagnostics) for one scene."""
    d = {"cloud_frac": np.nan, "valid_frac": np.nan, "threshold": np.nan,
         "separability": np.nan, "water_frac": np.nan, "length_m": np.nan,
         "reason": ""}

    n_aoi = int(inside.sum())
    cloudy = np.isin(scl, INVALID_SCL) & inside
    d["cloud_frac"] = cloudy.sum() / n_aoi

    denom = b03 + b08
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(np.abs(denom) > MIN_DENOM, (b03 - b08) / denom, np.nan)
    ndwi = np.where(np.abs(ndwi) <= NDWI_CLIP, ndwi, np.nan)

    valid = inside & ~cloudy & np.isfinite(ndwi)
    d["valid_frac"] = valid.sum() / n_aoi

    if d["cloud_frac"] > MAX_AOI_CLOUD:
        d["reason"] = "cloud"
        return None, d
    if d["valid_frac"] < MIN_VALID_FRAC:
        d["reason"] = "few valid pixels"
        return None, d

    thr, sep = otsu(ndwi[valid])
    d["threshold"], d["separability"] = thr, sep
    d["water_frac"] = float((ndwi[valid] > thr).mean())

    if not (THR_RANGE[0] <= thr <= THR_RANGE[1]):
        d["reason"] = "threshold out of range"
        return None, d
    if sep < MIN_SEPARABILITY:
        d["reason"] = "not bimodal"
        return None, d
    if not (WATER_FRAC_RANGE[0] <= d["water_frac"] <= WATER_FRAC_RANGE[1]):
        d["reason"] = "single class"
        return None, d

    # masked field, so no line follows a mask edge; land probed on the unmasked one
    segs = _contours(x, y, np.where(valid, ndwi, np.nan), thr, ndwi, masks)
    if not segs:
        d["reason"] = "no contour at the shore"
        return None, d

    d["length_m"] = sum(polyline_length(s) for s in segs)
    if d["length_m"] < MIN_SHORELINE_M:
        d["reason"] = "shoreline too short"
        return None, d

    d["reason"] = "accepted"
    return np.concatenate(segs), d


def extract_file(nc_path, aoi_path, exclude_path=None, progress=True):
    """(waterlines, quality, epsg) for every scene in one B03/B08/SCL netCDF."""
    import xarray as xr
    from pyproj import CRS, Transformer

    ds = xr.open_dataset(nc_path)
    x, y = ds["x"].values, ds["y"].values
    epsg = CRS.from_wkt(ds["crs"].attrs["crs_wkt"]).to_epsg()

    ring = np.asarray(json.loads(Path(aoi_path).read_text())["coordinates"][0])
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    inside = polygon_mask(x, y, np.column_stack(tf.transform(ring[:, 0],
                                                             ring[:, 1])))
    masks = exclude_polygons(exclude_path)

    lines, quality = [], []
    for i in range(ds.sizes["t"]):
        seg, d = extract_scene(ds.B03.isel(t=i).values.astype("f8"),
                               ds.B08.isel(t=i).values.astype("f8"),
                               ds.SCL.isel(t=i).values, inside, x, y, masks)
        d["date"] = str(ds["t"].values[i])[:10]
        quality.append(d)
        if seg is not None:
            lines.append({"date": d["date"], "xy": seg})
        if progress:
            print(f"  {d['date']}  {d['reason']}", flush=True)
    ds.close()
    return lines, quality, epsg


def _check():
    rng = np.random.default_rng(0)
    two = np.concatenate([rng.normal(-0.5, 0.15, 5000),
                          rng.normal(+0.5, 0.15, 5000)])
    thr, sep = otsu(two)
    assert abs(thr) < 0.05, thr
    assert sep > MIN_SEPARABILITY, sep
    _, sep_one = otsu(rng.normal(0.0, 1.0, 10000))
    assert sep_one < MIN_SEPARABILITY, sep_one

    line = np.column_stack([np.arange(11.0), np.zeros(11)])
    assert abs(polyline_length(line) - 10.0) < 1e-9

    from matplotlib.path import Path as MplPath
    box = [MplPath(np.array([[3.0, -1.0], [6.0, -1.0], [6.0, 1.0],
                             [3.0, 1.0]]))]
    parts = _split_outside(line, box)
    assert len(parts) == 2, parts
    assert parts[0][-1][0] <= 3.0 and parts[1][0][0] >= 6.0, parts
    print("waterline: Otsu separates a bimodal sample and not a unimodal one; "
          "the exclusion mask cuts a line into the pieces outside it")


if __name__ == "__main__":
    _check()
