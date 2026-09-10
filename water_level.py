"""Water level at each waterline vertex, on local mean sea level.

    z = eta_tide + eta_res + eta_setup,    eta_setup = 0.35 beta_f sqrt(H0 L0)

eta_tide and eta_res are TCARI values (Hess 2003) supplied per scene; the setup
follows Stockdon et al. (2006), with beta_f read from a frozen DEM so that the
slope does not depend on the DEM being built.
"""
from __future__ import annotations

import csv

import numpy as np

STOCKDON_C = 0.35
GRAVITY = 9.81
BETA_RANGE = (0.001, 0.10)   # 1:1000 flat to 1:10 face; outside this is noise
SLOPE_SMOOTH_M = 50.0        # the slope is a beach property, not a cell property


def stockdon_setup(beta, wave_factor):
    """Setup at each vertex, zero where the slope is outside BETA_RANGE."""
    b = np.where((beta >= BETA_RANGE[0]) & (beta <= BETA_RANGE[1]), beta,
                 np.nan)
    e = STOCKDON_C * b * wave_factor
    return np.where(np.isfinite(e), e, 0.0)


def wave_factor(hs, period):
    """sqrt(H0 L0) in metres, from deep-water height and period."""
    return np.sqrt(hs * GRAVITY * period ** 2 / (2.0 * np.pi))


def read_levels(csv_path):
    """{date: (tide_m, residual_m, wave_factor_m)} from the per-scene table."""
    out = {}
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = r["wave_factor_m"]
            out[r["date"]] = (float(r["tide_m"]), float(r["residual_m"]),
                              float(f) if f else 0.0)
    return out


def slope_sampler(dem_path):
    """beta_f at any projected coordinate, from the DEM smoothed over SLOPE_SMOOTH_M."""
    import rasterio
    from scipy.ndimage import uniform_filter

    with rasterio.open(dem_path) as d:
        z = d.read(1).astype("f8")
        tr, shape = d.transform, (d.height, d.width)

    ok = np.isfinite(z)
    k = max(int(round(SLOPE_SMOOTH_M / abs(tr.a))), 1)
    cnt = uniform_filter(ok.astype(float), k, mode="nearest")
    fil = uniform_filter(np.where(ok, z, 0.0), k, mode="nearest")
    zs = np.where(cnt > 0, fil / np.maximum(cnt, 1e-9), np.nan)
    gy, gx = np.gradient(np.where(np.isfinite(zs), zs, 0.0), abs(tr.a))
    beta = np.clip(np.where(ok, np.hypot(gx, gy), np.nan), *BETA_RANGE)

    def at(x, y):
        j = np.clip(((np.asarray(x) - tr.c) / tr.a).astype(int), 0, shape[1] - 1)
        i = np.clip(((np.asarray(y) - tr.f) / tr.e).astype(int), 0, shape[0] - 1)
        return beta[i, j]

    return at


def attach_levels(lines, levels, slope_at=None):
    """Each vertex's water level; waterlines with no level for their date are dropped."""
    out = []
    for e in lines:
        got = levels.get(e["date"])
        if got is None:
            continue
        tide, res, fac = got
        xy = e["xy"]
        s = (stockdon_setup(slope_at(xy[:, 0], xy[:, 1]), fac)
             if slope_at is not None else np.zeros(len(xy)))
        out.append({"date": e["date"], "xy": xy, "y": tide + res + s})
    return out


def _check():
    assert abs(stockdon_setup(np.array([0.02]), 10.0)[0] - 0.07) < 1e-12
    for bad in (0.0005, 0.5, np.nan):
        assert stockdon_setup(np.array([bad]), 10.0)[0] == 0.0, bad

    # Hs = 1 m, T = 8 s: L0 = 9.81 * 64 / (2 pi) = 99.9 m, sqrt(H0 L0) = 10.0 m
    assert abs(wave_factor(1.0, 8.0) - 9.997) < 1e-3, wave_factor(1.0, 8.0)

    lines = [{"date": "2025-01-01", "xy": np.zeros((3, 2))},
             {"date": "1999-01-01", "xy": np.zeros((3, 2))}]
    got = attach_levels(lines, {"2025-01-01": (-2.0, 0.1, 10.0)})
    assert len(got) == 1 and np.allclose(got[0]["y"], -1.9), got
    print("water_level: setup gated on slope, level = tide + residual + setup, "
          "dates without a level dropped")


if __name__ == "__main__":
    _check()
