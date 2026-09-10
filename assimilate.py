"""Sequential per-cell assimilation of dated waterlines into an intertidal DTM.

    forecast   z <- diffuse(z, D dt),        P <- P + q dt
    update     K = P / (P + R w),   z <- z + K r,   P <- (1 - K) P
               r = y - z,   w = max(1, |r| / (c sqrt(P + R)))       (Huber)

Before an update, an observation is dropped if its cell was seen before and
|r| > delta_max (a percentile of the run's own innovations), or if its cell is
new and lies further than L_max (a percentile of new-cell distances) from every
observed cell. The diffusion is numerical regularisation, not sediment
transport; it acts on the mean only, so one scalar P per cell is exact only at
D = 0.
"""
from __future__ import annotations

from datetime import date

import numpy as np

CELL_M = 10.0

P0_M2 = 9.0                  # (3 m)^2: the state starts unknown
Q_RATE_M2_PER_YEAR = 0.09
R_OBS_M2 = 0.09
HUBER_C = 1.345              # 95 % efficiency under Gaussian errors
DIFFUSIVITY_M2_PER_YEAR = 50.0
MAX_DIFFUSION_STEP = 0.25    # sub-step ceiling on alpha, for stability

DILATE_CELLS = 5             # cells carried around the crossed ones
MAX_ANALYSIS_SD_M = 1.0      # a cell is reported where sqrt(P) is below it

REJECT_DIFF_PCT = 95.0       # percentile of innovations setting delta_max
REJECT_DIST_PCT = 99.0       # percentile of new-cell distances setting L_max


def build_grid(lines):
    """(x0, x1, y0, y1, nx, ny) covering every vertex, snapped to CELL_M."""
    v = np.concatenate([e["xy"] for e in lines])
    x0 = np.floor(v[:, 0].min() / CELL_M) * CELL_M
    y1 = np.ceil(v[:, 1].max() / CELL_M) * CELL_M
    nx = int((np.ceil(v[:, 0].max() / CELL_M) * CELL_M - x0) / CELL_M)
    ny = int((y1 - np.floor(v[:, 1].min() / CELL_M) * CELL_M) / CELL_M)
    return x0, x0 + nx * CELL_M, y1 - ny * CELL_M, y1, nx, ny


def build_domain(lines, grid):
    """(active mask, cell centres, per-epoch (cells, mean vertex level) observations)."""
    from scipy.ndimage import binary_dilation

    x0, _, _, y1, nx, ny = grid
    crossed = np.zeros((ny, nx), bool)
    ij = []
    for e in lines:
        j = ((e["xy"][:, 0] - x0) / CELL_M).astype(int)
        i = ((y1 - e["xy"][:, 1]) / CELL_M).astype(int)
        ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        crossed[i[ok], j[ok]] = True
        ij.append((i[ok], j[ok], e["y"][ok]))

    r = DILATE_CELLS
    active = binary_dilation(crossed, np.ones((2 * r + 1, 2 * r + 1), bool))
    idx = np.full((ny, nx), -1, np.int32)
    ai, aj = np.nonzero(active)
    idx[ai, aj] = np.arange(len(ai))
    xy = np.column_stack([x0 + (aj + 0.5) * CELL_M, y1 - (ai + 0.5) * CELL_M])

    obs = []
    for (i, j, y) in ij:
        if not len(i):
            obs.append((np.zeros(0, np.int64), np.zeros(0)))
            continue
        k = idx[i, j]
        order = np.argsort(k, kind="stable")
        k, y = k[order], y[order]
        cut = np.concatenate([[0], np.flatnonzero(np.diff(k)) + 1])
        cnt = np.diff(np.concatenate([cut, [len(k)]]))
        obs.append((k[cut], np.add.reduceat(y, cut) / cnt))

    print(f"domain: {int(crossed.sum()):,} cells crossed, {len(ai):,} carried "
          f"after a {r}-cell dilation ({100 * len(ai) / (nx * ny):.1f}% of the "
          f"{nx} x {ny} grid)")
    return active, xy, obs


def _dt_years(lines):
    d = [date.fromisoformat(e["date"]).toordinal() for e in lines]
    return np.diff(np.concatenate([[d[0]], d])) / 365.25


def _diffuse(g, m, a):
    """One conservative-flux diffusion sub-step, no flux across the mask edge."""
    out = g.copy()
    for sa, sb in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                   ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        f = np.where(m[sa] & m[sb], (a / 4.0) * (g[sb] - g[sa]), 0.0)
        out[sa] += f
        out[sb] -= f
    return np.where(m, out, g)


def _forecast(z, P, active, dt, diffusivity):
    """Diffuse the mean over dt years and inflate the variance."""
    alpha = 4.0 * diffusivity * dt / CELL_M ** 2
    if alpha > 0:
        can = P < P0_M2                    # only cells that carry information
        g = np.zeros(active.shape)
        g[active] = z
        m = np.zeros(active.shape, bool)
        m[active] = can
        nsub = max(1, int(np.ceil(alpha / MAX_DIFFUSION_STEP)))
        for _ in range(nsub):
            g = _diffuse(g, m, alpha / nsub)
        z = g[active]
    return z, P + Q_RATE_M2_PER_YEAR * dt


def _update(z, P, o, y, huber_c=HUBER_C):
    """Huber-robustified scalar Kalman update of the observed cells."""
    r = y - z[o]
    V = P[o] + R_OBS_M2
    if huber_c > 0.0:
        w = np.maximum(1.0, np.abs(r) / (huber_c * np.sqrt(V)))
        V = P[o] + R_OBS_M2 * w
    K = P[o] / V
    z[o] = z[o] + K * r
    P[o] = (1.0 - K) * P[o]
    return z, P


def screening_thresholds(lines, obs, active, prior,
                         diffusivity=DIFFUSIVITY_M2_PER_YEAR):
    """(delta_max, L_max) from a first pass over the record that rejects nothing.

    Innovations are taken against the previous analysis, not the forecast the
    gate in assimilate() uses; the two differ by one diffusion step.
    """
    from scipy.ndimage import distance_transform_edt

    ny, nx = active.shape
    ai, aj = np.nonzero(active)
    z = np.full(int(active.sum()), prior)
    P = np.full(int(active.sum()), P0_M2)
    seen = np.zeros(len(z), bool)
    diffs, dists = [], []

    for dt, (o, y) in zip(_dt_years(lines), obs):
        if len(o):
            old = seen[o]
            if old.any():
                diffs.append(np.abs(y[old] - z[o][old]))
            if (~old).any() and seen.any():
                m = np.zeros((ny, nx), bool)
                m[ai[seen], aj[seen]] = True
                d = distance_transform_edt(~m) * CELL_M
                dists.append(d[ai[o[~old]], aj[o[~old]]])
        z, P = _forecast(z, P, active, dt, diffusivity)
        if len(o):
            z, P = _update(z, P, o, y)
            seen[o] = True

    diffs = np.concatenate(diffs) if diffs else np.zeros(0)
    dists = np.concatenate(dists) if dists else np.zeros(0)
    dmax = float(np.percentile(diffs, REJECT_DIFF_PCT)) if len(diffs) else np.inf
    lmax = float(np.percentile(dists, REJECT_DIST_PCT)) if len(dists) else np.inf
    print(f"screening: delta_max = p{REJECT_DIFF_PCT:g} = {dmax:.2f} m of "
          f"{len(diffs):,} innovations, L_max = p{REJECT_DIST_PCT:g} = "
          f"{lmax:.0f} m of {len(dists):,} new-cell distances")
    return dmax, lmax


def assimilate(lines, obs, active, prior, delta_max=None, l_max=None,
               diffusivity=DIFFUSIVITY_M2_PER_YEAR, huber_c=HUBER_C):
    """(z, P) per epoch over the active cells, as (n_epochs, n_active) arrays."""
    from scipy.ndimage import distance_transform_edt

    ny, nx = active.shape
    ai, aj = np.nonzero(active)
    n = int(active.sum())
    z = np.full(n, prior)
    P = np.full(n, P0_M2)
    zs = np.empty((len(lines), n), np.float32)
    Ps = np.empty((len(lines), n), np.float32)

    seen = np.zeros(n, bool)
    n_drop = n_obs = 0

    for k, (dt, (o, y)) in enumerate(zip(_dt_years(lines), obs)):
        z, P = _forecast(z, P, active, dt, diffusivity)
        n_obs += len(o)
        if len(o):
            drop = np.zeros(len(o), bool)
            old = seen[o]
            if delta_max is not None and old.any():
                drop |= old & (np.abs(y - z[o]) > delta_max)
            if l_max is not None and (~old).any() and seen.any():
                m = np.zeros((ny, nx), bool)
                m[ai[seen], aj[seen]] = True
                d = distance_transform_edt(~m) * CELL_M
                far = np.zeros(len(o), bool)
                far[~old] = d[ai[o[~old]], aj[o[~old]]] > l_max
                drop |= far
            n_drop += int(drop.sum())
            o, y = o[~drop], y[~drop]
        if len(o):
            z, P = _update(z, P, o, y, huber_c)
            seen[o] = True
        zs[k], Ps[k] = z, P

    if n_obs:
        print(f"assimilation: {len(lines)} epochs, {n_obs:,} cell observations, "
              f"{n_drop:,} screened out ({100 * n_drop / n_obs:.1f}%)")
    return zs, Ps


def to_grid(values, active, mask=None):
    """Scatter an active-cell vector back onto the grid, NaN elsewhere."""
    g = np.full(active.shape, np.nan)
    g[active] = values if mask is None else np.where(mask, values, np.nan)
    return g


def reported_dem(z, P, active, max_sd=MAX_ANALYSIS_SD_M):
    """The DTM as reported: cells whose analysis standard deviation is below max_sd."""
    return to_grid(z, active, P < max_sd ** 2)


def _check():
    """A flat bed observed repeatedly must be recovered, and an outlier ignored."""
    active = np.ones((5, 5), bool)
    n = active.sum()
    lines = [{"date": f"2025-01-{d:02d}"} for d in range(1, 11)]
    cells = np.arange(n)

    # ten noiseless observations of a bed at -1 m, from a prior of 0 m
    obs = [(cells, np.full(n, -1.0)) for _ in lines]
    zs, Ps = assimilate(lines, obs, active, prior=0.0)
    assert abs(zs[-1].mean() + 1.0) < 0.02, zs[-1].mean()
    assert Ps[-1].max() < R_OBS_M2, Ps[-1].max()
    assert Ps[-1].max() < Ps[0].max(), "the variance must fall as cells are seen"

    # a 5 m outlier on the last epoch: Huber holds the state, plain Kalman does not
    obs[-1] = (cells, np.full(n, 4.0))
    zh, _ = assimilate(lines, obs, active, prior=0.0)
    zk, _ = assimilate(lines, obs, active, prior=0.0, huber_c=0.0)
    assert abs(zh[-1].mean() + 1.0) < 0.10, zh[-1].mean()
    assert abs(zk[-1].mean() + 1.0) > 5 * abs(zh[-1].mean() + 1.0), (zh[-1].mean(),
                                                                     zk[-1].mean())

    # a delta_max below the outlier's innovation keeps it out entirely
    dmax, lmax = screening_thresholds(lines, obs, active, prior=0.0)
    assert np.isfinite(dmax) and dmax > 0.0, dmax
    zg, _ = assimilate(lines, obs, active, prior=0.0, delta_max=1.0)
    assert abs(zg[-1].mean() + 1.0) < 5e-3, zg[-1].mean()

    g = reported_dem(zs[-1], Ps[-1], active)
    assert np.isfinite(g).all() and g.shape == (5, 5)
    print(f"assimilate: state converges to the observed bed "
          f"({zs[-1].mean():+.3f} m), Huber holds it at {zh[-1].mean():+.3f} m "
          f"against {zk[-1].mean():+.3f} m without it")


if __name__ == "__main__":
    _check()
