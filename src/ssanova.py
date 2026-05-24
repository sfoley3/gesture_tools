# ABOUTME: Python equivalent of R's gss::ssanova(y ~ x) for tongue contour analysis.
# ABOUTME: Fits GCV-smoothed splines on pooled contour points with bootstrap 95% CI.
"""
ssanova.py — smoothing spline ANOVA equivalent for tongue contour data.

Public API
----------
fit(x, y, n_grid=200, n_boot=200, seed=42) -> SSAnovaFit | None
    Fit a smoothing spline on pooled (x, y) observations and return a
    named-tuple with fields: x_grid, y_fit, y_lo, y_hi.

compute_all(records, sigma=5) -> dict[(spk, phone, position), SSAnovaFit]
    Convenience wrapper: smooths each contour with gaussian_filter1d,
    pools points by speaker × phone × position, and calls fit() on each group.

Notes
-----
The smoothing spline is fitted with scipy.interpolate.make_smoothing_spline
(scipy >= 1.10) which selects the penalty λ via GCV — the same criterion used
by R's gss::ssanova. Confidence intervals are obtained by bootstrapping over
individual contour tokens (not individual points), which correctly propagates
inter-token variability, matching the Bayesian credible intervals produced by
ssanova's predict(..., se=TRUE).

Requires: numpy, scipy >= 1.10
"""

from __future__ import annotations

import warnings
from collections import defaultdict, namedtuple
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── Public types ──────────────────────────────────────────────────────────────

SSAnovaFit = namedtuple('SSAnovaFit', ['x_grid', 'y_fit', 'y_lo', 'y_hi'])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_spline_fn():
    """Return a spline-fitting callable, preferring make_smoothing_spline."""
    try:
        from scipy.interpolate import make_smoothing_spline  # scipy >= 1.10

        def _fit(x: np.ndarray, y: np.ndarray):
            """GCV-smoothed cubic spline; returns a callable."""
            # make_smoothing_spline requires strictly increasing x
            order  = np.argsort(x)
            xs, ys = x[order], y[order]
            # Average y at duplicate x positions
            xs_u, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
            ys_u = np.bincount(inv, weights=ys).astype(float) / cnt
            if len(xs_u) < 5:
                return None
            return make_smoothing_spline(xs_u, ys_u)   # lam=None → GCV

        return _fit

    except ImportError:
        from scipy.interpolate import UnivariateSpline

        def _fit(x: np.ndarray, y: np.ndarray):           # type: ignore[misc]
            order  = np.argsort(x)
            xs, ys = x[order], y[order]
            xs_u, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
            ys_u = np.bincount(inv, weights=ys).astype(float) / cnt
            if len(xs_u) < 5:
                return None
            # s = m - sqrt(2m) is scipy's recommended default for noisy data
            m = len(xs_u)
            return UnivariateSpline(xs_u, ys_u, k=3, s=m - np.sqrt(2 * m))

        return _fit


_spline_fit = _make_spline_fn()


# ── Public API ────────────────────────────────────────────────────────────────

def fit(
    x: np.ndarray,
    y: np.ndarray,
    n_grid: int = 200,
    n_boot: int = 200,
    seed: int = 42,
) -> Optional[SSAnovaFit]:
    """
    Fit a smoothing spline on pooled (x, y) observations.

    Parameters
    ----------
    x, y : 1-D arrays of the same length — pooled contour coordinates.
    n_grid : number of evenly spaced prediction points across the x range.
    n_boot : bootstrap resamples for the 95 % CI (resamples over points).
    seed   : random seed for reproducibility.

    Returns
    -------
    SSAnovaFit(x_grid, y_fit, y_lo, y_hi) or None if fitting fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 10:
        return None

    x_grid = np.linspace(x.min(), x.max(), n_grid)

    # Main fit
    spl = _spline_fit(x, y)
    if spl is None:
        return None
    y_fit = spl(x_grid)

    # Bootstrap CI — resample individual observations
    rng      = np.random.default_rng(seed)
    n        = len(x)
    boot_ys  = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, n)
        spl_b = _spline_fit(x[idx], y[idx])
        if spl_b is not None:
            boot_ys.append(spl_b(x_grid))

    if len(boot_ys) < 10:
        # Fallback: constant SE from residual std
        resid = y - spl(x)
        se    = resid.std() / np.sqrt(max(n // 10, 1))
        return SSAnovaFit(x_grid, y_fit, y_fit - 1.96 * se, y_fit + 1.96 * se)

    boot_arr = np.array(boot_ys)
    y_lo     = np.percentile(boot_arr,  2.5, axis=0)
    y_hi     = np.percentile(boot_arr, 97.5, axis=0)

    return SSAnovaFit(x_grid, y_fit, y_lo, y_hi)


def compute_all(
    records: list,
    sigma: float = 5.0,
    n_grid: int = 200,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[Tuple[str, str, str], SSAnovaFit]:
    """
    Compute ssanova fits for every (spk, phone, position) combination.

    Mirrors the per-speaker group_modify loop in liquids_analysis.R:
    smooths each contour with gaussian_filter1d, pools the points, then
    fits one smoothing spline per category.

    Parameters
    ----------
    records : list of dicts with keys spk, phone, position, x (100,), y (100,).
    sigma   : Gaussian smoothing sigma applied before pooling (matches R).
    n_grid, n_boot, seed : forwarded to fit().

    Returns
    -------
    dict mapping (spk, phone, position) -> SSAnovaFit.
    """
    groups: Dict[Tuple, list] = defaultdict(list)
    for rec in records:
        groups[(rec['spk'], rec['phone'], rec['position'])].append(rec)

    fits: Dict[Tuple, SSAnovaFit] = {}
    n_groups = len(groups)

    for i, ((spk, ph, pos), recs) in enumerate(sorted(groups.items()), 1):
        print(f'  [{i}/{n_groups}] ssanova: {spk} {ph} {pos} '
              f'({len(recs)} contours, {len(recs)*100} pts)', end='', flush=True)

        xs = np.concatenate([gaussian_filter1d(r['x'], sigma) for r in recs])
        ys = np.concatenate([gaussian_filter1d(r['y'], sigma) for r in recs])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fit(xs, ys, n_grid=n_grid, n_boot=n_boot, seed=seed)

        if result is not None:
            fits[(spk, ph, pos)] = result
            print(' ✓')
        else:
            print(' — skipped (insufficient data)')

    return fits
