#!/usr/bin/env python3
# ABOUTME: Computes vocal-tract distance (VTD) across a fixed Proctor/Kim semipolar grid from SAM2 tissue masks.
# ABOUTME: Outputs per-video (T, L) VTD timeseries, per-speaker min-max normalized VTD + histograms, and diagnostic frame/video overlays.
"""
extract_vtd.py — Vocal Tract Distance from SAM2 masks (Shi et al. 2024 style).

For each speaker we build ONE fixed, speaker-specific *semipolar* analysis grid
(Proctor et al. 2010; Kim et al. 2014) from the time-aggregated masks, then apply
that same grid to every frame of every video.  Because the grid is fixed, grid
line ``l`` denotes the same anatomical location across all frames/tokens, which
is exactly what the per-gridline VTD histograms in Shi (2024) require.

Unlike Proctor (who thresholds raw MRI intensity and runs Dijkstra to find the
centerline), *every landmark is already segmented here*, so grid construction is
fully deterministic from mask geometry:

  * palate apex       -> highest (min-y) point of the ``upper lip - palate`` mask
  * alveolar ridge    -> leftmost stable column of the time-averaged palate
  * lips              -> lip-aperture midpoint (upper-lip / lower-lip closest pair)
  * lingual origin    -> point equidistant from palate roof and rear pharyngeal
                         wall, near the time-averaged tongue centroid
  * posterior end     -> bottom of the tongue mask and its closest point on the
                         pharyngeal wall (there is NO larynx/glottis mask)
  * upper VT boundary -> palate + velum + pharyngeal-wall airway-facing edges
  * lower VT boundary -> lower-lip inner edge + tongue airway-facing contour
                         (tip over the top and down to the tongue bottom)

VTD on grid line ``l`` = Euclidean pixel distance between that line's
intersection with the upper (roof) boundary and the lower (floor) boundary. The
posterior-most line is pinned to tongue-bottom -> closest pharyngeal-wall point.

Grid layout (front -> back), following the natural tract bend:
  1. Labial     : vertical lines anterior to the alveolar ridge, through the lips
  2. Palatal fan: radial lines from the lingual origin across the oral bend
  3. Pharyngeal : horizontal lines from the lingual-origin level to the posterior
                  end (tongue bottom / pharyngeal wall)

``--n-gridlines`` is the TOTAL number of grid lines, distributed evenly by arc
length along the roof from the lip anchor to the pharyngeal (posterior) anchor.

Usage:
    conda run -n myenv python extract_vtd.py [--spk 2 3 ...] \
        [--n-gridlines 40] [--n-videos 5] [--bins 20]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

# ── Config (same pattern as the other gesture_tools scripts) ─────────────────
_DEFAULT_CFG = {
    "data_dir": ".",
    "n_diagnostic": 5,
    "spk_base": "",
    "video_dir": "video",
    "dataset": "lss",
    "n_gridlines": 40,
    "n_bins": 20,
}


def _load_config() -> dict:
    candidates = []
    env_path = os.environ.get("GESTURE_TOOLS_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parent.parent / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate) as _f:
                return {**_DEFAULT_CFG, **json.load(_f)}
    return dict(_DEFAULT_CFG)


_cfg = _load_config()
DATA_DIR = Path(_cfg["data_dir"])
N_DIAGNOSTIC = int(_cfg.get("n_diagnostic", 5))
SPK_BASE = _cfg.get("spk_base", "")
VIDEO_DIR = _cfg.get("video_dir", "video")
N_GRIDLINES = int(_cfg.get("n_gridlines", 40))
N_BINS = int(_cfg.get("n_bins", 20))
MAX_XY = 104

# Region key substrings (case-insensitive). Pharyngeal wall auto-detects "pharyn".
# There is no larynx/glottis mask; the posterior end of the tract is defined by
# the bottom of the tongue and its closest point on the pharyngeal wall.
ROOF_FRONT_SUB = "upper lip"   # "upper lip - palate"
VELUM_SUB = "velum"
PHARYNX_SUB = "pharyn"         # "pharyngeal wall"
TONGUE_SUB = "tongue"
LOWER_LIP_SUB = "lower lip"    # "lower lip - jaw"

# The five segmented regions, in canonical order.
REGION_SUBS = [ROOF_FRONT_SUB, VELUM_SUB, PHARYNX_SUB, TONGUE_SUB, LOWER_LIP_SUB]


# ── Mask-key helpers ─────────────────────────────────────────────────────────

def _find_mask_key(keys, substring: str):
    sub = substring.lower()
    for k in keys:
        if sub in k.lower():
            return k
    return None


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


# ── Contour / arc tracing ────────────────────────────────────────────────────
# (self-contained ports of the helpers in get_tongue_contours.py and
#  plot_pts_contour.py, so this tool runs standalone.)

def _ordered_loops(mask):
    if mask is None or not mask.any():
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return [c.squeeze(1).astype(np.float32) for c in contours if len(c) >= 2]


def _arc_between(loop, i, j):
    n = len(loop)
    if i == j:
        return loop[i:i + 1], loop[i:i + 1]
    fwd_idx = [(i + k) % n for k in range(((j - i) % n) + 1)]
    bwd_idx = [(i - k) % n for k in range(((i - j) % n) + 1)]
    return loop[fwd_idx], loop[bwd_idx]


def _bottom_arc_ordered(mask):
    """Bottom (high-y / airway-facing) arc of a mask, oriented front (low x) ->
    back (high x). Used for the palate and velum roof."""
    loops = _ordered_loops(mask)
    if not loops:
        return None
    loop = max(loops, key=len)
    L = int(np.argmin(loop[:, 0]))
    R = int(np.argmax(loop[:, 0]))
    fwd, bwd = _arc_between(loop, L, R)
    arc = fwd if fwd[:, 1].mean() > bwd[:, 1].mean() else bwd
    if len(arc) < 2:
        return None
    if arc[0, 0] > arc[-1, 0]:
        arc = arc[::-1]
    return arc


def _anterior_arc_ordered(mask):
    """Anterior (low-x / airway-facing) arc of the pharyngeal-wall mask, oriented
    top (low y) -> bottom (high y). The rear pharyngeal wall faces the airway on
    its front (minimum-x) side."""
    loops = _ordered_loops(mask)
    if not loops:
        return None
    loop = max(loops, key=len)
    T = int(np.argmin(loop[:, 1]))   # top
    B = int(np.argmax(loop[:, 1]))   # bottom
    fwd, bwd = _arc_between(loop, T, B)
    # airway-facing arc has the smaller mean x (anterior side)
    arc = fwd if fwd[:, 0].mean() < bwd[:, 0].mean() else bwd
    if len(arc) < 2:
        return None
    if arc[0, 1] > arc[-1, 1]:
        arc = arc[::-1]
    return arc


def _top_arc_ordered(mask):
    """Top (low-y / airway-facing) arc of a mask, oriented front (low x) -> back
    (high x). Used for the lower-lip inner edge."""
    loops = _ordered_loops(mask)
    if not loops:
        return None
    loop = max(loops, key=len)
    L = int(np.argmin(loop[:, 0]))
    R = int(np.argmax(loop[:, 0]))
    fwd, bwd = _arc_between(loop, L, R)
    arc = fwd if fwd[:, 1].mean() < bwd[:, 1].mean() else bwd
    if len(arc) < 2:
        return None
    if arc[0, 0] > arc[-1, 0]:
        arc = arc[::-1]
    return arc


def _arc_containing(loop, i, j, t):
    """Cyclic arc i->j that passes through index t (from plot_pts_contour.py)."""
    n = len(loop)
    fwd_len = (j - i) % n
    on_fwd = (t - i) % n <= fwd_len
    fwd_idx = [(i + k) % n for k in range(fwd_len + 1)]
    bwd_idx = [(i - k) % n for k in range(((i - j) % n) + 1)]
    return loop[fwd_idx] if on_fwd else loop[bwd_idx]


def _tongue_airway_arc(mask):
    """Airway-facing tongue contour: from the tip (front / min-x) over the top
    (min-y) and down the posterior edge to the tongue bottom (max-y). This is the
    lower VT boundary through both the oral cavity and the pharynx. Oriented
    front -> back/down."""
    loops = _ordered_loops(mask)
    if not loops:
        return None
    loop = max(loops, key=len)
    L = int(np.argmin(loop[:, 0]))    # tip (front)
    B = int(np.argmax(loop[:, 1]))    # bottom (posterior-inferior)
    Tp = int(np.argmin(loop[:, 1]))   # airway-facing apex (top)
    if Tp in (L, B):
        fwd, bwd = _arc_between(loop, L, B)
        arc = fwd if fwd[:, 1].mean() < bwd[:, 1].mean() else bwd
    else:
        arc = _arc_containing(loop, L, B, Tp)
    if len(arc) < 2:
        return None
    if arc[0, 0] > arc[-1, 0]:         # front (low x) first
        arc = arc[::-1]
    return arc


def _resample_polyline(poly, step=0.5):
    """Densify a polyline to ~step-pixel spacing (arc-length) for robust
    intersection tests. Returns (M, 2)."""
    if poly is None or len(poly) < 2:
        return poly
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return poly
    m = max(2, int(total / step) + 1)
    s = np.linspace(0, total, m)
    x = np.interp(s, cum, poly[:, 0])
    y = np.interp(s, cum, poly[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _join_front_back(front, back):
    """Concatenate two front->back arcs, trimming x-overlap (from plot_pts_contour)."""
    if front is None:
        return back
    if back is None:
        return front
    f_max = float(front[-1, 0])
    b_min = float(back[0, 0])
    if f_max >= b_min:
        cut = 0.5 * (f_max + b_min)
        front = front[front[:, 0] <= cut]
        back = back[back[:, 0] >= cut]
        if len(front) < 1 or len(back) < 1:
            return front if len(front) >= len(back) else back
    return np.vstack([front, back])


def trace_roof(region_by_sub: dict):
    """Upper VT boundary, front (lips) -> back/down (pharyngeal wall):
    palate bottom edge  ->  velum bottom edge  ->  pharyngeal-wall anterior edge."""
    palate = _bottom_arc_ordered(region_by_sub.get(ROOF_FRONT_SUB))
    velum = _bottom_arc_ordered(region_by_sub.get(VELUM_SUB))
    oral = _join_front_back(palate, velum)
    phar = _anterior_arc_ordered(region_by_sub.get(PHARYNX_SUB))
    if oral is None:
        return _resample_polyline(phar)
    if phar is None:
        return _resample_polyline(oral)
    # velum end (back of oral) joins the top of the pharyngeal arc
    roof = np.vstack([oral, phar])
    return _resample_polyline(roof)


def trace_floor(region_by_sub: dict):
    """Lower VT boundary, front (lips) -> back/down (tongue bottom):
    lower-lip inner (top) edge  ->  tongue airway-facing contour (tip over the
    top and down the posterior edge to the tongue bottom)."""
    lip = _top_arc_ordered(region_by_sub.get(LOWER_LIP_SUB))
    tongue = _tongue_airway_arc(region_by_sub.get(TONGUE_SUB))
    floor = _join_front_back(lip, tongue)
    return _resample_polyline(floor)


# ── Landmarks (all mask-derived) ─────────────────────────────────────────────

def _aggregate_occupancy(mask_stack: np.ndarray, thresh: float = 0.15) -> np.ndarray:
    """Time-average a (T,H,W) mask into a representative binary occupancy map."""
    if mask_stack is None:
        return None
    occ = mask_stack.mean(axis=0)
    m = occ > thresh
    return m if m.any() else mask_stack.any(axis=0)


def _alveolar_ridge(palate_agg):
    ys, xs = np.where(palate_agg)
    x_ar = int(xs.min())
    col = palate_agg[:, x_ar]
    y_ar = float(np.where(col)[0].mean())
    return np.array([float(x_ar), y_ar], np.float32)


def _palate_apex(palate_agg):
    ys, xs = np.where(palate_agg)
    y_top = int(ys.min())
    row = palate_agg[y_top, :]
    x_apex = float(np.where(row)[0].mean())
    return np.array([x_apex, float(y_top)], np.float32)


def _posterior_anchor(tongue_agg, pharynx_agg):
    """Posterior end of the tract: the bottom-most point of the tongue mask and
    its closest point on the pharyngeal wall. The final grid line spans this
    pair (this replaces any glottis/larynx reference — there is no larynx mask).
    Returns (tongue_bottom (2,), pharyngeal_point (2,))."""
    ty, tx = np.where(tongue_agg)
    y_bot = int(ty.max())
    x_bot = float(tx[ty == y_bot].mean())
    tongue_bottom = np.array([x_bot, float(y_bot)], np.float32)
    if pharynx_agg is not None and pharynx_agg.any():
        qy, qx = np.where(pharynx_agg)
        phar_pts = np.stack([qx, qy], 1).astype(np.float32)
        d = ((phar_pts - tongue_bottom[None, :]) ** 2).sum(1)
        phar_pt = phar_pts[d.argmin()].astype(np.float32)
    else:
        phar_pt = tongue_bottom.copy()
    return tongue_bottom, phar_pt


def _lips_point(palate_agg, lower_lip_agg):
    """Lip-aperture midpoint = midpoint of closest upper/lower lip pixels."""
    uy, ux = np.where(palate_agg)
    ly, lx = np.where(lower_lip_agg)
    # restrict to anterior region (front third) for speed & correctness
    x_lo = min(ux.min(), lx.min())
    span = max(ux.max(), lx.max()) - x_lo
    ux_m = ux < x_lo + 0.35 * span
    lx_m = lx < x_lo + 0.35 * span
    if ux_m.any():
        ux, uy = ux[ux_m], uy[ux_m]
    if lx_m.any():
        lx, ly = lx[lx_m], ly[lx_m]
    dx = ux[:, None] - lx[None, :]
    dy = uy[:, None] - ly[None, :]
    i_u, i_l = np.unravel_index((dx ** 2 + dy ** 2).argmin(), (len(ux), len(lx)))
    return np.array([(ux[i_u] + lx[i_l]) / 2.0, (uy[i_u] + ly[i_l]) / 2.0], np.float32)


def _lingual_origin(palate_agg, pharynx_agg, tongue_agg):
    """Proctor lingual origin: equidistant from palate roof and rear pharyngeal
    wall, near the resting tongue centroid. We take the tongue centroid and nudge
    it to balance distance-to-palate vs distance-to-pharynx."""
    ty, tx = np.where(tongue_agg)
    c = np.array([tx.mean(), ty.mean()], np.float32)
    py, px = np.where(palate_agg)
    palate_pts = np.stack([px, py], 1).astype(np.float32)
    if pharynx_agg is not None and pharynx_agg.any():
        qy, qx = np.where(pharynx_agg)
        phar_pts = np.stack([qx, qy], 1).astype(np.float32)
    else:
        phar_pts = None

    def _nearest(pts, p):
        d = ((pts - p) ** 2).sum(1)
        return pts[d.argmin()]

    # one balancing step toward the midpoint of the two nearest wall points
    p_pal = _nearest(palate_pts, c)
    if phar_pts is not None:
        p_ph = _nearest(phar_pts, c)
        mid = 0.5 * (p_pal + p_ph)
        origin = 0.5 * (c + mid)
    else:
        origin = c
    return origin.astype(np.float32)


def build_landmarks(region_agg: dict) -> dict:
    palate = region_agg.get(ROOF_FRONT_SUB)
    lower_lip = region_agg.get(LOWER_LIP_SUB)
    tongue = region_agg.get(TONGUE_SUB)
    pharynx = region_agg.get(PHARYNX_SUB)
    tongue_bottom, pharyngeal_end = _posterior_anchor(tongue, pharynx)
    lm = {
        "alveolar_ridge": _alveolar_ridge(palate),
        "palate_apex": _palate_apex(palate),
        "lips": _lips_point(palate, lower_lip),
        "lingual_origin": _lingual_origin(palate, pharynx, tongue),
        # posterior end: tongue bottom -> closest pharyngeal-wall point
        "tongue_bottom": tongue_bottom,
        "pharyngeal_end": pharyngeal_end,
    }
    return lm


# ── Fixed semipolar grid ─────────────────────────────────────────────────────

def _project_arclen(poly, pt):
    """Arc-length position (0..S) of the roof-arc point closest to *pt*."""
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = ((poly - pt[None, :]) ** 2).sum(1)
    return float(cum[d.argmin()]), cum


def build_semipolar_grid(landmarks: dict, roof_arc: np.ndarray, n_gridlines: int):
    """Return a list of grid lines, each a dict with the roof origin point, a
    unit direction pointing toward the floor, and a section label.

    n stations are placed evenly by arc length along the roof between the lip
    anchor (anterior) and the pharyngeal end (posterior = the pharyngeal-wall
    point closest to the tongue bottom). Direction per section:
      labial     -> straight down            (0, +1)
      palatal fan-> radial from lingual origin (origin - roof_pt)
      pharyngeal -> horizontal toward airway  (-1, 0)
    Section boundaries are the alveolar ridge (labial|fan) and the lingual-origin
    height projected onto the roof (fan|pharyngeal).

    The final grid line is pinned to the user-defined posterior measurement:
    tongue_bottom -> pharyngeal_end (via a floor_override)."""
    O = landmarks["lingual_origin"]
    s_lips, cum = _project_arclen(roof_arc, landmarks["lips"])
    s_ar, _ = _project_arclen(roof_arc, landmarks["alveolar_ridge"])
    s_post, _ = _project_arclen(roof_arc, landmarks["pharyngeal_end"])
    # fan|pharyngeal boundary ~ arc position nearest the lingual-origin height
    phar_ref = np.array([O[0], O[1]], np.float32)
    s_ph, _ = _project_arclen(roof_arc, phar_ref)

    s0, s1 = min(s_lips, s_post), max(s_lips, s_post)
    s_ar = float(np.clip(s_ar, s0, s1))
    s_ph = float(np.clip(s_ph, s0, s1))
    lo, hi = sorted([s_ar, s_ph])

    stations = np.linspace(s0, s1, n_gridlines)
    grid = []
    for s in stations:
        # roof point at arc length s
        rx = np.interp(s, cum, roof_arc[:, 0])
        ry = np.interp(s, cum, roof_arc[:, 1])
        p = np.array([rx, ry], np.float32)
        if s < lo:
            section = "labial"
            d = np.array([0.0, 1.0], np.float32)
        elif s > hi:
            section = "pharyngeal"
            d = np.array([-1.0, 0.0], np.float32)
        else:
            section = "fan"
            v = O - p
            nv = np.linalg.norm(v)
            d = (v / nv) if nv > 1e-6 else np.array([0.0, 1.0], np.float32)
            if d[1] < 0:      # ensure it points toward the floor (downward)
                d = -d
        grid.append({"origin": p, "dir": d, "section": section})

    # Pin the posterior-most line to tongue_bottom -> pharyngeal_end exactly.
    tb = landmarks["tongue_bottom"]
    pe = landmarks["pharyngeal_end"]
    v = tb - pe
    nv = np.linalg.norm(v)
    grid[-1] = {
        "origin": pe.astype(np.float32),
        "dir": (v / nv if nv > 1e-6 else np.array([-1.0, 0.0], np.float32)).astype(np.float32),
        "section": "pharyngeal",
        "floor_override": tb.astype(np.float32),
    }
    return grid


# ── Ray / polyline intersection & VTD ────────────────────────────────────────

def _ray_polyline_hit(origin, direction, poly, max_len=MAX_XY * 1.6):
    """Nearest intersection of the *line through origin along ±direction* with a
    polyline. Returns (point, dist_along) or (None, inf)."""
    if poly is None or len(poly) < 2:
        return None, np.inf
    o = origin.astype(np.float64)
    dirn = direction.astype(np.float64)
    best_pt, best_t = None, np.inf
    a = poly[:-1].astype(np.float64)
    b = poly[1:].astype(np.float64)
    seg = b - a
    # Solve o + t*dirn = a + u*seg   for each segment
    denom = dirn[0] * (-seg[:, 1]) - dirn[1] * (-seg[:, 0])
    rhs = a - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rhs[:, 0] * (-seg[:, 1]) - rhs[:, 1] * (-seg[:, 0])) / denom
        u = (dirn[0] * rhs[:, 1] - dirn[1] * rhs[:, 0]) / denom
    ok = np.isfinite(t) & np.isfinite(u) & (u >= 0) & (u <= 1) & (np.abs(t) <= max_len)
    if not ok.any():
        return None, np.inf
    ti = t[ok]
    k = np.argmin(np.abs(ti))
    tt = ti[k]
    pt = (o + tt * dirn).astype(np.float32)
    return pt, float(abs(tt))


def compute_vtd(grid, roof_poly, floor_poly):
    """Per grid line: roof intersection U, floor intersection L, VTD=||U-L||.
    Returns (vtd (L,), U (L,2), L_pts (L,2))."""
    n = len(grid)
    vtd = np.full(n, np.nan, np.float32)
    U = np.full((n, 2), np.nan, np.float32)
    Lp = np.full((n, 2), np.nan, np.float32)
    for i, gl in enumerate(grid):
        o, d = gl["origin"], gl["dir"]
        # roof point: prefer true intersection; fall back to the origin station
        u_pt, _ = _ray_polyline_hit(o, d, roof_poly)
        if u_pt is None:
            u_pt = o
        # posterior-most line is pinned to tongue_bottom via floor_override
        if "floor_override" in gl:
            l_pt = gl["floor_override"]
        else:
            l_pt, _ = _ray_polyline_hit(u_pt, d, floor_poly)
        if l_pt is None:
            continue
        U[i] = u_pt
        Lp[i] = l_pt
        vtd[i] = float(np.linalg.norm(u_pt - l_pt))
    return vtd, U, Lp


# ── NPZ loading ──────────────────────────────────────────────────────────────

def _load_regions(mask_path: Path):
    data = np.load(mask_path)
    keys = list(data.keys())
    subs = REGION_SUBS
    regions = {}
    for sub in subs:
        k = _find_mask_key(keys, sub)
        regions[sub] = data[k].astype(bool) if k is not None else None
    T = next((m.shape[0] for m in regions.values() if m is not None), 0)
    return regions, T


def _frame_regions(regions: dict, t: int):
    out = {}
    for sub, m in regions.items():
        out[sub] = m[t] if (m is not None and t < m.shape[0]) else None
    return out


# ── Diagnostics ──────────────────────────────────────────────────────────────

# matplotlib tab10-ish colors per region for the static figure
_DIAG_COLORS = {
    ROOF_FRONT_SUB: "#1f77b4",   # palate - blue
    LOWER_LIP_SUB: "#ff7f0e",    # lower lip - orange
    TONGUE_SUB: "#2ca02c",       # tongue - green
    VELUM_SUB: "#d62728",        # velum - red
    PHARYNX_SUB: "#9467bd",      # pharyngeal wall - purple
}
# BGR for the cv2 video
_ROOF_BGR = (0, 0, 255)      # red
_FLOOR_BGR = (255, 128, 0)   # blue
_GRID_BGR = (255, 255, 0)    # cyan (Shi Fig 1b grid)
_VTD_BGR = (0, 255, 255)     # yellow (Shi Fig 1b VTD segments)


def save_static_diagnostic(out_path, frame_regions, roof, floor, grid, U, Lp,
                           landmarks):
    """One matplotlib figure: masks (translucent) + roof/floor edges + grid +
    VTD points, for a single representative frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, MAX_XY)
    ax.set_ylim(MAX_XY, 0)     # image coords (y down)
    ax.set_aspect("equal")

    for sub, m in frame_regions.items():
        if m is None or not m.any():
            continue
        rgb = mcolors.to_rgb(_DIAG_COLORS.get(sub, "#888888"))
        rgba = np.zeros((*m.shape, 4), np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = m.astype(np.float32) * 0.35
        ax.imshow(rgba, interpolation="nearest")

    # grid lines (cyan) + VTD segments (yellow)
    for i, gl in enumerate(grid):
        u, l = U[i], Lp[i]
        if np.isnan(u[0]) or np.isnan(l[0]):
            continue
        ax.plot([u[0], l[0]], [u[1], l[1]], color="cyan", lw=0.8, alpha=0.9, zorder=3)
        ax.scatter([u[0], l[0]], [u[1], l[1]], s=10, color="yellow", zorder=5,
                   linewidths=0)

    if roof is not None:
        ax.plot(roof[:, 0], roof[:, 1], color="red", lw=2.0, zorder=4, label="roof")
    if floor is not None:
        ax.plot(floor[:, 0], floor[:, 1], color="deepskyblue", lw=2.0, zorder=4,
                label="floor")
    for name, p in landmarks.items():
        ax.scatter([p[0]], [p[1]], s=45, marker="+", color="white", zorder=6)
        ax.annotate(name, (p[0], p[1]), fontsize=6, color="white",
                    xytext=(2, -2), textcoords="offset points")

    ax.set_title(f"VTD grid ({len(grid)} lines)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scale(x, y, mask_h, mask_w, fh, fw):
    return int(round(x * fw / mask_w)), int(round(y * fh / mask_h))


def write_diagnostic_video(out_path, regions, T, video_path, grid,
                           roof_frames, floor_frames, U_frames, L_frames):
    """Per-frame overlay: masks (translucent), roof/floor edges, grid lines,
    VTD points."""
    mask_h = mask_w = MAX_XY
    for m in regions.values():
        if m is not None:
            mask_h, mask_w = m.shape[1], m.shape[2]
            break

    if video_path is not None and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or mask_w
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or mask_h
    else:
        cap = None
        fps = 50.0
        fh, fw = mask_h * 5, mask_w * 5   # upscale blank canvas for legibility

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (fw, fh))
    region_bgr = {
        ROOF_FRONT_SUB: (180, 120, 30), LOWER_LIP_SUB: (30, 140, 240),
        TONGUE_SUB: (40, 180, 40), VELUM_SUB: (40, 40, 200),
        PHARYNX_SUB: (180, 100, 150),
    }

    for t in range(T):
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.zeros((fh, fw, 3), np.uint8)

        overlay = frame.copy()
        for sub, m in regions.items():
            if m is None or t >= m.shape[0] or not m[t].any():
                continue
            mr = cv2.resize(m[t].astype(np.uint8) * 255, (fw, fh),
                            interpolation=cv2.INTER_NEAREST)
            colored = np.zeros_like(frame)
            colored[mr > 0] = region_bgr.get(sub, (128, 128, 128))
            cv2.addWeighted(colored, 0.4, overlay, 1.0, 0, overlay)
        frame = overlay

        roof, floor = roof_frames[t], floor_frames[t]
        U, Lp = U_frames[t], L_frames[t]
        # grid + VTD segments
        for i in range(len(grid)):
            u, l = U[i], Lp[i]
            if np.isnan(u[0]) or np.isnan(l[0]):
                continue
            p1 = _scale(u[0], u[1], mask_h, mask_w, fh, fw)
            p2 = _scale(l[0], l[1], mask_h, mask_w, fh, fw)
            cv2.line(frame, p1, p2, _GRID_BGR, 1)
            cv2.circle(frame, p1, 2, _VTD_BGR, -1)
            cv2.circle(frame, p2, 2, _VTD_BGR, -1)
        for poly, col in ((roof, _ROOF_BGR), (floor, _FLOOR_BGR)):
            if poly is None or len(poly) < 2:
                continue
            pts = np.array([_scale(x, y, mask_h, mask_w, fh, fw) for x, y in poly],
                           np.int32)
            cv2.polylines(frame, [pts], False, col, 2)

        writer.write(frame)

    if cap is not None:
        cap.release()
    writer.release()


# ── Per-speaker processing ───────────────────────────────────────────────────

def _discover_speakers():
    if not SPK_BASE:
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith(SPK_BASE)
        and (d / "sam_seg" / "masks").is_dir()
    )


def _build_speaker_grid(mask_files, spk, n_gridlines):
    """Aggregate occupancy across all of the speaker's videos, derive landmarks,
    and build the fixed semipolar grid + aggregate roof arc."""
    subs = REGION_SUBS
    acc = {s: None for s in subs}
    cnt = {s: 0 for s in subs}
    for mp in mask_files:
        data = np.load(mp)
        keys = list(data.keys())
        for s in subs:
            k = _find_mask_key(keys, s)
            if k is None:
                continue
            occ = data[k].astype(np.float32).mean(axis=0)
            acc[s] = occ if acc[s] is None else acc[s] + occ
            cnt[s] += 1
    region_agg = {}
    for s in subs:
        if acc[s] is not None and cnt[s] > 0:
            region_agg[s] = (acc[s] / cnt[s]) > 0.15
        else:
            region_agg[s] = None
    landmarks = build_landmarks(region_agg)
    roof_arc = trace_roof(region_agg)
    grid = build_semipolar_grid(landmarks, roof_arc, n_gridlines)
    return grid, landmarks, region_agg, roof_arc


def process_speaker(spk, n_gridlines, n_videos, n_bins):
    base = DATA_DIR / spk if spk is not None else DATA_DIR
    label = spk if spk is not None else DATA_DIR.name
    mask_dir = base / "sam_seg" / "masks"
    video_dir = base / VIDEO_DIR
    out_dir = base / "vtd"

    pattern = f"{spk}_*.npz" if spk is not None else "*.npz"
    mask_files = sorted(mask_dir.glob(pattern))
    if not mask_files:
        print(f"  No mask files in {mask_dir}")
        return

    print(f"  Building fixed grid from {len(mask_files)} videos ...")
    grid, landmarks, region_agg, roof_arc = _build_speaker_grid(
        mask_files, spk, n_gridlines)

    for sub in ("pts", "norm", "hist", "raw", "grid", "diagnostic"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # persist the fixed grid + landmarks
    with open(out_dir / "grid" / f"{label}_grid.json", "w") as f:
        json.dump({
            "n_gridlines": n_gridlines,
            "landmarks": {k: v.tolist() for k, v in landmarks.items()},
            "grid": [{"origin": g["origin"].tolist(), "dir": g["dir"].tolist(),
                      "section": g["section"]} for g in grid],
        }, f, indent=2)

    # Pass 1: compute raw VTD (+ per-frame points) for every video
    from tqdm import tqdm
    per_video = {}
    all_vtd = []
    for mp in tqdm(mask_files, desc=f"  {label} VTD"):
        basename = mp.stem[len(spk) + 1:] if spk is not None else mp.stem
        regions, T = _load_regions(mp)
        if T == 0:
            continue
        vtd = np.full((T, n_gridlines), np.nan, np.float32)
        U_f = np.full((T, n_gridlines, 2), np.nan, np.float32)
        L_f = np.full((T, n_gridlines, 2), np.nan, np.float32)
        roof_f, floor_f = [None] * T, [None] * T
        for t in range(T):
            fr = _frame_regions(regions, t)
            roof = trace_roof(fr)
            floor = trace_floor(fr)
            roof_f[t], floor_f[t] = roof, floor
            v, U, Lp = compute_vtd(grid, roof, floor)
            vtd[t], U_f[t], L_f[t] = v, U, Lp
        np.save(out_dir / "pts" / f"{basename}.npy", vtd)
        with open(out_dir / "raw" / f"{basename}.json", "w") as f:
            json.dump({"U": U_f.tolist(), "L": L_f.tolist()}, f)
        per_video[basename] = (mp, regions, T, roof_f, floor_f, U_f, L_f)
        all_vtd.append(vtd)

    if not all_vtd:
        return

    # Per-speaker global min-max per grid line (Shi Eq. 3). Grid lines that
    # never intersect the airway (all-NaN columns) are normalized to 0/1 safely.
    stacked = np.concatenate(all_vtd, axis=0)          # (sumT, L)
    all_nan = np.all(np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        vmin = np.where(all_nan, 0.0, np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), axis=0))
        vmax = np.where(all_nan, 1.0, np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0))
    rng = np.where((vmax - vmin) > 1e-6, vmax - vmin, 1.0)

    # Pass 2: write normalized VTD + histograms
    for basename, (mp, regions, T, rf, ff, U_f, L_f) in per_video.items():
        vtd = np.load(out_dir / "pts" / f"{basename}.npy")
        norm = (vtd - vmin[None, :]) / rng[None, :]
        norm = np.clip(norm, 0.0, 1.0)
        np.save(out_dir / "norm" / f"{basename}.npy", norm.astype(np.float32))
        hist = np.zeros((n_gridlines, n_bins), np.float32)
        for l in range(n_gridlines):
            col = norm[:, l]
            col = col[np.isfinite(col)]
            if col.size:
                h, _ = np.histogram(col, bins=n_bins, range=(0.0, 1.0))
                hist[l] = h
        np.save(out_dir / "hist" / f"{basename}.npy", hist)

    # Static diagnostic: one random frame from one random video
    rng_r = random.Random(sum(ord(c) for c in label))
    dbase = rng_r.choice(list(per_video.keys()))
    mp, regions, T, rf, ff, U_f, L_f = per_video[dbase]
    ti = T // 2
    save_static_diagnostic(
        out_dir / "diagnostic" / f"{label}_grid_frame.png",
        _frame_regions(regions, ti), rf[ti], ff[ti], grid, U_f[ti], L_f[ti],
        landmarks)

    # Diagnostic videos for up to n_videos random videos
    diag_bases = rng_r.sample(list(per_video.keys()),
                              min(n_videos, len(per_video)))
    for basename in tqdm(diag_bases, desc=f"  {label} diag videos"):
        mp, regions, T, rf, ff, U_f, L_f = per_video[basename]
        vpath = video_dir / f"{basename}.avi"
        write_diagnostic_video(
            out_dir / "diagnostic" / f"{basename}_vtd.mp4",
            regions, T, vpath if vpath.exists() else None,
            grid, rf, ff, U_f, L_f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    single = not SPK_BASE
    p = argparse.ArgumentParser(description="Extract vocal-tract distance (VTD) from SAM2 masks.")
    if not single:
        p.add_argument("--spk", nargs="+", type=int, default=None, metavar="N",
                       help=f"Speaker numbers (prefix '{SPK_BASE}'). Default: all.")
    p.add_argument("--n-gridlines", type=int, default=N_GRIDLINES,
                   help="TOTAL number of grid lines (default from config).")
    p.add_argument("--n-videos", type=int, default=N_DIAGNOSTIC,
                   help="Number of diagnostic videos per speaker.")
    p.add_argument("--bins", type=int, default=N_BINS,
                   help="Histogram bins per grid line (default from config).")
    args = p.parse_args()

    if single:
        print(f"\n[{DATA_DIR.name}] (single speaker)")
        process_speaker(None, args.n_gridlines, args.n_videos, args.bins)
    else:
        allspk = _discover_speakers()
        if not allspk:
            print(f"No speaker dirs matching '{SPK_BASE}*' in {DATA_DIR}")
            sys.exit(1)
        speakers = [f"{SPK_BASE}{n}" for n in args.spk] if args.spk else allspk
        for s in speakers:
            if s not in allspk:
                print(f"Unknown speaker: {s} (valid: {allspk})")
                sys.exit(1)
        for s in speakers:
            print(f"\n[{s}]")
            process_speaker(s, args.n_gridlines, args.n_videos, args.bins)

    print("\nDone.")


if __name__ == "__main__":
    main()
