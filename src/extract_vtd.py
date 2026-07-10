#!/usr/bin/env python3
# ABOUTME: Computes vocal-tract distance (VTD) between two traced VT walls from SAM2 masks.
# ABOUTME: Roof = lips->palate->velum->pharyngeal wall; floor = lower-lip + tongue upper surface to root. Outputs (T,L) VTD + diagnostics over the MRI frame.
"""
extract_vtd.py — Vocal Tract Distance from SAM2 masks.

Two boundary lines are traced per frame and VTD is the distance between them
along L grid lines that connect corresponding (equal arc-length) points, so the
grid follows the natural bend of the tract. No lingual origin, no semipolar /
Proctor construction — just two lines:

  ROOF  (upper wall), one continuous line, front -> back:
    bottom (airway-facing) edge of "upper lip - palate"   [lips -> hard palate]
      -> straight bridge -> bottom edge of "velum"
      -> straight bridge from the velum's bottom-right point to its CLOSEST
         point on the pharyngeal wall
      -> pharyngeal-wall (airway-facing / left) edge from that junction DOWN to
         the bottom (near where it meets the tongue). The wall is NOT traced
         upward past the junction.

  FLOOR (lower wall), one continuous line, front -> back:
    top (airway-facing) edge of "lower lip - jaw"
      -> straight bridge to the tongue's front (left-most) point
      -> tongue upper surface (existing contour method) down to the tongue root.

The posterior-most grid line therefore connects the tongue root/bottom to the
pharyngeal wall, and the anterior-most connects the two lips (lip aperture).

Masks are anti-alias smoothed before tracing (upsample -> Gaussian blur ->
threshold) so staircase pixelation does not inflate the distances.

Outputs (per speaker, under {data_dir}/[spk/]vtd/):
  pts/{basename}.npy    (T, L)      raw VTD in pixels
  norm/{basename}.npy   (T, L)      per-speaker min-max normalized (0=closed,1=open)
  hist/{basename}.npy   (L, bins)   per-gridline histogram of normalized VTD
  lines/{basename}.npz  roof,floor  (T, L, 2) resampled wall points, for QA
  diagnostic/{spk}_frame.png         one MRI frame + masks + lines + VTD points
  diagnostic/{basename}_vtd.mp4      per-frame MRI overlay for --n-videos videos

Speaker convention: face-left. Front of mouth = low x (left); back/pharynx =
high x (right); roof = low y (top); floor = high y (bottom).

Usage:
    conda run -n myenv python extract_vtd.py [--spk 2 3 ...] \
        [--n-gridlines 40] [--n-videos 5] [--bins 20] \
        [--upscale 8] [--pre-sigma 1.5] [--sigma-path 1.5]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label

# ── Config (same pattern as the other gesture_tools scripts) ─────────────────
_DEFAULT_CFG = {
    "data_dir": ".",
    "n_diagnostic": 5,
    "spk_base": "",
    "video_dir": "video",
    "dataset": "lss",
    "n_gridlines": 40,
    "n_bins": 20,
    "upscale": 8,
    "pre_sigma": 1.5,
    "sigma_path": 1.5,
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
UPSCALE = int(_cfg.get("upscale", 8))
PRE_SIGMA = float(_cfg.get("pre_sigma", 1.5))
SIGMA_PATH = float(_cfg.get("sigma_path", 1.5))

<<<<<<< HEAD
# Region key substrings (case-insensitive). Five segmented regions; no larynx.
ROOF_FRONT_SUB = "upper lip"   # "upper lip - palate" (lips + hard palate)
=======
# Region key substrings (case-insensitive). Pharyngeal wall auto-detects "pharyn".
# There is no larynx/glottis mask; the posterior end of the tract is defined by
# the bottom of the tongue and its closest point on the pharyngeal wall.
ROOF_FRONT_SUB = "upper lip"  # "upper lip - palate"
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
VELUM_SUB = "velum"
PHARYNX_SUB = "pharyn"  # "pharyngeal wall"
TONGUE_SUB = "tongue"
<<<<<<< HEAD
LOWER_LIP_SUB = "lower lip"    # "lower lip - jaw"
=======
LOWER_LIP_SUB = "lower lip"  # "lower lip - jaw"

# The five segmented regions, in canonical order.
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
REGION_SUBS = [ROOF_FRONT_SUB, VELUM_SUB, PHARYNX_SUB, TONGUE_SUB, LOWER_LIP_SUB]


# ── Mask helpers ─────────────────────────────────────────────────────────────


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


<<<<<<< HEAD
def smooth_mask(mask2d: np.ndarray, upscale: int = UPSCALE,
                pre_sigma: float = PRE_SIGMA) -> np.ndarray:
    """Anti-alias a binary mask: keep the largest component, upsample (cubic),
    Gaussian-blur, threshold. Returns an (H*upscale, W*upscale) uint8 mask.
    Tracing on this removes staircase pixelation so VTD isn't inflated."""
    if mask2d is None or not mask2d.any():
=======
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
        return loop[i : i + 1], loop[i : i + 1]
    fwd_idx = [(i + k) % n for k in range(((j - i) % n) + 1)]
    bwd_idx = [(i - k) % n for k in range(((i - j) % n) + 1)]
    return loop[fwd_idx], loop[bwd_idx]


def _bottom_arc_ordered(mask):
    """Bottom (high-y / airway-facing) arc of a mask, oriented front (low x) ->
    back (high x). Used for the palate and velum roof."""
    loops = _ordered_loops(mask)
    if not loops:
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
        return None
    core = _largest_component(mask2d).astype(np.float32)
    H, W = core.shape
    up = cv2.resize(core, (W * upscale, H * upscale), interpolation=cv2.INTER_CUBIC)
    up = gaussian_filter(up, sigma=pre_sigma)
    return (up > 0.5).astype(np.uint8)


# ── Edge tracing (operate on the upscaled smoothed mask) ─────────────────────

def _bottom_edge(mask_up: np.ndarray) -> np.ndarray:
    """Airway-facing bottom edge: per column, the max-y pixel. Ascending x."""
    ys, xs = np.where(mask_up)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux, idx = np.unique(xs, return_index=True)
    max_y = np.maximum.reduceat(ys, idx)
    return np.stack([ux, max_y], 1).astype(np.float32)


def _top_edge(mask_up: np.ndarray) -> np.ndarray:
    """Airway-facing top edge: per column, the min-y pixel. Ascending x."""
    ys, xs = np.where(mask_up)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux, idx = np.unique(xs, return_index=True)
    min_y = np.minimum.reduceat(ys, idx)
    return np.stack([ux, min_y], 1).astype(np.float32)


def _left_edge(mask_up: np.ndarray) -> np.ndarray:
    """Airway-facing left edge: per row, the min-x pixel. Ascending y (top->bottom)."""
    ys, xs = np.where(mask_up)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)
    order = np.argsort(ys)
    ys, xs = ys[order], xs[order]
    uy, idx = np.unique(ys, return_index=True)
    min_x = np.minimum.reduceat(xs, idx)
    return np.stack([min_x, uy], 1).astype(np.float32)


def _trim_wall_bottom(pts: np.ndarray) -> np.ndarray:
    """Strip the horizontal curl at the bottom of the pharyngeal-wall trace
    (pts sorted ascending y). Scans up from the bottom removing trailing
    segments that move more horizontally than vertically."""
    if len(pts) < 2:
        return pts
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    cut = len(pts)
    for i in range(len(dx) - 1, -1, -1):
        if abs(dx[i]) > abs(dy[i]):
            cut = i + 1
        else:
            break
    return pts if cut < 2 else pts[:cut]


def _bridge(p1, p2, spacing=1.0):
    """Interior points of a straight line between p1 and p2 (~spacing apart)."""
    d = float(np.linalg.norm(p2 - p1))
    n = int(round(d / spacing)) - 1
    if n <= 0:
        return np.empty((0, 2), np.float32)
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return (p1 + np.outer(t, p2 - p1)).astype(np.float32)


# ── Tongue upper surface (existing contour method) ───────────────────────────

def _find_jaw_anchor(tongue_masks, lower_lip_masks):
    """Median tongue-contour point closest to the lower lip, across frames (a
    stable anterior anchor). Computed on original-resolution masks."""
    T = tongue_masks.shape[0]
    junction = []
    for t in range(T):
        tm, lm = tongue_masks[t], lower_lip_masks[t]
        if not tm.any() or not lm.any():
            continue
        core = _largest_component(tm).astype(np.uint8) * 255
        cs, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        pts = max(cs, key=len).squeeze()
        if pts.ndim != 2 or len(pts) < 4:
            continue
        ly, lx = np.where(lm)
        dx = pts[:, 0:1].astype(np.float32) - lx[None, :]
        dy = pts[:, 1:2].astype(np.float32) - ly[None, :]
        junction.append(pts[int((dx ** 2 + dy ** 2).min(1).argmin())].astype(np.float32))
    if not junction:
        return None
    a = np.stack(junction)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def extract_upper_contour(mask_up, jaw_ref_up):
    """Upper (oral-cavity-facing) tongue surface from a single upscaled mask.
    Splits the outer contour at anterior (jaw junction) and posterior (root =
    right-most) anchors and keeps the upper path. Returns (M, 2) anterior->
    posterior in UPSCALED coords, or None."""
    if mask_up is None or not mask_up.any():
        return None
<<<<<<< HEAD
    cs, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
=======
    loop = max(loops, key=len)
    T = int(np.argmin(loop[:, 1]))  # top
    B = int(np.argmax(loop[:, 1]))  # bottom
    fwd, bwd = _arc_between(loop, T, B)
    # airway-facing arc has the smaller mean x (anterior side)
    arc = fwd if fwd[:, 0].mean() < bwd[:, 0].mean() else bwd
    if len(arc) < 2:
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
        return None
    pts = max(cs, key=len).squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return None
<<<<<<< HEAD
    idx_root = int(pts[:, 0].argmax())
    if jaw_ref_up is not None:
        rx, ry = jaw_ref_up
        idx_j = int(((pts[:, 0] - rx) ** 2 + (pts[:, 1] - ry) ** 2).argmin())
=======
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
    L = int(np.argmin(loop[:, 0]))  # tip (front)
    B = int(np.argmax(loop[:, 1]))  # bottom (posterior-inferior)
    Tp = int(np.argmin(loop[:, 1]))  # airway-facing apex (top)
    if Tp in (L, B):
        fwd, bwd = _arc_between(loop, L, B)
        arc = fwd if fwd[:, 1].mean() < bwd[:, 1].mean() else bwd
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    else:
        idx_j = int(pts[:, 0].argmin())
    a, b = sorted([idx_j, idx_root])
    path_a = pts[a:b + 1]
    path_b = np.concatenate([pts[b:], pts[:a + 1]])
    upper = path_a if path_a[:, 1].mean() <= path_b[:, 1].mean() else path_b
    if upper[0, 0] > upper[-1, 0]:
        upper = upper[::-1]
    return upper.astype(np.float32)


# ── Path smoothing & resampling ──────────────────────────────────────────────

def _smooth_path(line, sigma):
    """Gaussian-smooth an open (M,2) polyline along its path (mode=nearest)."""
    if line is None or len(line) < 3 or sigma <= 0:
        return line
    out = line.astype(np.float32, copy=True)
    out[:, 0] = gaussian_filter1d(out[:, 0], sigma=sigma, mode="nearest")
    out[:, 1] = gaussian_filter1d(out[:, 1], sigma=sigma, mode="nearest")
    return out


def _resample(line, n):
    """Arc-length resample an (M,2) polyline to exactly n points. Returns (n,2)."""
    if line is None or len(line) < 2:
        return None
<<<<<<< HEAD
    seg = np.sqrt((np.diff(line, axis=0) ** 2).sum(1))
=======
    if arc[0, 0] > arc[-1, 0]:  # front (low x) first
        arc = arc[::-1]
    return arc


def _resample_polyline(poly, step=0.5):
    """Densify a polyline to ~step-pixel spacing (arc-length) for robust
    intersection tests. Returns (M, 2)."""
    if poly is None or len(poly) < 2:
        return poly
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(axis=1))
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.tile(line[0], (n, 1)).astype(np.float32)
    s = np.linspace(0, total, n)
    x = np.interp(s, cum, line[:, 0])
    y = np.interp(s, cum, line[:, 1])
    return np.stack([x, y], 1).astype(np.float32)


# ── Wall line assembly (original-resolution coords) ──────────────────────────

<<<<<<< HEAD
def build_roof(reg_up: dict):
    """One line: lips/palate bottom edge -> velum bottom edge -> pharyngeal
    wall (from the velum-junction down). Returns (M,2) front->back in original
    coords, or None."""
    U = UPSCALE
    palate = reg_up.get(ROOF_FRONT_SUB)
    if palate is None:
=======

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
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
        return None
    pal = _bottom_edge(palate)
    if len(pal) < 2:
        return None
    parts = [pal]
    tail = pal[-1]

    velum = reg_up.get(VELUM_SUB)
    if velum is not None:
        vel = _bottom_edge(velum)
        if len(vel) >= 2:
            br = _bridge(tail, vel[0])
            if len(br):
                parts.append(br)
            parts.append(vel)
            tail = vel[-1]              # velum bottom-right

    wall = reg_up.get(PHARYNX_SUB)
    if wall is not None:
        wl = _left_edge(wall)
        if len(wl) >= 2:
            # closest wall point to the velum bottom-right (or palate tail)
            j = int(((wl - tail[None, :]) ** 2).sum(1).argmin())
            seg = _trim_wall_bottom(wl[j:])       # from junction DOWN only
            if len(seg) >= 1:
                br = _bridge(tail, seg[0])
                if len(br):
                    parts.append(br)
                parts.append(seg)

    line = np.concatenate(parts, axis=0) / U
    return _smooth_path(line, SIGMA_PATH)


def build_floor(reg_up: dict, jaw_ref_up):
    """One line: lower-lip top edge -> tongue upper surface (tip -> root).
    Returns (M,2) front->back in original coords, or None."""
    U = UPSCALE
    tongue = reg_up.get(TONGUE_SUB)
    upper = extract_upper_contour(tongue, jaw_ref_up) if tongue is not None else None
    if upper is None or len(upper) < 2:
        return None
    parts = []
    lower = reg_up.get(LOWER_LIP_SUB)
    if lower is not None:
        low = _top_edge(lower)
        if len(low) >= 2:
            parts.append(low)
            br = _bridge(low[-1], upper[0])
            if len(br):
                parts.append(br)
    parts.append(upper)
    line = np.concatenate(parts, axis=0) / U
    return _smooth_path(line, SIGMA_PATH)


# ── VTD ──────────────────────────────────────────────────────────────────────

def compute_vtd(roof, floor, n):
    """Resample both walls to n corresponding (equal arc-length) points and
    measure the distance between them. Returns (vtd (n,), roof_pts (n,2),
    floor_pts (n,2)) or (nan, nan, nan) if either wall is missing."""
    r = _resample(roof, n)
    f = _resample(floor, n)
    if r is None or f is None:
        nan1 = np.full(n, np.nan, np.float32)
        nan2 = np.full((n, 2), np.nan, np.float32)
        return nan1, nan2, nan2
    vtd = np.linalg.norm(r - f, axis=1).astype(np.float32)
    return vtd, r, f


<<<<<<< HEAD
# ── NPZ / frame helpers ──────────────────────────────────────────────────────
=======
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
    i_u, i_l = np.unravel_index((dx**2 + dy**2).argmin(), (len(ux), len(lx)))
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
            if d[1] < 0:  # ensure it points toward the floor (downward)
                d = -d
        grid.append({"origin": p, "dir": d, "section": section})

    # Pin the posterior-most line to tongue_bottom -> pharyngeal_end exactly.
    tb = landmarks["tongue_bottom"]
    pe = landmarks["pharyngeal_end"]
    v = tb - pe
    nv = np.linalg.norm(v)
    grid[-1] = {
        "origin": pe.astype(np.float32),
        "dir": (v / nv if nv > 1e-6 else np.array([-1.0, 0.0], np.float32)).astype(
            np.float32
        ),
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
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6


def _load_regions(mask_path: Path):
    data = np.load(mask_path)
    keys = list(data.keys())
    regions = {}
    for sub in REGION_SUBS:
        k = _find_mask_key(keys, sub)
        regions[sub] = data[k].astype(bool) if k is not None else None
    T = next((m.shape[0] for m in regions.values() if m is not None), 0)
    return regions, T


def _frame_walls(regions, t, jaw_ref):
    """Smooth-upscale each region at frame t, then trace roof & floor."""
    reg_up = {}
    for sub, m in regions.items():
        reg_up[sub] = smooth_mask(m[t]) if (m is not None and t < m.shape[0]) else None
    jaw_up = (jaw_ref[0] * UPSCALE, jaw_ref[1] * UPSCALE) if jaw_ref else None
    return build_roof(reg_up), build_floor(reg_up, jaw_up), reg_up


# ── Diagnostics (over the MRI frame) ─────────────────────────────────────────

<<<<<<< HEAD
# BGR for cv2 video overlay
_REGION_BGR = {
    ROOF_FRONT_SUB: (75, 180, 60),    # green   (upper lip - palate)
    LOWER_LIP_SUB: (75, 25, 230),     # red     (lower lip - jaw)
    TONGUE_SUB: (216, 99, 67),        # blue    (tongue)
    VELUM_SUB: (180, 30, 145),        # purple  (velum)
    PHARYNX_SUB: (49, 130, 245),      # orange  (pharyngeal wall)
}
_ROOF_BGR = (0, 200, 0)      # green line
_FLOOR_BGR = (0, 0, 255)     # red line
_GRID_BGR = (255, 255, 0)    # cyan grid
_VTD_BGR = (0, 255, 255)     # yellow VTD points


def _read_frame(video_path, t, mask_hw):
    """Return the MRI frame t (BGR) or a blank canvas sized to the mask."""
    mh, mw = mask_hw
    if video_path is not None and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
    return np.full((mh * 6, mw * 6, 3), 20, np.uint8)


def _overlay(frame, regions, t, roof, floor, vtd_r, vtd_f):
    fh, fw = frame.shape[:2]
    mh = mw = None
=======
# matplotlib tab10-ish colors per region for the static figure
_DIAG_COLORS = {
    ROOF_FRONT_SUB: "#1f77b4",  # palate - blue
    LOWER_LIP_SUB: "#ff7f0e",  # lower lip - orange
    TONGUE_SUB: "#2ca02c",  # tongue - green
    VELUM_SUB: "#d62728",  # velum - red
    PHARYNX_SUB: "#9467bd",  # pharyngeal wall - purple
}
# BGR for the cv2 video
_ROOF_BGR = (0, 0, 255)  # red
_FLOOR_BGR = (255, 128, 0)  # blue
_GRID_BGR = (255, 255, 0)  # cyan (Shi Fig 1b grid)
_VTD_BGR = (0, 255, 255)  # yellow (Shi Fig 1b VTD segments)


def save_static_diagnostic(
    out_path, frame_regions, roof, floor, grid, U, Lp, landmarks
):
    """One matplotlib figure: masks (translucent) + roof/floor edges + grid +
    VTD points, for a single representative frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, MAX_XY)
    ax.set_ylim(MAX_XY, 0)  # image coords (y down)
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
        ax.scatter(
            [u[0], l[0]], [u[1], l[1]], s=10, color="yellow", zorder=5, linewidths=0
        )

    if roof is not None:
        ax.plot(roof[:, 0], roof[:, 1], color="red", lw=2.0, zorder=4, label="roof")
    if floor is not None:
        ax.plot(
            floor[:, 0],
            floor[:, 1],
            color="deepskyblue",
            lw=2.0,
            zorder=4,
            label="floor",
        )
    for name, p in landmarks.items():
        ax.scatter([p[0]], [p[1]], s=45, marker="+", color="white", zorder=6)
        ax.annotate(
            name,
            (p[0], p[1]),
            fontsize=6,
            color="white",
            xytext=(2, -2),
            textcoords="offset points",
        )

    ax.set_title(f"VTD grid ({len(grid)} lines)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scale(x, y, mask_h, mask_w, fh, fw):
    return int(round(x * fw / mask_w)), int(round(y * fh / mask_h))


def write_diagnostic_video(
    out_path,
    regions,
    T,
    video_path,
    grid,
    roof_frames,
    floor_frames,
    U_frames,
    L_frames,
):
    """Per-frame overlay: masks (translucent), roof/floor edges, grid lines,
    VTD points."""
    mask_h = mask_w = MAX_XY
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    for m in regions.values():
        if m is not None:
            mh, mw = m.shape[1], m.shape[2]
            break
    if mh is None:
        return frame
    sx, sy = fw / mw, fh / mh

    for sub, m in regions.items():
        if m is None or t >= m.shape[0] or not m[t].any():
            continue
        mr = cv2.resize(m[t].astype(np.uint8) * 255, (fw, fh),
                        interpolation=cv2.INTER_NEAREST)
        colored = np.zeros_like(frame)
        colored[mr > 0] = _REGION_BGR.get(sub, (150, 150, 150))
        cv2.addWeighted(colored, 0.35, frame, 1.0, 0, frame)

    def _poly(line, color):
        if line is None or len(line) < 2:
            return
        pts = np.array([[int(x * sx), int(y * sy)] for x, y in line], np.int32)
        cv2.polylines(frame, [pts], False, color, 2)

    _poly(roof, _ROOF_BGR)
    _poly(floor, _FLOOR_BGR)
    if vtd_r is not None and vtd_f is not None:
        for u, l in zip(vtd_r, vtd_f):
            if np.isnan(u[0]) or np.isnan(l[0]):
                continue
            p1 = (int(u[0] * sx), int(u[1] * sy))
            p2 = (int(l[0] * sx), int(l[1] * sy))
            cv2.line(frame, p1, p2, _GRID_BGR, 1)
            cv2.circle(frame, p1, 3, _VTD_BGR, -1)
            cv2.circle(frame, p2, 3, _VTD_BGR, -1)
    return frame


def save_static_diagnostic(out_path, regions, t, video_path, roof, floor, r, f):
    frame = _read_frame(video_path, t, (regions_first_hw(regions)))
    frame = _overlay(frame, regions, t, roof, floor, r, f)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def regions_first_hw(regions):
    for m in regions.values():
        if m is not None:
            return m.shape[1], m.shape[2]
    return 104, 104


def write_diagnostic_video(out_path, regions, T, video_path, n_gridlines, jaw_ref):
    mh, mw = regions_first_hw(regions)
    if video_path is not None and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or mw * 6
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or mh * 6
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T
    else:
        cap = None
<<<<<<< HEAD
        fps, fw, fh, n_frames = 50.0, mw * 6, mh * 6, T

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (fw, fh))
    for t in range(min(T, n_frames)):
=======
        fps = 50.0
        fh, fw = mask_h * 5, mask_w * 5  # upscale blank canvas for legibility

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
    )
    region_bgr = {
        ROOF_FRONT_SUB: (180, 120, 30),
        LOWER_LIP_SUB: (30, 140, 240),
        TONGUE_SUB: (40, 180, 40),
        VELUM_SUB: (40, 40, 200),
        PHARYNX_SUB: (180, 100, 150),
    }

    for t in range(T):
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
<<<<<<< HEAD
            frame = np.full((fh, fw, 3), 20, np.uint8)
        roof, floor, _ = _frame_walls(regions, t, jaw_ref)
        vtd, r, f = compute_vtd(roof, floor, n_gridlines)
        _overlay(frame, regions, t, roof, floor, r, f)
=======
            frame = np.zeros((fh, fw, 3), np.uint8)

        overlay = frame.copy()
        for sub, m in regions.items():
            if m is None or t >= m.shape[0] or not m[t].any():
                continue
            mr = cv2.resize(
                m[t].astype(np.uint8) * 255, (fw, fh), interpolation=cv2.INTER_NEAREST
            )
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
            pts = np.array(
                [_scale(x, y, mask_h, mask_w, fh, fw) for x, y in poly], np.int32
            )
            cv2.polylines(frame, [pts], False, col, 2)

>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
        writer.write(frame)
    if cap is not None:
        cap.release()
    writer.release()


# ── Per-speaker processing ───────────────────────────────────────────────────


def _discover_speakers():
    if not SPK_BASE:
        return []
    return sorted(
        d.name
        for d in DATA_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith(SPK_BASE)
        and (d / "sam_seg" / "masks").is_dir()
    )


def process_speaker(spk, n_gridlines, n_videos, n_bins):
    from tqdm import tqdm
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
<<<<<<< HEAD
    for sub in ("pts", "norm", "hist", "lines", "diagnostic"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    per_video = {}   # basename -> (mask_path, video_path, T)
=======

    print(f"  Building fixed grid from {len(mask_files)} videos ...")
    grid, landmarks, region_agg, roof_arc = _build_speaker_grid(
        mask_files, spk, n_gridlines
    )

    for sub in ("pts", "norm", "hist", "raw", "grid", "diagnostic"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # persist the fixed grid + landmarks
    with open(out_dir / "grid" / f"{label}_grid.json", "w") as f:
        json.dump(
            {
                "n_gridlines": n_gridlines,
                "landmarks": {k: v.tolist() for k, v in landmarks.items()},
                "grid": [
                    {
                        "origin": g["origin"].tolist(),
                        "dir": g["dir"].tolist(),
                        "section": g["section"],
                    }
                    for g in grid
                ],
            },
            f,
            indent=2,
        )

    # Pass 1: compute raw VTD (+ per-frame points) for every video
    from tqdm import tqdm

    per_video = {}
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    all_vtd = []
    for mp in tqdm(mask_files, desc=f"  {label} VTD"):
        basename = mp.stem[len(spk) + 1 :] if spk is not None else mp.stem
        regions, T = _load_regions(mp)
        if T == 0:
            continue
        jaw_ref = _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB]) \
            if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None else None
        vtd = np.full((T, n_gridlines), np.nan, np.float32)
        roof_pts = np.full((T, n_gridlines, 2), np.nan, np.float32)
        floor_pts = np.full((T, n_gridlines, 2), np.nan, np.float32)
        for t in range(T):
            roof, floor, _ = _frame_walls(regions, t, jaw_ref)
            v, r, f = compute_vtd(roof, floor, n_gridlines)
            vtd[t], roof_pts[t], floor_pts[t] = v, r, f
        np.save(out_dir / "pts" / f"{basename}.npy", vtd)
        np.savez(out_dir / "lines" / f"{basename}.npz", roof=roof_pts, floor=floor_pts)
        vpath = video_dir / f"{basename}.avi"
        per_video[basename] = (mp, vpath if vpath.exists() else None, T)
        all_vtd.append(vtd)

    if not all_vtd:
        return

<<<<<<< HEAD
    # Per-speaker global min-max per grid line (Shi Eq. 3)
    stacked = np.concatenate(all_vtd, axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        vmin = np.where(all_nan, 0.0, np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), 0))
        vmax = np.where(all_nan, 1.0, np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), 0))
=======
    # Per-speaker global min-max per grid line (Shi Eq. 3). Grid lines that
    # never intersect the airway (all-NaN columns) are normalized to 0/1 safely.
    stacked = np.concatenate(all_vtd, axis=0)  # (sumT, L)
    all_nan = np.all(np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        vmin = np.where(
            all_nan,
            0.0,
            np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), axis=0),
        )
        vmax = np.where(
            all_nan,
            1.0,
            np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0),
        )
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    rng = np.where((vmax - vmin) > 1e-6, vmax - vmin, 1.0)

    for basename in per_video:
        vtd = np.load(out_dir / "pts" / f"{basename}.npy")
        norm = np.clip((vtd - vmin[None, :]) / rng[None, :], 0.0, 1.0)
        np.save(out_dir / "norm" / f"{basename}.npy", norm.astype(np.float32))
        hist = np.zeros((n_gridlines, n_bins), np.float32)
        for l in range(n_gridlines):
            col = norm[:, l][np.isfinite(norm[:, l])]
            if col.size:
                hist[l], _ = np.histogram(col, bins=n_bins, range=(0.0, 1.0))
        np.save(out_dir / "hist" / f"{basename}.npy", hist)

    # Diagnostics: one static frame + up to n_videos overlay videos
    rng_r = random.Random(sum(ord(c) for c in label))
    names = list(per_video.keys())
    dbase = rng_r.choice(names)
    mp, vpath, T = per_video[dbase]
    regions, _ = _load_regions(mp)
    jaw_ref = _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB]) \
        if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None else None
    ti = T // 2
<<<<<<< HEAD
    roof, floor, _ = _frame_walls(regions, ti, jaw_ref)
    _, r, f = compute_vtd(roof, floor, n_gridlines)
    save_static_diagnostic(out_dir / "diagnostic" / f"{label}_frame.png",
                           regions, ti, vpath, roof, floor, r, f)

    for basename in tqdm(rng_r.sample(names, min(n_videos, len(names))),
                         desc=f"  {label} diag videos"):
        mp, vpath, T = per_video[basename]
        regions, _ = _load_regions(mp)
        jaw_ref = _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB]) \
            if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None else None
        write_diagnostic_video(out_dir / "diagnostic" / f"{basename}_vtd.mp4",
                               regions, T, vpath, n_gridlines, jaw_ref)
=======
    save_static_diagnostic(
        out_dir / "diagnostic" / f"{label}_grid_frame.pdf",
        _frame_regions(regions, ti),
        rf[ti],
        ff[ti],
        grid,
        U_f[ti],
        L_f[ti],
        landmarks,
    )

    # Diagnostic videos for up to n_videos random videos
    diag_bases = rng_r.sample(list(per_video.keys()), min(n_videos, len(per_video)))
    for basename in tqdm(diag_bases, desc=f"  {label} diag videos"):
        mp, regions, T, rf, ff, U_f, L_f = per_video[basename]
        vpath = video_dir / f"{basename}.avi"
        write_diagnostic_video(
            out_dir / "diagnostic" / f"{basename}_vtd.mp4",
            regions,
            T,
            vpath if vpath.exists() else None,
            grid,
            rf,
            ff,
            U_f,
            L_f,
        )
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    global UPSCALE, PRE_SIGMA, SIGMA_PATH
    single = not SPK_BASE
    p = argparse.ArgumentParser(
        description="Extract vocal-tract distance (VTD) from SAM2 masks."
    )
    if not single:
<<<<<<< HEAD
        p.add_argument("--spk", nargs="+", type=int, default=None, metavar="N",
                       help=f"Speaker numbers (prefix '{SPK_BASE}'). Default: all.")
    p.add_argument("--n-gridlines", type=int, default=N_GRIDLINES)
    p.add_argument("--n-videos", type=int, default=N_DIAGNOSTIC)
    p.add_argument("--bins", type=int, default=N_BINS)
    p.add_argument("--upscale", type=int, default=UPSCALE)
    p.add_argument("--pre-sigma", type=float, default=PRE_SIGMA)
    p.add_argument("--sigma-path", type=float, default=SIGMA_PATH)
=======
        p.add_argument(
            "--spk",
            nargs="+",
            type=int,
            default=None,
            metavar="N",
            help=f"Speaker numbers (prefix '{SPK_BASE}'). Default: all.",
        )
    p.add_argument(
        "--n-gridlines",
        type=int,
        default=N_GRIDLINES,
        help="TOTAL number of grid lines (default from config).",
    )
    p.add_argument(
        "--n-videos",
        type=int,
        default=N_DIAGNOSTIC,
        help="Number of diagnostic videos per speaker.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help="Histogram bins per grid line (default from config).",
    )
>>>>>>> 1970b3d0298ccdfb9a7ce735a14da632b6d5f6f6
    args = p.parse_args()
    UPSCALE, PRE_SIGMA, SIGMA_PATH = args.upscale, args.pre_sigma, args.sigma_path

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
