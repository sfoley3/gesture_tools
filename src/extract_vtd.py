#!/usr/bin/env python3
# ABOUTME: Computes vocal-tract distance (VTD) between two traced VT walls from SAM2 masks.
# ABOUTME: Roof = lips->palate->velum->pharyngeal wall; floor = lower-lip + tongue upper surface to root. Outputs (T,L) VTD + diagnostics over the MRI frame.
"""
extract_vtd.py — Vocal Tract Distance from SAM2 masks.

Two boundary lines are traced per frame and VTD is the closest distance between
them. No lingual origin, no semipolar / Proctor construction — just two lines:

  ROOF  (upper wall), one continuous line, front -> back:
    bottom (airway-facing) edge of "upper lip - palate"   [lips -> hard palate]
      -> spliced to the bottom edge of "velum" WHERE THEY MEET (closest pair),
         so the palate's posterior curl is dropped and no loop forms
      -> straight bridge from the velum's bottom-right point to its closest
         point on the pharyngeal wall
      -> pharyngeal-wall (airway-facing / left) edge DOWN only as far as the
         tongue reaches (constriction region; not to the wall's bottom).

  FLOOR (lower wall), one continuous line, front -> back:
    the lower lip's APERTURE point (top-most / inner-upper corner of "lower lip -
    jaw") connected straight to the tongue's airway-facing contour (dorsum from
    the jaw junction to the root, then down the posterior/backside edge to the
    bottom). The lower-lip/jaw top edge is NOT traced back toward the jaw — that
    would place floor points under the tongue. Constrictions are formed against
    the tongue upper surface, so the floor follows that surface.

VTD grid: THREE anchors — the lips, the center of the velum's lower edge, and
the tongue back — split each wall into an oral cavity (lips->velum) and a
pharyngeal cavity (velum->tongue back); the velum split on the floor is the point
CLOSEST to the velum center. Within each cavity both walls are arc-length
resampled to the same number of points and connected index-to-index, so each
line joins a point to its counterpart on the opposite wall. Connectors are
monotonic, so they never cross each other and never cut across the tongue
surface, and each cavity is filled with the same number of lines regardless of
its length (this controls for VT-length differences). VTD is the length of each
connector. Total lines L = 2n+3 (odd, default: n=5 -> 13) or 2n+2 (even,
--parity even), where n = --n-gridlines is the interior lines per cavity.

De-staircasing: by default the two lines are traced on the raw 104-px masks and
the derived polyline is Gaussian-smoothed (`sigma_path`) — fast and enough to
remove pixelation. Optionally set `upscale > 1` to anti-alias the masks
(upsample -> blur -> threshold) before tracing, at ~40x the cost.

Outputs (per speaker, under {data_dir}/[spk/]vtd/):
  pts/{basename}.npy    (T, L)      raw VTD in pixels
  norm/{basename}.npy   (T, L)      per-speaker min-max normalized (0=closed,1=open)
  hist/{basename}.npy   (L, bins)   per-gridline histogram of normalized VTD
  lines/{basename}.npz  roof,floor  (T, L, 2) grid endpoint points, for QA
  diagnostic/{spk}_frame.pdf         one MRI frame + masks + lines + VTD points
  diagnostic/{basename}_vtd.mp4      per-frame MRI overlay for --n-videos videos

Speaker convention: face-left. Front of mouth = low x (left); back/pharynx =
high x (right); roof = low y (top); floor = high y (bottom).

Usage:
    conda run -n myenv python extract_vtd.py [--spk 2 3 ...] \
        [--n-gridlines 10] [--parity even|odd] [--n-videos 5] [--bins 20] \
        [--upscale 1] [--pre-sigma 1.5] [--sigma-path 2.0]

grid_meta.json (written per speaker) records n_per_cavity, even_total, the total
line count, and the anchor indices [lips, velum, tongue_back].
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── Config (same pattern as the other gesture_tools scripts) ─────────────────
_DEFAULT_CFG = {
    "data_dir": ".",
    "n_diagnostic": 5,
    "spk_base": "",
    "video_dir": "video",
    "dataset": "lss",
    "n_gridlines": 40,
    "n_bins": 20,
    "upscale": 1,  # 1 = fast (trace raw mask, smooth the line); >1 = anti-alias masks
    "pre_sigma": 1.5,
    "sigma_path": 2.0,  # Gaussian smoothing of the derived line (pixels)
    "even_total": False,  # False -> 2n+3 grid lines (odd, default: n=5 -> 13); True -> 2n+2
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
UPSCALE = int(_cfg.get("upscale", 1))
PRE_SIGMA = float(_cfg.get("pre_sigma", 1.5))
SIGMA_PATH = float(_cfg.get("sigma_path", 2.0))
EVEN_TOTAL = bool(_cfg.get("even_total", False))
MAX_XY = 104  # mask side length (used as a ray-length cap)

# Region key substrings (case-insensitive). Five segmented regions; no larynx.
ROOF_FRONT_SUB = "upper lip"  # "upper lip - palate" (lips + hard palate)
VELUM_SUB = "velum"
PHARYNX_SUB = "pharyn"  # "pharyngeal wall"
TONGUE_SUB = "tongue"
LOWER_LIP_SUB = "lower lip"  # "lower lip - jaw"
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


def smooth_mask(
    mask2d: np.ndarray, upscale: int = None, pre_sigma: float = None
) -> np.ndarray:
    """Return the largest connected component as a uint8 mask, optionally
    anti-aliased by upsampling (cubic) + Gaussian blur + threshold.

    Default (upscale<=1) is the FAST path: no upsampling — the derived line is
    Gaussian-smoothed later by `_smooth_path`, which achieves the same de-
    staircasing at a fraction of the cost (the 8x upsample + blur on every
    region every frame is the script's main bottleneck). Set --upscale >1 only
    if you want the extra sub-pixel boundary before tracing.

    Reads the module globals when args are None so the --upscale CLI flag works
    (avoids the default-argument binding trap)."""
    if mask2d is None or not mask2d.any():
        return None
    up = UPSCALE if upscale is None else upscale
    core = _largest_component(mask2d)
    if up <= 1:
        return core.astype(np.uint8)
    ps = PRE_SIGMA if pre_sigma is None else pre_sigma
    H, W = core.shape
    big = cv2.resize(
        core.astype(np.float32), (W * up, H * up), interpolation=cv2.INTER_CUBIC
    )
    big = gaussian_filter(big, sigma=ps)
    return (big > 0.5).astype(np.uint8)


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
        junction.append(pts[int((dx**2 + dy**2).min(1).argmin())].astype(np.float32))
    if not junction:
        return None
    a = np.stack(junction)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def _walk_backside(pts, idx_root):
    """From the tongue root (right-most contour point) walk along the contour in
    the increasing-y direction (down the posterior/airway-facing edge), then
    TRUNCATE at the lowest point (max y). This follows the backside of a curled
    tongue root down to the bottom but does not continue leftward along the
    underside (which would make connectors cross). Returns (K,2) from the root
    to the bottom."""
    n = len(pts)
    y_next = pts[(idx_root + 1) % n, 1]
    y_prev = pts[(idx_root - 1) % n, 1]
    step = 1 if y_next >= y_prev else -1     # direction that goes downward
    path = [pts[idx_root]]
    running_max = float(pts[idx_root, 1])
    i = idx_root
    for _ in range(n // 2):
        j = (i + step) % n
        path.append(pts[j])
        running_max = max(running_max, float(pts[j, 1]))
        if float(pts[j, 1]) < running_max - 3.0:   # clearly past the bottom
            break
        i = j
    path = np.asarray(path, np.float32)
    cut = int(np.argmax(path[:, 1]))         # stop at the lowest (max-y) point
    return path[: cut + 1]


def extract_upper_contour(mask_up, jaw_ref_up):
    """Airway-facing tongue surface from a single upscaled mask.

    Splits the outer contour at an anterior anchor (the jaw junction, which
    delineates the tongue front underside) and the right-most point (tongue
    root), and keeps the airway-facing (upper) path along the dorsum. Then it
    CONTINUES down the posterior edge from the root to the tongue's bottom, so a
    curled tongue back is captured (its wall-facing backside), without wrapping
    under the tongue. Returns (M, 2) anterior->posterior in UPSCALED coords, or
    None."""
    if mask_up is None or not mask_up.any():
        return None
    cs, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len).squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return None
    idx_root = int(pts[:, 0].argmax())       # right-most = tongue root
    if jaw_ref_up is not None:
        rx, ry = jaw_ref_up
        idx_j = int(((pts[:, 0] - rx) ** 2 + (pts[:, 1] - ry) ** 2).argmin())
    else:
        idx_j = int(pts[:, 0].argmin())
    a, b = sorted([idx_j, idx_root])
    path_a = pts[a : b + 1]
    path_b = np.concatenate([pts[b:], pts[: a + 1]])
    upper = path_a if path_a[:, 1].mean() <= path_b[:, 1].mean() else path_b
    if upper[0, 0] > upper[-1, 0]:
        upper = upper[::-1]                   # anterior -> posterior (ends at root)
    # Extend down the posterior/backside edge to the tongue bottom.
    backside = _walk_backside(pts, idx_root)
    if len(backside) > 1:
        upper = np.concatenate([upper, backside[1:]], axis=0)
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
    seg = np.sqrt((np.diff(line, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.tile(line[0], (n, 1)).astype(np.float32)
    s = np.linspace(0, total, n)
    x = np.interp(s, cum, line[:, 0])
    y = np.interp(s, cum, line[:, 1])
    return np.stack([x, y], 1).astype(np.float32)


# ── Wall line assembly (original-resolution coords) ──────────────────────────


def _closest_pair(a, b):
    """Indices (i, j) of the closest point between polylines a and b."""
    tree = cKDTree(b)
    d, idx = tree.query(a)
    i = int(d.argmin())
    return i, int(idx[i])


def build_roof(reg_up: dict):
    """One line, front -> back, in original coords:
      palate bottom edge  ->(spliced where they meet)->  velum bottom edge
      ->(bridge from velum bottom-right to closest wall point)->
      pharyngeal-wall edge, DOWN only as far as the tongue reaches.
    Returns (M,2) or None."""
    U = UPSCALE
    palate = reg_up.get(ROOF_FRONT_SUB)
    if palate is None:
        return None
    pal = _bottom_edge(palate)  # ascending x (lips -> hard palate)
    if len(pal) < 2:
        return None
    parts = [pal]
    tail = pal[-1]

    velum = reg_up.get(VELUM_SUB)
    if velum is not None:
        vel = _bottom_edge(velum)  # ascending x
        if len(vel) >= 2:
            # Splice where the two bottom edges MEET (closest pair): keep the
            # palate up to the junction, then the velum onward. This avoids the
            # palate's posterior curl going up above the velum.
            i, j = _closest_pair(pal, vel)
            parts = [pal[: i + 1], vel[j:]]
            tail = vel[-1]  # velum bottom-right

    wall = reg_up.get(PHARYNX_SUB)
    tongue = reg_up.get(TONGUE_SUB)
    if wall is not None:
        wl = _left_edge(wall)  # ascending y (top -> bottom)
        if len(wl) >= 2:
            # Depth limit = the tongue's lowest extent (constriction region only;
            # do NOT run to the bottom of the pharyngeal-wall mask).
            if tongue is not None and tongue.any():
                y_limit = float(np.where(tongue)[0].max())
            else:
                y_limit = float(wl[:, 1].max())
            k = int(((wl - tail[None, :]) ** 2).sum(1).argmin())  # velum junction
            seg = wl[k:]
            seg = seg[seg[:, 1] <= y_limit]
            if len(seg) >= 1:
                br = _bridge(tail, seg[0])
                if len(br):
                    parts.append(br)
                parts.append(seg)

    line = np.concatenate(parts, axis=0) / U
    return _smooth_path(line, SIGMA_PATH)


def build_floor(reg_up: dict, jaw_ref_up):
    """One line, front -> back, in original coords: lower-lip top edge ->
    tongue airway-facing contour down to the back-bottom.

    The lower-lip/jaw edge may legitimately run BELOW the tongue (the mouth
    floor), but those points are not the airway floor — using them would make a
    grid connector cross the tongue surface. So we drop only the lip points that
    sit under the tongue (a tongue point is above them at the same x); anterior
    lip points are kept even where they are low. Returns (M,2) or None."""
    U = UPSCALE
    tongue = reg_up.get(TONGUE_SUB)
    upper = extract_upper_contour(tongue, jaw_ref_up) if tongue is not None else None
    if upper is None or len(upper) < 2:
        return None
    parts = [upper]
    lower = reg_up.get(LOWER_LIP_SUB)
    if lower is not None and lower.any():
        # Use ONLY the lower lip's aperture point (its top-most / inner-upper
        # corner) and connect it straight to the tongue's upper surface. We do
        # NOT trace the lower-lip/jaw top edge back toward the jaw, because that
        # puts floor points under the tongue and makes connectors cross it.
        # Constrictions are formed against the tongue UPPER surface, so the floor
        # is simply: lip aperture -> tongue upper surface.
        ys, xs = np.where(lower)
        y0 = int(ys.min())
        x0 = float(xs[ys == y0].min())          # front-most of the top-most row
        lip_pt = np.array([[x0, float(y0)]], np.float32)
        parts = [lip_pt, upper]
    line = np.concatenate(parts, axis=0) / U
    return _smooth_path(line, SIGMA_PATH)


# ── VTD ──────────────────────────────────────────────────────────────────────


def _edge_center(edge):
    """Arc-length midpoint of an (M,2) polyline (e.g. the velum's lower edge)."""
    if edge is None or len(edge) == 0:
        return None
    if len(edge) == 1:
        return edge[0].astype(np.float32)
    seg = np.sqrt((np.diff(edge, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return edge[len(edge) // 2].astype(np.float32)
    s = total / 2.0
    return np.array(
        [np.interp(s, cum, edge[:, 0]), np.interp(s, cum, edge[:, 1])], np.float32
    )


def _total_lines(n, even_total):
    """Total grid-line count: 2n+3 (odd, default) or 2n+2 (even)."""
    return 2 * n + 2 if even_total else 2 * n + 3


def anchor_indices(n, even_total):
    """Indices of the 3 anchor lines (lips, velum, tongue-back) in the grid."""
    return [0, n + 1, (2 * n + 1) if even_total else (2 * n + 2)]


def _split_index(poly, point):
    """Index on `poly` of the point closest to `point`, kept off the ends."""
    i = int(((poly - np.asarray(point, np.float32)[None, :]) ** 2).sum(1).argmin())
    return min(max(i, 1), len(poly) - 2)


def compute_vtd(roof, floor, velum_center, n, even_total=False):
    """VTD by connecting CORRESPONDING points on the two walls.

    Each wall is split into an oral cavity (lips -> velum) and a pharyngeal
    cavity (velum -> tongue back). The split is the velum-lower-edge center on
    the roof and its CLOSEST counterpart on the floor. Within each cavity both
    walls are arc-length resampled to the same number of points and connected
    index-to-index, so:
      * every line joins a point to its counterpart on the opposite wall,
      * the same number of lines fill each cavity regardless of its length,
      * connectors are monotonic -> they never cross each other and never cut
        across the tongue surface (the failure mode of a straight normal ray).
    VTD is the length of each connector.

    Returns (vtd (L,), roof_pts (L,2), floor_pts (L,2), anchor_idx), with
    L = 2n+3 (odd) or 2n+2 (even). The velum anchor is the shared cavity
    boundary, counted once."""
    L = _total_lines(n, even_total)
    a_idx = anchor_indices(n, even_total)
    nanL = np.full(L, np.nan, np.float32)
    nanL2 = np.full((L, 2), np.nan, np.float32)
    if roof is None or floor is None or len(roof) < 3 or len(floor) < 3:
        return nanL, nanL2.copy(), nanL2.copy(), a_idx

    # Velum split: center on the roof, closest counterpart on the floor.
    i_bu = (
        _split_index(roof, velum_center) if velum_center is not None else len(roof) // 2
    )
    i_bl = _split_index(floor, roof[i_bu])

    # Points per cavity, sharing the velum anchor (counted once):
    #   odd  -> oral n+2, phar n+2  => 2n+3
    #   even -> oral n+2, phar n+1  => 2n+2
    k_o = n + 2
    k_p = (n + 1) if even_total else (n + 2)

    ru_o = _resample(roof[: i_bu + 1], k_o)
    fl_o = _resample(floor[: i_bl + 1], k_o)
    ru_p = _resample(roof[i_bu:], k_p)
    fl_p = _resample(floor[i_bl:], k_p)

    u = np.concatenate([ru_o, ru_p[1:]], axis=0).astype(np.float32)  # drop dup velum
    l = np.concatenate([fl_o, fl_p[1:]], axis=0).astype(np.float32)
    vtd = np.linalg.norm(u - l, axis=1).astype(np.float32)
    return vtd, u, l, a_idx


# ── NPZ / frame helpers ──────────────────────────────────────────────────────


def _load_regions(mask_path: Path):
    data = np.load(mask_path)
    keys = list(data.keys())
    regions = {}
    for sub in REGION_SUBS:
        k = _find_mask_key(keys, sub)
        regions[sub] = data[k].astype(bool) if k is not None else None
    T = next((m.shape[0] for m in regions.values() if m is not None), 0)
    return regions, T


def _velum_lower_center(reg_up):
    """Center of the velum's lower (airway-facing) edge, in original coords."""
    vel = reg_up.get(VELUM_SUB)
    if vel is None or not vel.any():
        return None
    edge = _bottom_edge(vel)
    c = _edge_center(edge)
    return None if c is None else (c / UPSCALE)


def _frame_walls(regions, t, jaw_ref):
    """Smooth-upscale each region at frame t, then trace roof & floor and locate
    the velum lower-edge center. Returns (roof, floor, velum_center, reg_up)."""
    reg_up = {}
    for sub, m in regions.items():
        reg_up[sub] = smooth_mask(m[t]) if (m is not None and t < m.shape[0]) else None
    jaw_up = (jaw_ref[0] * UPSCALE, jaw_ref[1] * UPSCALE) if jaw_ref else None
    return (
        build_roof(reg_up),
        build_floor(reg_up, jaw_up),
        _velum_lower_center(reg_up),
        reg_up,
    )


# ── Diagnostics (over the MRI frame) ─────────────────────────────────────────

# Region colors (match the SAM2 REGION_DEFS palette).
_REGION_HEX = {
    ROOF_FRONT_SUB: "#3cb44b",  # green   (upper lip - palate)
    LOWER_LIP_SUB: "#e6194b",  # red     (lower lip - jaw)
    TONGUE_SUB: "#4363d8",  # blue    (tongue)
    VELUM_SUB: "#911eb4",  # purple  (velum)
    PHARYNX_SUB: "#f58231",  # orange  (pharyngeal wall)
}
# Same colors as BGR for the cv2 video overlay.
_REGION_BGR = {
    ROOF_FRONT_SUB: (75, 180, 60),
    LOWER_LIP_SUB: (75, 25, 230),
    TONGUE_SUB: (216, 99, 67),
    VELUM_SUB: (180, 30, 145),
    PHARYNX_SUB: (49, 130, 245),
}
_ROOF_BGR = (0, 200, 0)  # green line
_FLOOR_BGR = (0, 0, 255)  # red line
_GRID_BGR = (255, 255, 0)  # cyan grid (interior lines)
_ANCHOR_BGR = (255, 0, 255)  # magenta grid (anchor lines: lips, velum, tongue-back)
_VTD_BGR = (0, 255, 255)  # yellow VTD points


def regions_first_hw(regions):
    for m in regions.values():
        if m is not None:
            return m.shape[1], m.shape[2]
    return 104, 104


def _mri_frame(cap, t, mask_hw):
    """Read frame t (or the current sequential frame) from an open VideoCapture,
    resized to mask (H, W) grayscale. Returns (H, W) uint8 or None."""
    mh, mw = mask_hw
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _draw_overlay_bgr(canvas, regions, t, roof, floor, r, f, scale, anchor_idx=()):
    """Draw masks (translucent), wall lines and VTD grid+points onto a BGR
    canvas whose size is (mask * scale). Coordinates are in mask space. Anchor
    grid lines (indices in `anchor_idx`) are drawn magenta and thicker."""
    fh, fw = canvas.shape[:2]
    for sub, m in regions.items():
        if m is None or t >= m.shape[0] or not m[t].any():
            continue
        mr = cv2.resize(
            m[t].astype(np.uint8) * 255, (fw, fh), interpolation=cv2.INTER_NEAREST
        )
        colored = np.zeros_like(canvas)
        colored[mr > 0] = _REGION_BGR.get(sub, (150, 150, 150))
        cv2.addWeighted(colored, 0.35, canvas, 1.0, 0, canvas)

    def _poly(line, color):
        if line is None or len(line) < 2:
            return
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in line], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2)

    anchor_set = set(anchor_idx)
    if r is not None and f is not None:
        for i, (u, l) in enumerate(zip(r, f)):
            if np.isnan(u[0]) or np.isnan(l[0]):
                continue
            p1 = (int(u[0] * scale), int(u[1] * scale))
            p2 = (int(l[0] * scale), int(l[1] * scale))
            is_anchor = i in anchor_set
            cv2.line(
                canvas,
                p1,
                p2,
                _ANCHOR_BGR if is_anchor else _GRID_BGR,
                2 if is_anchor else 1,
            )
            cv2.circle(canvas, p1, 3, _VTD_BGR, -1)
            cv2.circle(canvas, p2, 3, _VTD_BGR, -1)
    _poly(roof, _ROOF_BGR)
    _poly(floor, _FLOOR_BGR)
    return canvas


def save_static_diagnostic(
    out_path, regions, t, video_path, roof, floor, r, f, anchor_idx=()
):
    """Vector PDF: MRI frame (resized to mask space) + translucent masks + wall
    lines + VTD grid/points, all in mask coordinates. Anchor grid lines are
    drawn magenta; interior lines cyan."""
    mh, mw = regions_first_hw(regions)
    mri = None
    if video_path is not None and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        mri = _mri_frame(cap, t, (mh, mw))
        cap.release()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if mri is not None:
        ax.imshow(mri, cmap="gray", interpolation="nearest")
    else:
        ax.set_xlim(0, mw)
        ax.set_ylim(mh, 0)
    # for sub, m in regions.items():
    #     if m is None or t >= m.shape[0] or not m[t].any():
    #         continue
    #     import matplotlib.colors as mcolors

    #     rgb = mcolors.to_rgb(_REGION_HEX.get(sub, "#888888"))
    #     rgba = np.zeros((*m[t].shape, 4), np.float32)
    #     rgba[..., :3] = rgb
    #     rgba[..., 3] = m[t].astype(np.float32) * 0.35
    #     ax.imshow(rgba, interpolation="nearest")
    anchor_set = set(anchor_idx)
    if r is not None and f is not None:
        for i, (u, l) in enumerate(zip(r, f)):
            if np.isnan(u[0]) or np.isnan(l[0]):
                continue
            is_anchor = i in anchor_set
            ax.plot(
                [u[0], l[0]],
                [u[1], l[1]],
                color="magenta" if is_anchor else "cyan",
                lw=1.4 if is_anchor else 0.6,
                zorder=3,
            )
            ax.scatter(
                [u[0], l[0]], [u[1], l[1]], s=8, color="yellow", zorder=5, linewidths=0
            )
    if roof is not None:
        ax.plot(roof[:, 0], roof[:, 1], color="lime", lw=1.8, zorder=4)
    if floor is not None:
        ax.plot(floor[:, 0], floor[:, 1], color="red", lw=1.8, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_diagnostic_video(
    out_path, regions, T, video_path, n_gridlines, jaw_ref, scale=6
):
    """Per-frame MRI overlay video (mask space, upscaled by `scale`)."""
    mh, mw = regions_first_hw(regions)
    fw, fh = mw * scale, mh * scale
    cap = (
        cv2.VideoCapture(str(video_path))
        if (video_path is not None and Path(video_path).exists())
        else None
    )
    fps = (cap.get(cv2.CAP_PROP_FPS) or 50.0) if cap is not None else 50.0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
    )
    for t in range(T):
        mri = _mri_frame(cap, t, (mh, mw)) if cap is not None else None
        if mri is not None:
            canvas = cv2.resize(
                cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR),
                (fw, fh),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            canvas = np.full((fh, fw, 3), 20, np.uint8)
        roof, floor, vel_c, _ = _frame_walls(regions, t, jaw_ref)
        _, r, f, a_idx = compute_vtd(roof, floor, vel_c, n_gridlines, EVEN_TOTAL)
        _draw_overlay_bgr(canvas, regions, t, roof, floor, r, f, scale, a_idx)
        writer.write(canvas)
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

    base = DATA_DIR / spk if spk is not None else DATA_DIR
    label = spk if spk is not None else DATA_DIR.name
    mask_dir = base / "sam_seg" / "masks"
    video_dir = base / VIDEO_DIR
    out_dir = base / "vtd"

    pattern = f"{spk}_*.npz" if spk is not None else "*.npz"
    mask_files = sorted(mask_dir.glob(pattern))
    # mask_files = mask_files[:2]
    if not mask_files:
        print(f"  No mask files in {mask_dir}")
        return
    for sub in ("pts", "norm", "hist", "lines", "diagnostic"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    L = _total_lines(n_gridlines, EVEN_TOTAL)
    a_idx = anchor_indices(n_gridlines, EVEN_TOTAL)
    with open(out_dir / "grid_meta.json", "w") as fh:
        json.dump(
            {
                "n_per_cavity": n_gridlines,
                "even_total": EVEN_TOTAL,
                "total_lines": L,
                "anchor_indices": a_idx,
                "anchor_names": ["lips", "velum", "tongue_back"],
            },
            fh,
            indent=2,
        )

    per_video = {}  # basename -> (mask_path, video_path, T)
    all_vtd = []
    for mp in tqdm(mask_files, desc=f"  {label} VTD"):
        basename = mp.stem
        regions, T = _load_regions(mp)
        if T == 0:
            continue
        jaw_ref = (
            _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB])
            if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None
            else None
        )
        vtd = np.full((T, L), np.nan, np.float32)
        roof_pts = np.full((T, L, 2), np.nan, np.float32)
        floor_pts = np.full((T, L, 2), np.nan, np.float32)
        for t in range(T):
            roof, floor, vel_c, _ = _frame_walls(regions, t, jaw_ref)
            v, r, f, _ = compute_vtd(roof, floor, vel_c, n_gridlines, EVEN_TOTAL)
            vtd[t], roof_pts[t], floor_pts[t] = v, r, f
        np.save(out_dir / "pts" / f"{basename}.npy", vtd)
        np.savez(out_dir / "lines" / f"{basename}.npz", roof=roof_pts, floor=floor_pts)
        vpath = video_dir / f"{basename}.avi"
        if not vpath.exists():
            print(f"  Warning: video not found for {basename}: {vpath}")
        per_video[basename] = (mp, vpath if vpath.exists() else None, T)
        all_vtd.append(vtd)

    if not all_vtd:
        return

    # Per-speaker global min-max per grid line (Shi Eq. 3)
    stacked = np.concatenate(all_vtd, axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    with np.errstate(invalid="ignore"):
        vmin = np.where(
            all_nan, 0.0, np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), 0)
        )
        vmax = np.where(
            all_nan, 1.0, np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), 0)
        )
    rng = np.where((vmax - vmin) > 1e-6, vmax - vmin, 1.0)

    for basename in per_video:
        vtd = np.load(out_dir / "pts" / f"{basename}.npy")
        norm = np.clip((vtd - vmin[None, :]) / rng[None, :], 0.0, 1.0)
        np.save(out_dir / "norm" / f"{basename}.npy", norm.astype(np.float32))
        hist = np.zeros((L, n_bins), np.float32)
        for l in range(L):
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
    jaw_ref = (
        _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB])
        if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None
        else None
    )
    ti = T // 2
    roof, floor, vel_c, _ = _frame_walls(regions, ti, jaw_ref)
    _, r, f, a_idx = compute_vtd(roof, floor, vel_c, n_gridlines, EVEN_TOTAL)
    save_static_diagnostic(
        out_dir / "diagnostic" / f"{label}_frame.pdf",
        regions,
        ti,
        vpath,
        roof,
        floor,
        r,
        f,
        a_idx,
    )

    for basename in tqdm(
        rng_r.sample(names, min(n_videos, len(names))), desc=f"  {label} diag videos"
    ):
        mp, vpath, T = per_video[basename]
        regions, _ = _load_regions(mp)
        jaw_ref = (
            _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB])
            if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None
            else None
        )
        write_diagnostic_video(
            out_dir / "diagnostic" / f"{basename}_vtd.mp4",
            regions,
            T,
            vpath,
            n_gridlines,
            jaw_ref,
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    global UPSCALE, PRE_SIGMA, SIGMA_PATH, EVEN_TOTAL
    single = not SPK_BASE
    p = argparse.ArgumentParser(
        description="Extract vocal-tract distance (VTD) from SAM2 masks."
    )
    if not single:
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
        help="Interior lines per cavity (n). Total = 2n+2 (even) or 2n+3 (odd).",
    )
    p.add_argument("--n-videos", type=int, default=N_DIAGNOSTIC)
    p.add_argument("--bins", type=int, default=N_BINS)
    p.add_argument("--upscale", type=int, default=UPSCALE)
    p.add_argument("--pre-sigma", type=float, default=PRE_SIGMA)
    p.add_argument("--sigma-path", type=float, default=SIGMA_PATH)
    p.add_argument(
        "--parity",
        choices=["even", "odd"],
        default="even" if EVEN_TOTAL else "odd",
        help="even -> 2n+2 grid lines; odd -> 2n+3.",
    )
    args = p.parse_args()
    UPSCALE, PRE_SIGMA, SIGMA_PATH = args.upscale, args.pre_sigma, args.sigma_path
    EVEN_TOTAL = args.parity == "even"

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
