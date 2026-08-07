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
    starts at the LIP APERTURE (the lower-lip point closest to the upper lip, so
    it never sits below the upper-lip edge), then the per-column TOP-most edge of
    (tongue UNION lower-lip) from the lips to the tongue root — the tongue
    dorsum wherever the tongue is present (the higher surface), so it never dips
    under the tongue or onto the jaw and never misses the tongue -> tongue
    posterior/backside edge from the root down to the tongue point closest to the
    lowest point of the pharyngeal wall (its inferior-posterior corner, the
    posterior terminus).

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
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, median_filter
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
    "grid_method": "arc",  # {arc, midline}. arc = index-to-index (legacy); midline = normal cross-sections
    "norm_method": "minmax",  # {minmax, zscore} per-speaker per-grid-line normalization
    "anchor_smooth": {
        "median": 5,
        "sigma": 2.5,
    },  # temporal stabilization of velum/tongue-bottom anchors
    "recenter_iters": 1,  # midline medial-recentering iterations
    "floor_front": "skyline",  # {skyline, contour} tongue/lip surface tracing for the floor
    "velum_anchor": "median",  # {median, smooth} velum split: per-video median fraction (firm) vs per-frame smoothed
    "wall_bottom": "median",  # {median, smooth} pharyngeal-wall bottom (tongue-back terminus ref): firm vs smoothed
    "jump_thresh": {"frac": 0.15, "px": 10.0},  # anchor jump limits: velum fraction / position (px). carry-forward on jump or dropout
    "fixed_window": 0.15,  # grid_method=fixed: arc-fraction window for the per-frame local floor-crossing search
    "session": None,  # session subdir for longitudinal data ({spk}/{session}/...); None = flat
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
SESSION = _cfg.get("session") or None  # e.g. "D1A" for longitudinal; None = flat layout
VIDEO_DIR = _cfg.get("video_dir", "video")
N_GRIDLINES = int(_cfg.get("n_gridlines", 40))
N_BINS = int(_cfg.get("n_bins", 20))
UPSCALE = int(_cfg.get("upscale", 1))
PRE_SIGMA = float(_cfg.get("pre_sigma", 1.5))
SIGMA_PATH = float(_cfg.get("sigma_path", 2.0))
EVEN_TOTAL = bool(_cfg.get("even_total", False))
GRID_METHOD = str(_cfg.get("grid_method", "arc"))
NORM_METHOD = str(_cfg.get("norm_method", "minmax"))
ANCHOR_SMOOTH = dict(_cfg.get("anchor_smooth", {"median": 5, "sigma": 2.5}))
RECENTER_ITERS = int(_cfg.get("recenter_iters", 1))
FLOOR_FRONT = str(_cfg.get("floor_front", "skyline"))
VELUM_ANCHOR = str(_cfg.get("velum_anchor", "median"))
WALL_BOTTOM = str(_cfg.get("wall_bottom", "median"))
JUMP_THRESH = dict(_cfg.get("jump_thresh", {"frac": 0.15, "px": 10.0}))
FIXED_WINDOW = float(_cfg.get("fixed_window", 0.15))
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
    step = 1 if y_next >= y_prev else -1  # direction that goes downward
    path = [pts[idx_root]]
    running_max = float(pts[idx_root, 1])
    i = idx_root
    for _ in range(n // 2):
        j = (i + step) % n
        path.append(pts[j])
        running_max = max(running_max, float(pts[j, 1]))
        if float(pts[j, 1]) < running_max - 3.0:  # clearly past the bottom
            break
        i = j
    path = np.asarray(path, np.float32)
    cut = int(np.argmax(path[:, 1]))  # stop at the lowest (max-y) point
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
    idx_root = int(pts[:, 0].argmax())  # right-most = tongue root
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
        upper = upper[::-1]  # anterior -> posterior (ends at root)
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

    # Anterior start = the LIP APERTURE on the UPPER lip: the upper-lip bottom-edge
    # point closest to the lower lip (the SAME closest pair the floor uses for its
    # start). Without this, pal[0] is the front/left face of the upper lip — the
    # per-column bottom pixels there run DOWN the lip's near-vertical front edge and
    # hang below the true aperture, so the lips anchor connects that stray corner
    # instead of the two closest lip points. Trimming to the aperture drops it.
    lower = reg_up.get(LOWER_LIP_SUB)
    if lower is not None and lower.any():
        lt = _top_edge(lower)  # lower-lip airway-facing edge
        if len(lt):
            d, idx = cKDTree(pal).query(lt)  # nearest upper-edge pt per lower pt
            aperture = pal[int(idx[int(d.argmin())])]  # upper-lip aperture point
            kept = pal[pal[:, 0] >= aperture[0] - 0.5]  # drop the front/left face
            pal = kept if len(kept) >= 2 else pal
            if not np.allclose(pal[0], aperture):
                pal = np.vstack([aperture[None, :], pal])
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


def _wall_bottom_pixel(mask):
    """Lowest pixel (max y) of the largest component of a wall mask, or None.
    Operates in the mask's own coords (used on raw masks for the stabilization
    pre-pass)."""
    if mask is None:
        return None
    core = _largest_component(np.asarray(mask).astype(bool))
    if not core.any():
        return None
    wy, wx = np.where(core)
    i = int(wy.argmax())
    return np.array([float(wx[i]), float(wy[i])], np.float32)


def _wall_bottom_up(reg_up, w_low=None):
    """Pharyngeal-wall bottom in UPSCALED coords for `_tongue_backside`. Uses the
    provided (stabilized, original-coord) `w_low` scaled up when finite; otherwise
    falls back to this frame's lowest wall pixel from `reg_up`."""
    if w_low is not None and np.all(np.isfinite(w_low)):
        return np.asarray(w_low, np.float32) * UPSCALE
    wall = reg_up.get(PHARYNX_SUB)
    if wall is None or not np.asarray(wall).any():
        return None
    wy, wx = np.where(wall)
    i = int(wy.argmax())
    return np.array([float(wx[i]), float(wy[i])], np.float32)


def _tongue_backside(tongue_mask, w_low=None):
    """Posterior/backside edge of the tongue: the contour arc from the right-most
    point (root) to the tongue-contour point CLOSEST to the pharyngeal-wall bottom
    `w_low` (the tongue's inferior-posterior corner). Terminating at that point —
    rather than the tongue's own geometric max-y — is what lets the backside reach
    the true bottom instead of being cut short: the descent past the wall-facing
    corner curls to lower x, so an x-based cut deletes it. Falls back to the
    bottom-most pixel when `w_low` is None. Taken on the higher-x (wall-facing)
    side, oriented root -> terminus. Returns (K,2) upscaled or None."""
    core = _largest_component(tongue_mask.astype(bool)).astype(np.uint8)
    cs, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len).squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return None
    root = int(pts[:, 0].argmax())  # right-most (tongue root)
    if w_low is not None:
        end = int(((pts - np.asarray(w_low, np.float32)[None, :]) ** 2).sum(1).argmin())
    else:
        end = int(pts[:, 1].argmax())  # fallback: bottom-most pixel
    if root == end:
        return None
    a, b = sorted([root, end])
    arc1 = pts[a : b + 1]
    arc2 = np.concatenate([pts[b:], pts[: a + 1]])
    arc = arc1 if arc1[:, 0].mean() >= arc2[:, 0].mean() else arc2  # posterior side
    # Orient root -> terminus (robust to the terminus being above or below root).
    if ((arc[0] - pts[root]) ** 2).sum() > ((arc[-1] - pts[root]) ** 2).sum():
        arc = arc[::-1]
    return arc.astype(np.float32)


def _tongue_dorsum(mask_up, jaw_ref_up=None):
    """Airway-facing tongue surface (dorsum) from the mask CONTOUR: the upper arc
    between the jaw-junction (anterior underside anchor) and the tongue root
    (right-most). Unlike the per-column skyline (`_top_edge`), this follows the
    ACTUAL mask edge, so it stays on the surface where the tongue is steep or bent
    (e.g. toward the back) instead of flattening it to one height per column.
    Returns (M,2) front -> root, or None. (Same upper-arc logic as
    `extract_upper_contour` but without the backside walk — the backside is traced
    separately by `_tongue_backside` to the pharyngeal-wall terminus.)"""
    if mask_up is None or not mask_up.any():
        return None
    core = _largest_component(mask_up.astype(bool)).astype(np.uint8)
    cs, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len).squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return None
    idx_root = int(pts[:, 0].argmax())  # right-most = tongue root
    if jaw_ref_up is not None:
        rx, ry = jaw_ref_up
        idx_j = int(((pts[:, 0] - rx) ** 2 + (pts[:, 1] - ry) ** 2).argmin())
    else:
        idx_j = int(pts[:, 0].argmin())  # left-most = tongue front
    if idx_j == idx_root:
        return None
    a, b = sorted([idx_j, idx_root])
    path_a = pts[a : b + 1]
    path_b = np.concatenate([pts[b:], pts[: a + 1]])
    upper = (
        path_a if path_a[:, 1].mean() <= path_b[:, 1].mean() else path_b
    )  # airway side
    if upper[0, 0] > upper[-1, 0]:
        upper = upper[::-1]  # front (low x) -> root (high x)
    return upper.astype(np.float32)


def _build_floor_contour(reg_up, jaw_ref_up=None, w_low=None):
    """Floor traced from the actual mask CONTOURS, per region (no lip/tongue union
    skyline). lip aperture -> lower-lip top -> bridge -> tongue dorsum (contour) ->
    tongue backside (contour) to the pharyngeal terminus. Each segment follows its
    own region's real edge, so there is no lip/tongue skyline hop and steep/bent
    tongue surfaces are preserved. Returns (M,2) or None (caller falls back to the
    skyline path)."""
    tongue = reg_up.get(TONGUE_SUB)
    if tongue is None or not tongue.any():
        return None
    dorsum = _tongue_dorsum(tongue, jaw_ref_up)
    if dorsum is None:
        dorsum = _top_edge(tongue)  # fall back to skyline for the dorsum only
        if dorsum is None or len(dorsum) < 2:
            return None

    parts = []
    lower = reg_up.get(LOWER_LIP_SUB)
    upper_lip = reg_up.get(ROOF_FRONT_SUB)
    if lower is not None and lower.any():
        lip_top = _top_edge(lower)  # lower lip's OWN airway-facing top (not unioned)
        if len(lip_top):
            aperture = None
            if upper_lip is not None and upper_lip.any():
                ub = _bottom_edge(upper_lip)
                if len(ub):
                    d, _idx = cKDTree(ub).query(lip_top)
                    aperture = lip_top[int(d.argmin())]  # lip aperture (closest pair)
            if aperture is not None:
                lip_top = lip_top[lip_top[:, 0] >= aperture[0] - 0.5]
                lip_top = (
                    np.vstack([aperture[None, :], lip_top])
                    if len(lip_top)
                    else aperture[None, :]
                )
            lip_top = lip_top[
                lip_top[:, 0] <= dorsum[0, 0] + 0.5
            ]  # anterior of tongue front
            if len(lip_top):
                parts.append(lip_top)
                br = _bridge(lip_top[-1], dorsum[0])  # bridge lip -> tongue front
                if len(br):
                    parts.append(br)
    parts.append(dorsum)

    backside = _tongue_backside(
        tongue, _wall_bottom_up(reg_up, w_low)
    )  # root -> terminus
    if backside is not None and len(backside) >= 1:
        parts.append(backside[1:] if len(backside) > 1 else backside)  # drop dup root

    line = np.concatenate(parts, axis=0) / UPSCALE
    return _smooth_path(line, SIGMA_PATH)


def build_floor(reg_up: dict, jaw_ref_up=None, w_low=None):
    """One line, front -> back, in original coords: a single airway-facing upper
    edge from the lip through the tongue, then down the tongue backside.

    Front edge = per-column TOP-most pixel of (tongue UNION lower-lip), from the
    lips to the tongue root: it follows the lip aperture where only the lip is
    present and the tongue dorsum wherever the tongue is present (the tongue is
    the higher surface), so it can never dip under the tongue or onto the jaw
    below it, and never misses the tongue. Back edge = the tongue posterior edge
    from the root down to the bottom. Returns (M,2) or None.

    When `floor_front == "contour"`, the front is instead traced from the actual
    mask contours per region (`_build_floor_contour`), which follows steep/bent
    tongue surfaces the skyline flattens; falls back to this skyline path if the
    contour trace fails."""
    if FLOOR_FRONT == "contour":
        _line = _build_floor_contour(reg_up, jaw_ref_up, w_low)
        if _line is not None:
            return _line
    U = UPSCALE
    tongue = reg_up.get(TONGUE_SUB)
    if tongue is None or not tongue.any():
        return None
    lower = reg_up.get(LOWER_LIP_SUB)
    union = tongue.astype(bool)
    if lower is not None and lower.any():
        union = union | lower.astype(bool)
    front = _top_edge(union.astype(np.uint8))  # lip aperture -> tongue dorsum
    if front is None or len(front) < 2:
        return None
    root_x = float(np.where(tongue)[1].max())
    front = front[front[:, 0] <= root_x + 0.5]  # stop at the tongue root

    # Anterior start = the LIP APERTURE: the lower-lip point closest to the upper
    # lip (closest pair between the two lip masks). Forcing the floor to start
    # here guarantees the first point never sits below the upper-lip edge.
    upper_lip = reg_up.get(ROOF_FRONT_SUB)
    if upper_lip is not None and upper_lip.any() and lower is not None and lower.any():
        ub = _bottom_edge(upper_lip)  # upper-lip airway-facing edge
        lt = _top_edge(lower)  # lower-lip airway-facing edge
        if len(ub) and len(lt):
            d, _idx = cKDTree(ub).query(lt)
            aperture = lt[int(d.argmin())]  # lower-lip aperture point
            front = front[front[:, 0] > aperture[0] + 0.5]
            front = np.vstack([aperture[None, :], front])
    parts = [front]

    # Tongue backside: trace from the root all the way to the tongue-contour point
    # closest to the pharyngeal-wall bottom (its inferior-posterior corner), so it
    # reaches the true bottom instead of being cut short. That terminus is also the
    # posterior VTD anchor, keeping the tongue-back and rear-wall endpoints aligned.
    backside = _tongue_backside(tongue, _wall_bottom_up(reg_up, w_low))
    if backside is not None and len(backside) >= 1:
        parts.append(backside)

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


# ── Anchor + polyline geometry (midline method) ──────────────────────────────


def _velum_centroid(reg_up):
    """Velum anchor from the mask CENTROID (not the noisy airway edge). Centroid
    of the right half (x >= W//2, the posterior/tip side) of the largest velum
    component — same landmark as `compute_velum_kinematics` in
    extract_mask_kinematics.py. Averaging over every pixel makes it far more
    stable than `_velum_lower_center`. Returned in original coords."""
    vel = reg_up.get(VELUM_SUB)
    if vel is None or not np.asarray(vel).any():
        return None
    core = _largest_component(np.asarray(vel).astype(bool))
    ys, xs = np.where(core)
    if len(xs) == 0:
        return None
    mid_x = core.shape[1] // 2
    right = xs >= mid_x
    if not right.any():
        right = np.ones_like(xs, dtype=bool)
    c = np.array([xs[right].mean(), ys[right].mean()], np.float32)
    return c / UPSCALE


def _cumarc(poly):
    """Cumulative arc length along an (M,2) polyline; cum[-1] = total length."""
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at_fraction(poly, frac):
    """Point at arc-length fraction `frac` in [0,1] along an (M,2) polyline."""
    poly = np.asarray(poly, np.float32)
    cum = _cumarc(poly)
    total = cum[-1]
    if total <= 0:
        return poly[0].astype(np.float32)
    s = float(np.clip(frac, 0.0, 1.0)) * total
    return np.array(
        [np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], np.float32
    )


def _project_to_polyline(poly, point):
    """Nearest vertex on `poly` to `point`: returns (index, arc-length fraction,
    vertex)."""
    poly = np.asarray(poly, np.float32)
    i = int(((poly - np.asarray(point, np.float32)[None, :]) ** 2).sum(1).argmin())
    cum = _cumarc(poly)
    frac = float(cum[i] / cum[-1]) if cum[-1] > 0 else 0.5
    return i, frac, poly[i]


def _nearest_on(poly, point):
    """Nearest polyline vertex to `point` (fallback when a normal misses a wall)."""
    poly = np.asarray(poly, np.float32)
    i = int(((poly - np.asarray(point, np.float32)[None, :]) ** 2).sum(1).argmin())
    return poly[i].astype(np.float32)


def stabilize(arr, median_size=5, sigma=2.5):
    """Temporally de-jitter an anchor trajectory: interpolate NaN (dropout)
    frames, median-filter out isolated bad frames, then Gaussian low-pass the
    residual jitter. Accepts (T,) or (T,2); a no-op smoothing (median<=1, sigma<=0)
    still fills NaNs. Slow real motion survives; frame-rate segmentation noise dies."""
    arr = np.asarray(arr, np.float64)
    two_d = arr.ndim == 2
    cols = arr if two_d else arr[:, None]
    T = cols.shape[0]
    idx = np.arange(T)
    out = np.empty_like(cols)
    for c in range(cols.shape[1]):
        x = cols[:, c]
        ok = np.isfinite(x)
        if ok.sum() == 0:
            out[:, c] = x
            continue
        x = np.interp(idx, idx[ok], x[ok])
        if median_size and median_size > 1:
            x = median_filter(x, size=int(median_size))
        if sigma and sigma > 0:
            x = gaussian_filter1d(x, sigma=float(sigma))
        out[:, c] = x
    return out if two_d else out[:, 0]


def _hold_jumps(x, thresh, max_hold=3):
    """Suppress transient anchor glitches: if a frame jumps more than `thresh` from
    the last accepted value (or the mask is missing), carry the last value forward —
    but only for up to `max_hold` consecutive frames, after which the new level is
    accepted as genuine motion (not a glitch). This is the velum-disappears /
    fragmentation fallback: use the previous position instead of the jumped one.
    Accepts (T,) or (T,2)."""
    x = np.asarray(x, np.float64).copy()
    two = x.ndim == 2
    last = None
    held = 0
    for t in range(len(x)):
        v = x[t]
        ok = bool(np.all(np.isfinite(v))) if two else bool(np.isfinite(v))
        if last is None:
            if ok:
                last = v.copy() if two else v
            continue
        if not ok:  # dropout -> hold previous
            if held < max_hold:
                x[t] = last
                held += 1
            continue
        d = float(np.linalg.norm(v - last)) if two else abs(float(v - last))
        if d > thresh and held < max_hold:  # jump -> hold previous
            x[t] = last
            held += 1
        else:
            last = v.copy() if two else v
            held = 0
    return x


def _line_crossings(origin, ndir, poly):
    """All signed intersections of the infinite line (origin + t*ndir) with an
    (M,2) polyline, sorted by |t| (t is signed distance since |ndir|=1). Returns
    (ts, pts) or (None, None). Vectorized over segments."""
    poly = np.asarray(poly, np.float64)
    o = np.asarray(origin, np.float64)
    d = np.asarray(ndir, np.float64)
    a = poly[:-1]
    e = poly[1:] - poly[:-1]  # (S,2)
    det = d[1] * e[:, 0] - d[0] * e[:, 1]
    rhs = a - o[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-rhs[:, 0] * e[:, 1] + e[:, 0] * rhs[:, 1]) / det
        u = (d[0] * rhs[:, 1] - d[1] * rhs[:, 0]) / det
    valid = (np.abs(det) > 1e-9) & (u >= -1e-6) & (u <= 1 + 1e-6)
    if not valid.any():
        return None, None
    ts = t[valid]
    pts = (a[valid] + u[valid, None] * e[valid]).astype(np.float32)
    order = np.argsort(np.abs(ts))
    return ts[order], pts[order]


def _straddle_hits(origin, nrm, R, F):
    """Roof/floor hits of the normal line that STRADDLE the midline point: the
    nearest roof crossing, then the nearest floor crossing on the OPPOSITE side.
    This yields one clean straight cross-section per station and avoids the
    degenerate concave-corner case where both nearest hits land on the same side
    (which would report a spuriously tiny width). Falls back to nearest wall point
    when the normal misses."""
    tr, pr = _line_crossings(origin, nrm, R)
    tf, pf = _line_crossings(origin, nrm, F)
    rp = pr[0] if pr is not None else _nearest_on(R, origin)
    t_roof = tr[0] if tr is not None else 0.0
    if pf is None:
        fp = _nearest_on(F, origin)
    elif pr is not None:
        opp = np.where(np.sign(tf) != np.sign(t_roof))[0]
        j = opp[np.abs(tf[opp]).argmin()] if len(opp) else 0
        fp = pf[j]
    else:
        fp = pf[0]
    return rp.astype(np.float32), fp.astype(np.float32)


def _frac_of(poly, cum, p):
    """Arc-length fraction along `poly` of the vertex nearest point `p`."""
    i = int(((poly - np.asarray(p, np.float32)[None, :]) ** 2).sum(1).argmin())
    return float(cum[i] / cum[-1]) if cum[-1] > 0 else 0.5


def _local_crossing(origin, nrm, poly, cum, s_ref, window):
    """Crossing of the fixed gridline (origin, nrm) with `poly` that is nearest the
    origin AND whose arc-length fraction is within `window` of the reference fraction
    `s_ref`. Restricting to a local arc window stops a fixed gridline from grabbing a
    far crossing on the other side of the tongue when the articulator has moved (the
    'connector jumps to the tongue front' bug). Falls back to the reference location
    on the polyline when the line misses locally."""
    ts, pts = _line_crossings(origin, nrm, poly)
    if pts is None:
        return _point_at_fraction(poly, s_ref) if s_ref is not None else _nearest_on(poly, origin)
    if s_ref is None:
        return pts[0]
    fr = np.array([_frac_of(poly, cum, p) for p in pts])
    m = np.abs(fr - s_ref) <= window
    if m.any():
        idx = np.where(m)[0]
        return pts[idx[np.abs(ts[idx]).argmin()]]
    return _point_at_fraction(poly, s_ref)  # gridline missed locally -> reference spot


def build_midline(roof, floor, recenter_iters=1, m=150, sigma=3.0):
    """Smooth centerline between the two walls, lips -> posterior terminus. A
    coarse arc-length average is medial-recentered (nearest point on each wall,
    averaged) and smoothed; the final VTD measurement is normal to this line, so
    the midline only has to be approximately central."""
    ru = _resample(np.asarray(roof, np.float32), m)
    fl = _resample(np.asarray(floor, np.float32), m)
    if ru is None or fl is None:
        return None
    mid = _smooth_path(((ru + fl) / 2.0).astype(np.float32), sigma)
    for _ in range(max(0, int(recenter_iters))):
        new = np.empty_like(mid)
        for i in range(len(mid)):
            new[i] = (_nearest_on(roof, mid[i]) + _nearest_on(floor, mid[i])) / 2.0
        mid = _smooth_path(new, sigma)
    return mid


def _cavity_gridlines(roof_seg, floor_seg, k, recenter_iters=1):
    """Fixed gridline geometry for one cavity, per the original VTD construction:
    k origins spaced EVENLY along the cavity midline, each with the perpendicular
    direction (normal to the local midline). Returns (origins (k,2), normals (k,2))
    or None. The gridlines curve/fan with the midline through the bend (the semi-
    polar shape). VTD is later the airway width along each gridline."""
    R = np.asarray(roof_seg, np.float32)
    F = np.asarray(floor_seg, np.float32)
    if len(R) < 2 or len(F) < 2:
        return None
    mid = build_midline(R, F, recenter_iters)
    if mid is None or len(mid) < 2:
        return None
    G = _resample(mid, k)
    if G is None or len(G) != k:
        return None
    N = np.zeros((k, 2), np.float32)
    for i in range(k):
        tang = G[min(k - 1, i + 1)] - G[max(0, i - 1)]
        nrm = np.array([-tang[1], tang[0]], np.float32)
        ln = float(np.linalg.norm(nrm))
        N[i] = nrm / ln if ln > 1e-6 else np.array([0.0, 1.0], np.float32)
    return G.astype(np.float32), N


def _cavity_grid(roof_seg, floor_seg, k, recenter_iters=1):
    """One cavity's grid (per-frame midline mode): k gridlines evenly along the
    cavity midline, VTD = width between BOTH walls along each perpendicular gridline
    (a straddle hit on each wall). Even quantity is the gridlines along the midline
    (as in the original VTD), not either wall. Returns (roof_pts, floor_pts) or None."""
    gl = _cavity_gridlines(roof_seg, floor_seg, k, recenter_iters)
    if gl is None:
        return None
    O, N = gl
    R = np.asarray(roof_seg, np.float32)
    F = np.asarray(floor_seg, np.float32)
    rp = np.full((k, 2), np.nan, np.float32)
    fp = np.full((k, 2), np.nan, np.float32)
    for i in range(k):
        rp[i], fp[i] = _straddle_hits(O[i], N[i], R, F)
    return rp, fp


def midline_grid(
    roof, floor, f_vel, tongue_bottom, n, even_total=False, recenter_iters=1
):
    """Unified VTD grid for BOTH cavities: L points along a shared midline, each
    VTD measured PERPENDICULAR to the local tract axis (straight, shortest cross-
    section — no angled connectors). The velum split (fraction `f_vel` on the roof)
    only pins the middle index; the rear wall is clipped at its nearest point to
    the (smoothed) tongue-bottom, and the last line is that tongue-bottom straight
    across to the closest wall point. Same (L, anchor_idx) contract as compute_vtd."""
    L = _total_lines(n, even_total)
    a_idx = anchor_indices(n, even_total)
    nanL = np.full(L, np.nan, np.float32)
    nanL2 = np.full((L, 2), np.nan, np.float32)
    if roof is None or floor is None or len(roof) < 3 or len(floor) < 3:
        return nanL, nanL2.copy(), nanL2.copy(), a_idx
    R = np.asarray(roof, np.float32)
    F = np.asarray(floor, np.float32)

    # Posterior clip: drop rear-wall points beyond the nearest point to the tongue
    # bottom, so the grid never extends below where the tongue reaches (req 3).
    tb = None
    if tongue_bottom is not None and np.all(np.isfinite(tongue_bottom)):
        tb = np.asarray(tongue_bottom, np.float32)
        wi = int(((R - tb[None, :]) ** 2).sum(1).argmin())
        if wi >= 2:
            R = R[: wi + 1]
    if len(R) < 3:
        return nanL, nanL2.copy(), nanL2.copy(), a_idx

    # Velum split: the fraction pins the oral/pharyngeal boundary on each wall.
    if f_vel is None or not np.isfinite(f_vel):
        f_vel = 0.5
    p_vel = _point_at_fraction(R, float(f_vel))
    i_ru = _split_index(R, p_vel)  # boundary index on the roof
    i_rf = _split_index(F, R[i_ru])  # matching boundary on the floor

    k_o = n + 2
    k_p = (n + 1) if even_total else (n + 2)
    # BOTH cavities use the identical method: even midline stations, straight
    # cross-section perpendicular to the local tract axis. Building each cavity's
    # midline from its own wall segments lets the pharyngeal midline reach the
    # terminus, so the last station lands near tb without a jump.
    og = _cavity_grid(R[: i_ru + 1], F[: i_rf + 1], k_o, recenter_iters)
    pg = _cavity_grid(R[i_ru:], F[i_rf:], k_p, recenter_iters)
    if og is None or pg is None:
        return nanL, nanL2.copy(), nanL2.copy(), a_idx
    roof_pts = np.vstack([og[0], pg[0][1:]]).astype(np.float32)  # share velum point
    floor_pts = np.vstack([og[1], pg[1][1:]]).astype(np.float32)

    # Pin the endpoints: lips, and the tongue-bottom terminus straight across to the
    # nearest rear-wall point (a small correction now that the midline reaches tb).
    roof_pts[0], floor_pts[0] = R[0], F[0]
    if tb is not None:
        floor_pts[-1] = tb
        roof_pts[-1] = _nearest_on(R, tb)

    vtd = np.linalg.norm(roof_pts - floor_pts, axis=1).astype(np.float32)
    return vtd, roof_pts, floor_pts, a_idx


def build_fixed_grid(walls, f_vel, tb, n, even_total=False, recenter_iters=1, m=200):
    """Build ONE fixed grid for the whole clip (original-VTD style). The reference
    roof and floor are the per-point MEDIAN of the traced walls over all frames;
    split at the median velum fraction into oral/pharyngeal cavities; lay fixed
    perpendicular gridlines evenly along each cavity's reference midline. Per frame
    only the two boundary crossings move (`measure_fixed_grid`) — the gridlines
    never move, which is what removes the per-frame geometry jitter. Returns a dict
    {O, N, a_idx} or None."""
    # Roof reference = median of the dedicated roof-airway boundary (so Rref and the
    # gridline geometry match what's measured). Floor reference = median traced floor.
    roofs = [
        _resample(w[3]["roof"][0], m)
        for w in walls
        if w[3].get("roof") and w[3]["roof"][0] is not None and len(w[3]["roof"][0]) >= 2
    ]
    floors = [_resample(w[1], m) for w in walls if w[1] is not None and len(w[1]) >= 2]
    roofs = [r for r in roofs if r is not None]
    floors = [f for f in floors if f is not None]
    if not roofs or not floors:
        return None
    with np.errstate(invalid="ignore"):
        ref_roof = np.nanmedian(np.stack(roofs, 0), axis=0).astype(np.float32)
        ref_floor = np.nanmedian(np.stack(floors, 0), axis=0).astype(np.float32)
    tb = np.asarray(tb, np.float32)
    ref_tb = (
        np.array([np.nanmedian(tb[:, 0]), np.nanmedian(tb[:, 1])], np.float32)
        if np.isfinite(tb).any()
        else ref_floor[-1]
    )
    R = ref_roof
    if np.all(np.isfinite(ref_tb)):
        wi = int(((R - ref_tb[None, :]) ** 2).sum(1).argmin())
        if wi >= 2:
            R = R[: wi + 1]  # clip rear wall at the reference terminus
    F = ref_floor
    if len(R) < 3 or len(F) < 3:
        return None
    fv = float(np.nanmedian(f_vel)) if np.isfinite(f_vel).any() else 0.5
    if not np.isfinite(fv):
        fv = 0.5
    p_vel = _point_at_fraction(R, fv)
    i_ru = _split_index(R, p_vel)
    i_rf = _split_index(F, R[i_ru])
    k_o = n + 2
    k_p = (n + 1) if even_total else (n + 2)
    go = _cavity_gridlines(R[: i_ru + 1], F[: i_rf + 1], k_o, recenter_iters)
    gp = _cavity_gridlines(R[i_ru:], F[i_rf:], k_p, recenter_iters)
    if go is None or gp is None:
        return None
    O = np.vstack([go[0], gp[0][1:]]).astype(np.float32)  # share the velum gridline
    N = np.vstack([go[1], gp[1][1:]]).astype(np.float32)
    # Record where each gridline crosses the REFERENCE (median) roof and floor, as
    # POINTS. Per frame, the raw-contour crossing nearest each of these is taken —
    # so both wall anchors' positions are set once here from the median, and neither
    # a curving tongue back nor a shortening upper-lip contour can make them snap.
    Rref = np.zeros((len(O), 2), np.float32)
    Fref = np.zeros((len(O), 2), np.float32)
    for i in range(len(O)):
        rp, fp = _straddle_hits(O[i], N[i], ref_roof, ref_floor)
        Rref[i], Fref[i] = rp, fp
    return {
        "O": O,
        "N": N,
        "a_idx": anchor_indices(n, even_total),
        "Rref": Rref,
        "Fref": Fref,
    }


def measure_fixed_grid(contours, grid):
    """Measure VTD for one frame against a FIXED grid. BOTH walls are measured the
    same way: for each fixed gridline, the raw-mask-contour crossing nearest that
    gridline's reference point (Rref for the roof, Fref for the floor). No traced
    polyline, no terminus, no arc — so neither a curving tongue back nor a shortening
    upper-lip contour can make a connector snap. VTD = distance between the two
    crossings. Returns (vtd (L,), roof_pts, floor_pts, a_idx)."""
    O, N, a_idx = grid["O"], grid["N"], grid["a_idx"]
    Rref, Fref = grid.get("Rref"), grid.get("Fref")
    L = len(O)
    roof_pts = np.full((L, 2), np.nan, np.float32)
    floor_pts = np.full((L, 2), np.nan, np.float32)
    rc = contours.get("roof") if contours else None
    fc = contours.get("floor") if contours else None
    for i in range(L):
        r_ref = Rref[i] if Rref is not None else O[i]
        f_ref = Fref[i] if Fref is not None else O[i]
        roof_pts[i] = _contour_hit(O[i], N[i], rc, r_ref) if rc else r_ref
        floor_pts[i] = _contour_hit(O[i], N[i], fc, f_ref) if fc else f_ref
    vtd = np.linalg.norm(roof_pts - floor_pts, axis=1).astype(np.float32)
    return vtd, roof_pts, floor_pts, a_idx


def _region_contours(reg_up, subs):
    """Closed mask-boundary polylines (ALL connected components) for the union of the
    given regions, in original coords. Raw boundaries — no landmark/arc/terminus
    definition, so nothing snaps or shortens when the tissue moves. Keeping all
    components means a detached lip (or a disconnected wall) is never dropped."""
    union = None
    for s in subs:
        m = reg_up.get(s)
        if m is not None and np.asarray(m).any():
            mb = np.asarray(m).astype(bool)
            union = mb if union is None else (union | mb)
    if union is None or not union.any():
        return None
    cs, _ = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        pts = c.squeeze()
        if pts.ndim == 2 and len(pts) >= 3:
            out.append((np.vstack([pts, pts[0]]).astype(np.float32)) / UPSCALE)  # closed
    return out or None


def _roof_airway(reg_up):
    """Airway-facing roof boundary, built to be SMOOTH and STRAIGHT across the velum
    port (never curling around the velum flap):
      palate/upper-lip bottom edge (front, raw — NO aperture trim, so the lip is
      stable) -> velum bottom edge traced to its ACTUAL bottom-most point -> a
      single straight bridge from that bottom point to the rear wall -> wall airway
      edge downward. Returns one open polyline in original coords, or None."""
    palate = reg_up.get(ROOF_FRONT_SUB)
    if palate is None or not np.asarray(palate).any():
        return None
    pal = _bottom_edge(np.asarray(palate).astype(np.uint8))  # airway-facing, ascending x
    if pal is None or len(pal) < 2:
        return None
    parts = [pal]
    tail = pal[-1]
    velum = reg_up.get(VELUM_SUB)
    if velum is not None and np.asarray(velum).any():
        vel = _bottom_edge(np.asarray(velum).astype(np.uint8))
        if len(vel) >= 2:
            i, j = _closest_pair(pal, vel)  # splice palate <-> velum where they meet
            bot = int(vel[:, 1].argmax())  # ACTUAL bottom-most (max-y) velum flesh point
            velseg = vel[j : bot + 1] if bot >= j else vel[bot : j + 1][::-1]
            if len(velseg) >= 1:
                parts = [pal[: i + 1], velseg]
                tail = vel[bot]  # bridge starts from the true bottom point (not the curl)
    wall = reg_up.get(PHARYNX_SUB)
    if wall is not None and np.asarray(wall).any():
        wl = _left_edge(np.asarray(wall).astype(np.uint8))  # airway edge, ascending y
        if len(wl) >= 2:
            k = int(((wl - tail[None, :]) ** 2).sum(1).argmin())  # nearest wall point
            br = _bridge(tail, wl[k])  # ONE straight span, velum bottom -> wall
            if len(br):
                parts.append(br)
            parts.append(wl[k:])  # wall airway edge downward
    line = np.concatenate(parts, axis=0) / UPSCALE
    return _smooth_path(line, SIGMA_PATH)


def _contour_hit(origin, nrm, contours, ref_pt):
    """Where the fixed gridline crosses the raw mask contours, choosing the crossing
    nearest the reference point `ref_pt` (which sits on the airway-facing side).
    Robust to a moving/curving/shortening boundary: no landmark, just the mask edge
    along this fixed line, disambiguated by proximity to the reference. ALWAYS on
    tissue — if the line misses every component, falls back to the nearest contour
    point, never to a spot floating in the airway."""
    ref = np.asarray(ref_pt, np.float32)
    best, best_d = None, np.inf
    near_pt, near_d = None, np.inf
    for C in contours:
        ts, pts = _line_crossings(origin, nrm, C)
        if pts is not None:
            d = np.sqrt(((pts - ref[None, :]) ** 2).sum(1))
            j = int(d.argmin())
            if d[j] < best_d:
                best_d, best = d[j], pts[j]
        dd = ((C - ref[None, :]) ** 2).sum(1)
        jj = int(dd.argmin())
        if dd[jj] < near_d:
            near_d, near_pt = dd[jj], C[jj]
    return best if best is not None else near_pt


def _utterance_anchors(regions, T, jaw_ref):
    """Trace walls for every frame of one utterance and derive the velum split
    fraction and tongue-bottom anchor, honoring VELUM_ANCHOR / ANCHOR_SMOOTH.

    Returns (walls, f_vel, tb): walls[t] = (roof, floor, vel_c); f_vel (T,) is the
    velum split as a roof-arc fraction; tb (T,2) the tongue-bottom point.

    VELUM_ANCHOR == "median" gives ONE firm split fraction for the whole clip — the
    median of the per-frame roof-arc fractions. Because the fraction is measured
    relative to the palate/roof, it is invariant to gross head TRANSLATION (the
    velum and roof shift together), so a fixed median needs no separate head-motion
    correction. "smooth" keeps the per-frame trajectory, temporally de-jittered.

    The pharyngeal-wall bottom (tongue-back terminus reference) is likewise
    stabilized per WALL_BOTTOM ("median" = one firm point per clip) so the terminus
    stops chasing wall-mask fraying."""
    msize = int(ANCHOR_SMOOTH.get("median", 1))
    sig = float(ANCHOR_SMOOTH.get("sigma", 0.0))

    # Pre-pass: stabilized pharyngeal-wall bottom, from the raw wall masks.
    w_raw = np.full((T, 2), np.nan, np.float32)
    wall_masks = regions.get(PHARYNX_SUB)
    if wall_masks is not None:
        for t in range(min(T, wall_masks.shape[0])):
            wl = _wall_bottom_pixel(wall_masks[t])
            if wl is not None:
                w_raw[t] = wl
    w_raw = _hold_jumps(w_raw, JUMP_THRESH.get("px", 10.0))
    if WALL_BOTTOM == "median" and np.isfinite(w_raw).any():
        wm = np.array(
            [np.nanmedian(w_raw[:, 0]), np.nanmedian(w_raw[:, 1])], np.float32
        )
        w_low_s = np.tile(wm, (T, 1))
    else:
        w_low_s = stabilize(w_raw, msize, sig)  # (T,2), NaN-filled

    walls = []
    f_raw = np.full(T, np.nan, np.float32)
    tb_raw = np.full((T, 2), np.nan, np.float32)
    for t in range(T):
        roof, floor, vel_c, reg_up = _frame_walls(regions, t, jaw_ref, w_low=w_low_s[t])
        ra = _roof_airway(reg_up)
        cont = {
            # floor: raw tongue(+lip) contour — single blob, airway edge is clean and
            # captures the curling back with no terminus landmark.
            "floor": _region_contours(reg_up, [TONGUE_SUB, LOWER_LIP_SUB]),
            # roof: dedicated airway boundary — raw palate/lip front (stable lip) +
            # velum bottom edge to its bottom-most point + straight bridge to the wall.
            "roof": [ra] if (ra is not None and len(ra) >= 2) else None,
        }
        walls.append((roof, floor, vel_c, cont))
        vcent = _velum_centroid(reg_up)
        if roof is not None and vcent is not None and len(roof) >= 3:
            _, f_raw[t], _ = _project_to_polyline(roof, vcent)
        if floor is not None and len(floor) >= 2:
            tb_raw[t] = floor[-1]
    # Combat velum-mask dropouts/fragmentation and terminus glitches: carry the
    # previous position through transient jumps before firming.
    f_raw = _hold_jumps(f_raw, JUMP_THRESH.get("frac", 0.15))
    tb_raw = _hold_jumps(tb_raw, JUMP_THRESH.get("px", 10.0))
    if VELUM_ANCHOR == "median":
        fm = float(np.nanmedian(f_raw)) if np.isfinite(f_raw).any() else np.nan
        f_vel = np.full(T, fm, np.float64)  # NaN -> midline_grid falls back to 0.5
    else:  # "smooth"
        f_vel = stabilize(f_raw, msize, sig)
    tb = stabilize(tb_raw, msize, sig)
    return walls, f_vel, tb


def _grid_with_anchors(roof, floor, vel_c, f_vel_t, tb_t, n, fixed_grid=None, contours=None):
    """Single-frame VTD for the configured grid method using precomputed anchors.
    When `fixed_grid` is provided (grid_method='fixed'), measure both walls against
    the raw mask contours."""
    if fixed_grid is not None:
        return measure_fixed_grid(contours, fixed_grid)
    if GRID_METHOD == "midline":
        return midline_grid(roof, floor, f_vel_t, tb_t, n, EVEN_TOTAL, RECENTER_ITERS)
    if (
        roof is not None
        and len(roof) >= 3
        and f_vel_t is not None
        and np.isfinite(f_vel_t)
    ):
        vc = _point_at_fraction(roof, float(f_vel_t))
    else:
        vc = vel_c
    return compute_vtd(roof, floor, vc, n, EVEN_TOTAL)


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


def _frame_walls(regions, t, jaw_ref, w_low=None):
    """Smooth-upscale each region at frame t, then trace roof & floor and locate
    the velum lower-edge center. `w_low` (stabilized pharyngeal-wall bottom, original
    coords) is threaded to the floor's tongue-back terminus. Returns
    (roof, floor, velum_center, reg_up)."""
    reg_up = {}
    for sub, m in regions.items():
        reg_up[sub] = smooth_mask(m[t]) if (m is not None and t < m.shape[0]) else None
    jaw_up = (jaw_ref[0] * UPSCALE, jaw_ref[1] * UPSCALE) if jaw_ref else None
    return (
        build_roof(reg_up),
        build_floor(reg_up, jaw_up, w_low=w_low),
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


def _trim_roof_to_last(roof, r):
    """Trim the drawn roof/rear-wall polyline at the last VTD connection point
    (r[-1]) so it does not extend below where the last tongue point connects."""
    if roof is None or r is None or len(r) == 0 or len(roof) < 2:
        return roof
    last = np.asarray(r[-1], np.float32)
    if not np.all(np.isfinite(last)):
        return roof
    idx = int(((np.asarray(roof, np.float32) - last[None, :]) ** 2).sum(1).argmin())
    return roof[: idx + 1] if idx >= 1 else roof


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
    _poly(_trim_roof_to_last(roof, r), _ROOF_BGR)
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
        roof_draw = _trim_roof_to_last(roof, r)
        ax.plot(roof_draw[:, 0], roof_draw[:, 1], color="lime", lw=1.8, zorder=4)
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
    # mpeg4 (mp4v) requires a timebase denominator <= 65535; a fractional source
    # rate like 81.967 fps yields 1000/81967 and fails to open the writer. Round to
    # an integer (and guard against absurd/zero values) for the diagnostic video.
    fps = float(fps)
    fps = round(fps) if np.isfinite(fps) and 1.0 <= fps <= 240.0 else 50

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
    )
    # Same anchors/grid as the output (e.g. the fixed per-clip grid).
    walls, f_vel, tb = _utterance_anchors(regions, T, jaw_ref)
    fixed_grid = (
        build_fixed_grid(walls, f_vel, tb, n_gridlines, EVEN_TOTAL, RECENTER_ITERS)
        if GRID_METHOD == "fixed"
        else None
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
        roof, floor, vel_c, cont = walls[t]
        _, r, f, a_idx = _grid_with_anchors(
            roof, floor, vel_c, f_vel[t], tb[t], n_gridlines, fixed_grid, cont
        )
        # Draw the actual boundary the points are measured against: in fixed mode the
        # roof is the airway-boundary (bridged velum), not the build_roof curl.
        roof_draw = cont["roof"][0] if (fixed_grid is not None and cont.get("roof")) else roof
        _draw_overlay_bgr(canvas, regions, t, roof_draw, floor, r, f, scale, a_idx)
        writer.write(canvas)
    if cap is not None:
        cap.release()
    writer.release()


# ── Per-speaker processing ───────────────────────────────────────────────────


def _discover_speakers():
    if not SPK_BASE:
        return []
    if SESSION is not None:
        return sorted(
            d.name
            for d in DATA_DIR.iterdir()
            if d.is_dir()
            and d.name.startswith(SPK_BASE)
            # and d.name in ["ID16", "ID17", "ID18", "ID20", "ID21"]
            and (d / SESSION / "sam_seg" / "masks").is_dir()
        )
    else:
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
    session = SESSION if SESSION is not None else ""
    mask_dir = base / session / "sam_seg" / "masks"
    video_dir = base / session / VIDEO_DIR
    out_dir = base / session / f"vtd_{GRID_METHOD}_{NORM_METHOD}"

    pattern = "*.npz"
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
                "grid_method": GRID_METHOD,
                "norm_method": NORM_METHOD,
                "anchor_smooth": ANCHOR_SMOOTH,
                "recenter_iters": RECENTER_ITERS,
                "floor_front": FLOOR_FRONT,
                "velum_anchor": VELUM_ANCHOR,
                "wall_bottom": WALL_BOTTOM,
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
        # Pass 1: trace walls + derive the velum split fraction and tongue-bottom,
        # per VELUM_ANCHOR (median = one firm fraction per clip) / ANCHOR_SMOOTH.
        walls, f_vel, tb = _utterance_anchors(regions, T, jaw_ref)
        # grid_method='fixed': build one fixed grid for the whole clip from the
        # median walls; per frame only the boundary crossings move.
        fixed_grid = (
            build_fixed_grid(walls, f_vel, tb, n_gridlines, EVEN_TOTAL, RECENTER_ITERS)
            if GRID_METHOD == "fixed"
            else None
        )
        # Pass 2: compute VTD with the stabilized anchors (or the fixed grid).
        vtd = np.full((T, L), np.nan, np.float32)
        roof_pts = np.full((T, L, 2), np.nan, np.float32)
        floor_pts = np.full((T, L, 2), np.nan, np.float32)
        for t in range(T):
            roof, floor, vel_c, cont = walls[t]
            v, r, f, _ = _grid_with_anchors(
                roof, floor, vel_c, f_vel[t], tb[t], n_gridlines, fixed_grid, cont
            )
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

    # Per-speaker per-grid-line normalization (norm_method).
    stacked = np.concatenate(all_vtd, axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    if NORM_METHOD == "zscore":
        with np.errstate(invalid="ignore"):
            mean = np.where(all_nan, 0.0, np.nanmean(stacked, 0))
            std = np.where(all_nan, 1.0, np.nanstd(stacked, 0))
        std = np.where(std > 1e-6, std, 1.0)
        hist_range = (-3.0, 3.0)
    else:  # minmax (legacy behavior; Shi Eq. 3)
        with np.errstate(invalid="ignore"):
            vmin = np.where(
                all_nan, 0.0, np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), 0)
            )
            vmax = np.where(
                all_nan,
                1.0,
                np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), 0),
            )
        rng = np.where((vmax - vmin) > 1e-6, vmax - vmin, 1.0)
        hist_range = (0.0, 1.0)

    for basename in per_video:
        vtd = np.load(out_dir / "pts" / f"{basename}.npy")
        if NORM_METHOD == "zscore":
            norm = (vtd - mean[None, :]) / std[None, :]
        else:
            norm = np.clip((vtd - vmin[None, :]) / rng[None, :], 0.0, 1.0)
        np.save(out_dir / "norm" / f"{basename}.npy", norm.astype(np.float32))
        hist = np.zeros((L, n_bins), np.float32)
        for l in range(L):
            col = norm[:, l][np.isfinite(norm[:, l])]
            if col.size:
                hist[l], _ = np.histogram(col, bins=n_bins, range=hist_range)
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
    walls_d, f_vel_d, tb_d = _utterance_anchors(regions, T, jaw_ref)
    fixed_grid_d = (
        build_fixed_grid(walls_d, f_vel_d, tb_d, n_gridlines, EVEN_TOTAL, RECENTER_ITERS)
        if GRID_METHOD == "fixed"
        else None
    )
    roof, floor, vel_c, cont = walls_d[ti]
    _, r, f, a_idx = _grid_with_anchors(
        roof, floor, vel_c, f_vel_d[ti], tb_d[ti], n_gridlines, fixed_grid_d, cont
    )
    roof_draw = cont["roof"][0] if (fixed_grid_d is not None and cont.get("roof")) else roof
    save_static_diagnostic(
        out_dir / "diagnostic" / f"{label}_frame.pdf",
        regions,
        ti,
        vpath,
        roof_draw,
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
    global GRID_METHOD, NORM_METHOD, ANCHOR_SMOOTH, RECENTER_ITERS, SESSION, FLOOR_FRONT
    global VELUM_ANCHOR, WALL_BOTTOM, FIXED_WINDOW
    single = not SPK_BASE
    p = argparse.ArgumentParser(
        description="Extract vocal-tract distance (VTD) from SAM2 masks."
    )
    if not single:
        p.add_argument(
            "--spk",
            nargs="+",
            type=str,
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
    p.add_argument(
        "--grid-method",
        choices=["arc", "midline", "fixed"],
        default=GRID_METHOD,
        help="arc = legacy index-to-index; midline = per-frame perpendicular; "
        "fixed = one grid per video from median walls (original-VTD style).",
    )
    p.add_argument(
        "--norm-method",
        choices=["minmax", "zscore"],
        default=NORM_METHOD,
        help="per-speaker per-grid-line normalization.",
    )
    p.add_argument(
        "--anchor-median",
        type=int,
        default=int(ANCHOR_SMOOTH.get("median", 5)),
        help="temporal median-filter window for anchors (<=1 disables).",
    )
    p.add_argument(
        "--anchor-sigma",
        type=float,
        default=float(ANCHOR_SMOOTH.get("sigma", 2.5)),
        help="temporal Gaussian sigma for anchors (0 disables).",
    )
    p.add_argument(
        "--recenter-iters",
        type=int,
        default=RECENTER_ITERS,
        help="midline medial-recentering iterations.",
    )
    p.add_argument(
        "--session",
        default=SESSION,
        help="session subdir for longitudinal data ({spk}/{session}/...).",
    )
    p.add_argument(
        "--floor-front",
        choices=["skyline", "contour"],
        default=FLOOR_FRONT,
        help="skyline = per-column union top edge (legacy); contour = per-region mask-edge trace.",
    )
    p.add_argument(
        "--velum-anchor",
        choices=["median", "smooth"],
        default=VELUM_ANCHOR,
        help="median = one firm per-video split fraction; smooth = per-frame smoothed.",
    )
    p.add_argument(
        "--wall-bottom",
        choices=["median", "smooth"],
        default=WALL_BOTTOM,
        help="pharyngeal-wall bottom (tongue-back terminus ref): median (firm) vs smoothed.",
    )
    p.add_argument(
        "--fixed-window",
        type=float,
        default=FIXED_WINDOW,
        help="grid_method=fixed: arc-fraction window for the local floor-crossing search.",
    )
    args = p.parse_args()
    UPSCALE, PRE_SIGMA, SIGMA_PATH = args.upscale, args.pre_sigma, args.sigma_path
    EVEN_TOTAL = args.parity == "even"
    GRID_METHOD, NORM_METHOD = args.grid_method, args.norm_method
    ANCHOR_SMOOTH = {"median": args.anchor_median, "sigma": args.anchor_sigma}
    RECENTER_ITERS = args.recenter_iters
    SESSION = args.session or None
    FLOOR_FRONT = args.floor_front
    VELUM_ANCHOR = args.velum_anchor
    WALL_BOTTOM = args.wall_bottom
    FIXED_WINDOW = args.fixed_window

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
