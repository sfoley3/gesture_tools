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

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
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
    mask2d: np.ndarray, upscale: int = UPSCALE, pre_sigma: float = PRE_SIGMA
) -> np.ndarray:
    """Anti-alias a binary mask: keep the largest component, upsample (cubic),
    Gaussian-blur, threshold. Returns an (H*upscale, W*upscale) uint8 mask.
    Tracing on this removes staircase pixelation so VTD isn't inflated."""
    if mask2d is None or not mask2d.any():
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
        junction.append(pts[int((dx**2 + dy**2).min(1).argmin())].astype(np.float32))
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
    cs, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len).squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return None
    idx_root = int(pts[:, 0].argmax())
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


def build_roof(reg_up: dict):
    """One line: lips/palate bottom edge -> velum bottom edge -> pharyngeal
    wall (from the velum-junction down). Returns (M,2) front->back in original
    coords, or None."""
    U = UPSCALE
    palate = reg_up.get(ROOF_FRONT_SUB)
    if palate is None:
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
            tail = vel[-1]  # velum bottom-right

    wall = reg_up.get(PHARYNX_SUB)
    if wall is not None:
        wl = _left_edge(wall)
        if len(wl) >= 2:
            # closest wall point to the velum bottom-right (or palate tail)
            j = int(((wl - tail[None, :]) ** 2).sum(1).argmin())
            seg = _trim_wall_bottom(wl[j:])  # from junction DOWN only
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


def _frame_walls(regions, t, jaw_ref):
    """Smooth-upscale each region at frame t, then trace roof & floor."""
    reg_up = {}
    for sub, m in regions.items():
        reg_up[sub] = smooth_mask(m[t]) if (m is not None and t < m.shape[0]) else None
    jaw_up = (jaw_ref[0] * UPSCALE, jaw_ref[1] * UPSCALE) if jaw_ref else None
    return build_roof(reg_up), build_floor(reg_up, jaw_up), reg_up


# ── Diagnostics (over the MRI frame) ─────────────────────────────────────────

# BGR for cv2 video overlay
_REGION_BGR = {
    ROOF_FRONT_SUB: (75, 180, 60),  # green   (upper lip - palate)
    LOWER_LIP_SUB: (75, 25, 230),  # red     (lower lip - jaw)
    TONGUE_SUB: (216, 99, 67),  # blue    (tongue)
    VELUM_SUB: (180, 30, 145),  # purple  (velum)
    PHARYNX_SUB: (49, 130, 245),  # orange  (pharyngeal wall)
}
_ROOF_BGR = (0, 200, 0)  # green line
_FLOOR_BGR = (0, 0, 255)  # red line
_GRID_BGR = (255, 255, 0)  # cyan grid
_VTD_BGR = (0, 255, 255)  # yellow VTD points


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
        mr = cv2.resize(
            m[t].astype(np.uint8) * 255, (fw, fh), interpolation=cv2.INTER_NEAREST
        )
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
    fh, fw = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(fw / 100, fh / 100))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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
        fps, fw, fh, n_frames = 50.0, mw * 6, mh * 6, T

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
    )
    for t in range(min(T, n_frames)):
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = np.full((fh, fw, 3), 20, np.uint8)
        roof, floor, _ = _frame_walls(regions, t, jaw_ref)
        vtd, r, f = compute_vtd(roof, floor, n_gridlines)
        _overlay(frame, regions, t, roof, floor, r, f)
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

    base = DATA_DIR / spk if spk is not None else DATA_DIR
    label = spk if spk is not None else DATA_DIR.name
    mask_dir = base / "sam_seg" / "masks"
    video_dir = base / VIDEO_DIR
    out_dir = base / "vtd"

    pattern = f"{spk}_*.npz" if spk is not None else "*.npz"
    mask_files = sorted(mask_dir.glob(pattern))
    mask_files = mask_files[:10]
    if not mask_files:
        print(f"  No mask files in {mask_dir}")
        return
    for sub in ("pts", "norm", "hist", "lines", "diagnostic"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    per_video = {}  # basename -> (mask_path, video_path, T)
    all_vtd = []
    for mp in tqdm(mask_files, desc=f"  {label} VTD"):
        basename = mp.stem[len(spk) + 1 :] if spk is not None else mp.stem
        regions, T = _load_regions(mp)
        if T == 0:
            continue
        jaw_ref = (
            _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB])
            if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None
            else None
        )
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
    jaw_ref = (
        _find_jaw_anchor(regions[TONGUE_SUB], regions[LOWER_LIP_SUB])
        if regions[TONGUE_SUB] is not None and regions[LOWER_LIP_SUB] is not None
        else None
    )
    ti = T // 2
    roof, floor, _ = _frame_walls(regions, ti, jaw_ref)
    _, r, f = compute_vtd(roof, floor, n_gridlines)
    save_static_diagnostic(
        out_dir / "diagnostic" / f"{label}_frame.pdf",
        regions,
        ti,
        vpath,
        roof,
        floor,
        r,
        f,
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
    global UPSCALE, PRE_SIGMA, SIGMA_PATH
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
    p.add_argument("--n-gridlines", type=int, default=N_GRIDLINES)
    p.add_argument("--n-videos", type=int, default=N_DIAGNOSTIC)
    p.add_argument("--bins", type=int, default=N_BINS)
    p.add_argument("--upscale", type=int, default=UPSCALE)
    p.add_argument("--pre-sigma", type=float, default=PRE_SIGMA)
    p.add_argument("--sigma-path", type=float, default=SIGMA_PATH)
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
