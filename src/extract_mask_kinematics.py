#!/usr/bin/env python3
# ABOUTME: Extracts articulatory kinematics (velum, tongue tip, lip aperture, tongue body, tongue root, larynx) from SAM2 mask NPZ files.
# ABOUTME: Outputs per-video NPY timeseries (T, 7) and diagnostic MP4 overlays with color-coded tracked points.
"""
Processes SAM2 segmentation masks to extract raw articulatory kinematic timeseries.

For each speaker × video:
  - Velum: tip position (x, y) in 104×104 pixel space — right-half centroid of largest component
  - Tongue tip: distance to alveolar ridge reference point
  - Lip aperture: distance between closest points on lower lip and upper lip masks
  - Tongue body: distance from closest tongue pixel to leftmost velum pixel (per-frame)
  - Tongue root: distance from rightmost tongue pixel y-coordinate to max y (104)
  - Larynx: y-coordinate of mask centroid

Output NPY shape: (T, 7) — [velum_x, velum_y, tt_dist, lip_aperture, tongue_body_dist, tongue_root_dist, larynx_y]

Reads config.json from the project root for data_dir, n_diagnostic, spk_base, video_dir.
If spk_base is empty, single-speaker mode (no speaker subdirectories).
Otherwise, speakers are {spk_base}{number} and can be specified via --spk with just numbers.

Usage:
    conda run -n myenv python extract_mask_kinematics.py [--spk 2 3 ...]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label, binary_erosion
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── Load config ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

DATA_DIR          = Path(_cfg["data_dir"])
N_DIAGNOSTIC      = int(_cfg.get("n_diagnostic", 10))
SPK_BASE          = _cfg.get("spk_base", "")
VIDEO_DIR         = _cfg.get("video_dir", "video")
MAX_Y             = 104  # frame height for tongue root distance
VELUM_PROCESSING  = _cfg.get("velum_processing", "")  # "pca" → 1D PC1 projection; empty → x,y

# BGR colors — named for readability
PURPLE    = (128, 0, 128)
BLUE      = (200, 100, 0)
GREEN     = (50, 180, 0)
DARK_RED  = (180, 50, 50)
RED       = (0, 0, 255)
YELLOW    = (0, 255, 255)
CYAN      = (255, 255, 0)
MAGENTA   = (255, 0, 255)
ORANGE    = (0, 165, 255)
WHITE     = (255, 255, 255)


MASK_ALPHA = 0.80
DOT_RADIUS = 2


# ── Kinematics helpers ──────────────────────────────────────────────────────

def _find_mask_key(keys, substring: str) -> str:
    """Find the first key in *keys* that contains *substring* (case-insensitive)."""
    sub = substring.lower()
    for k in keys:
        if sub in k.lower():
            return k
    raise KeyError(f"No mask key containing '{substring}' in {list(keys)}")


def _has_mask_key(keys, substring: str) -> bool:
    """Return True if any key in *keys* contains *substring* (case-insensitive)."""
    sub = substring.lower()
    return any(sub in k.lower() for k in keys)


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill so no NaNs remain."""
    out = arr.copy()
    # Forward pass (LOCF)
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    # Backward pass for any leading NaNs
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask containing only the largest connected component."""
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == sizes.argmax()

def _erode_to_thick_part(mask: np.ndarray, min_pixels: int = 20) -> np.ndarray:
    """
    Iteratively erode a 2D binary mask until it shrinks below min_pixels,
    then return the last erosion that still had enough pixels.
    This strips thin regions while preserving structurally thick ones.
    """
    struct = np.ones((3, 3), dtype=bool)  # 3x3 square structuring element
    current = mask.copy()
    last_valid = mask.copy()

    while True:
        eroded = binary_erosion(current, structure=struct)
        if eroded.sum() < min_pixels:
            break
        last_valid = eroded
        current = eroded

    return last_valid


# def compute_velum_kinematics(velum_masks: np.ndarray):
#     """
#     velum_masks: (T, H, W) bool
#     Tracks the center of the thick inferior portion of the velum by iteratively
#     eroding the mask until thin regions are stripped away, then taking the centroid.
#     Falls back to the pre-erosion centroid if erosion eliminates the mask entirely.
#     Returns raw x, y arrays of shape (T,), float32.
#     """
#     T, H, W = velum_masks.shape
#     x = np.full(T, np.nan, dtype=np.float32)
#     y = np.full(T, np.nan, dtype=np.float32)

#     for t in range(T):
#         mask = velum_masks[t]
#         if not mask.any():
#             continue

#         core = _largest_component(mask)
#         thick = _erode_to_thick_part(core)

#         ys, xs = np.where(thick)
#         x[t] = xs.mean()
#         y[t] = ys.mean()

#     return _fill_nans(x), _fill_nans(y)


def _velum_pca_project(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit PCA on the (x, y) velum centroids and project onto PC1.
    Returns a 1D array of shape (T,), float32.
    """
    xy = np.stack([x, y], axis=1)  # (T, 2)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(xy).ravel().astype(np.float32)  # (T,)
    return proj


def compute_velum_kinematics(velum_masks: np.ndarray):
    """
    velum_masks: (T, H, W) bool
    Tracks the velum tip as the centroid of the right half (x >= mask_width/2) of the
    largest connected component. The right half is anatomically where the velum tip sits.
    Returns raw x, y arrays of shape (T,), float32.
    Coordinates are in mask pixel space (104×104).
    """
    T, H, W = velum_masks.shape
    x = np.full(T, np.nan, dtype=np.float32)
    y = np.full(T, np.nan, dtype=np.float32)
    mid_x = H // 2

    for t in range(T):
        mask = velum_masks[t]
        if mask.any():
            core = _largest_component(mask)
            ys, xs = np.where(core)
            right = xs >= mid_x
            if right.any():
                x[t] = xs[right].mean()
                y[t] = ys[right].mean()

    return _fill_nans(x), _fill_nans(y)

def _find_alveolar_ridge(palate_masks: np.ndarray) -> tuple:
    """
    Derive a fixed alveolar ridge reference point from the time-averaged palate mask.
    Front of mouth is on the left side of the frame (minimum x).
    Returns (x_ar, y_ar) in mask pixel space.
    """
    # Average mask across frames to find the stable palate region
    avg = palate_masks.mean(axis=0)
    active = avg > 0.1  # pixels active in >10% of frames
    if not active.any():
        active = palate_masks.any(axis=0)
    ys, xs = np.where(active)
    # Leftmost column where palate is present
    x_ar = int(xs.min())
    # Vertical centroid at that x column
    col_mask = active[:, x_ar]
    y_ar = float(np.where(col_mask)[0].mean())
    return float(x_ar), y_ar


def compute_tt_kinematics(tongue_masks: np.ndarray, palate_masks: np.ndarray):
    """
    tongue_masks, palate_masks: (T, H, W) bool
    Tracks the tongue pixel closest to a fixed alveolar ridge reference point.
    Returns raw dist array of shape (T,), float32.
    Also returns tt_points array (T, 2) and the alveolar ridge point (x_ar, y_ar).
    """
    T = tongue_masks.shape[0]
    dist = np.full(T, np.nan, dtype=np.float32)
    tt_points = np.full((T, 2), np.nan, dtype=np.float32)

    x_ar, y_ar = _find_alveolar_ridge(palate_masks)

    for t in range(T):
        tmask = tongue_masks[t]
        if not tmask.any():
            continue
        ys, xs = np.where(tmask)
        # Euclidean distance from each tongue pixel to the fixed alveolar ridge point
        dists = np.sqrt((xs - x_ar) ** 2 + (ys - y_ar) ** 2)
        idx = dists.argmin()
        tt_points[t, 0] = float(xs[idx])
        tt_points[t, 1] = float(ys[idx])
        dist[t] = dists[idx]

    dist = _fill_nans(dist)
    # Fill tt_points to match (forward then backward)
    for i in range(1, T):
        if np.isnan(tt_points[i, 0]):
            tt_points[i] = tt_points[i - 1]
    for i in range(T - 2, -1, -1):
        if np.isnan(tt_points[i, 0]):
            tt_points[i] = tt_points[i + 1]

    return dist, tt_points, (x_ar, y_ar)


def _fill_nans_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill a (T, 2) array on rows where column 0 is NaN."""
    out = arr.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i, 0]):
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i, 0]):
            out[i] = out[i + 1]
    return out


def compute_lip_aperture(lower_lip_masks: np.ndarray, palate_masks: np.ndarray):
    """
    lower_lip_masks, palate_masks: (T, H, W) bool
    Measures lip aperture as the Euclidean distance between the closest points
    on the lower-lip and upper-lip masks each frame.
    Returns (dist, upper_points, lower_points) — dist is (T,), points are (T, 2).
    """
    T = lower_lip_masks.shape[0]
    dist = np.full(T, np.nan, dtype=np.float32)
    upper_pts = np.full((T, 2), np.nan, dtype=np.float32)
    lower_pts = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        ul = palate_masks[t]
        ll = lower_lip_masks[t]
        if not ul.any() or not ll.any():
            continue

        uy, ux = np.where(ul)
        ly, lx = np.where(ll)

        # Brute-force closest pair via broadcasting
        dx = ux[:, None] - lx[None, :]   # (Nu, Nl)
        dy = uy[:, None] - ly[None, :]
        d2 = dx ** 2 + dy ** 2
        idx_flat = d2.argmin()
        i_u, i_l = np.unravel_index(idx_flat, d2.shape)

        dist[t] = float(np.sqrt(d2[i_u, i_l]))
        upper_pts[t] = [float(ux[i_u]), float(uy[i_u])]
        lower_pts[t] = [float(lx[i_l]), float(ly[i_l])]

    dist = _fill_nans(dist)
    upper_pts = _fill_nans_2d(upper_pts)
    lower_pts = _fill_nans_2d(lower_pts)
    return dist, upper_pts, lower_pts


def compute_tongue_body(tongue_masks: np.ndarray, velum_masks: np.ndarray):
    """
    tongue_masks, velum_masks: (T, H, W) bool
    Measures tongue body constriction as the distance from the closest tongue pixel
    to the leftmost point on the velum mask (computed per-frame).
    Returns (dist, tongue_points, velum_ref_points) — dist is (T,), points are (T, 2).
    """
    T = tongue_masks.shape[0]
    dist = np.full(T, np.nan, dtype=np.float32)
    tongue_pts = np.full((T, 2), np.nan, dtype=np.float32)
    velum_pts = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        tmask = tongue_masks[t]
        vmask = velum_masks[t]
        if not tmask.any() or not vmask.any():
            continue

        # Leftmost point on velum (min x; vertical centroid at that column for tie-break)
        vy, vx = np.where(vmask)
        x_min = int(vx.min())
        col_ys = vy[vx == x_min]
        ref_x = float(x_min)
        ref_y = float(col_ys.mean())

        # Closest tongue pixel to the velum reference
        ty, tx = np.where(tmask)
        dists = np.sqrt((tx - ref_x) ** 2 + (ty - ref_y) ** 2)
        idx = dists.argmin()

        dist[t] = dists[idx]
        tongue_pts[t] = [float(tx[idx]), float(ty[idx])]
        velum_pts[t] = [ref_x, ref_y]

    dist = _fill_nans(dist)
    tongue_pts = _fill_nans_2d(tongue_pts)
    velum_pts = _fill_nans_2d(velum_pts)
    return dist, tongue_pts, velum_pts


def compute_tongue_root(tongue_masks: np.ndarray, max_y: int = MAX_Y):
    """
    tongue_masks: (T, H, W) bool
    Tracks the rightmost point on the tongue mask each frame.
    Returns the vertical distance from that point's y-coordinate to max_y (frame height).
    Returns (dist_arr, root_points) — dist_arr is (T,), root_points is (T, 2).
    """
    T = tongue_masks.shape[0]
    dist_arr = np.full(T, np.nan, dtype=np.float32)
    root_pts = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        tmask = tongue_masks[t]
        if not tmask.any():
            continue

        ty, tx = np.where(tmask)
        x_max = int(tx.max())
        col_ys = ty[tx == x_max]
        root_x = float(x_max)
        root_y = float(col_ys.mean())

        dist_arr[t] = float(max_y) - root_y
        root_pts[t] = [root_x, root_y]

    dist_arr = _fill_nans(dist_arr)
    root_pts = _fill_nans_2d(root_pts)
    return dist_arr, root_pts


def compute_larynx_kinematics(larynx_masks: np.ndarray):
    """
    larynx_masks: (T, H, W) bool
    Tracks the y-coordinate of the mask centroid at each frame.
    Returns (y_arr, center_points) — y_arr is (T,), center_points is (T, 2).
    """
    T = larynx_masks.shape[0]
    y_arr = np.full(T, np.nan, dtype=np.float32)
    center_pts = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        m = larynx_masks[t]
        if not m.any():
            continue
        ys, xs = np.where(m)
        center_pts[t] = [float(xs.mean()), float(ys.mean())]
        y_arr[t] = float(ys.mean())

    y_arr = _fill_nans(y_arr)
    center_pts = _fill_nans_2d(center_pts)
    return y_arr, center_pts


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_kinematics(mask_path: Path, video_path: Path) -> tuple:
    """
    Returns (kinematics_array, fps, tracked_points, mask_data).

    If VELUM_PROCESSING == "pca":
        kinematics shape: (T, 6) — [velum_pc1, tt_dist, lip_aperture,
                                     tongue_body_dist, tongue_root_dist, larynx_y]
    Otherwise:
        kinematics shape: (T, 7) — [velum_x, velum_y, tt_dist, lip_aperture,
                                     tongue_body_dist, tongue_root_dist, larynx_y]

    Only computes kinematics for masks that are present in the NPZ.
    Missing columns are filled with NaN.
    tracked_points: dict of per-feature point arrays for diagnostic drawing.
    """
    data = np.load(mask_path)
    keys = list(data.keys())

    # Determine which masks are available
    has_velum = _has_mask_key(keys, "velum")
    has_tongue = _has_mask_key(keys, "tongue")
    has_upper_lip = _has_mask_key(keys, "upper lip")
    has_lower_lip = _has_mask_key(keys, "lower lip")
    has_larynx = _has_mask_key(keys, "larynx")

    # Load available masks
    velum_masks = data[_find_mask_key(keys, "velum")] if has_velum else None
    tongue_masks = data[_find_mask_key(keys, "tongue")] if has_tongue else None
    palate_masks = data[_find_mask_key(keys, "upper lip")] if has_upper_lip else None
    lower_lip_masks = data[_find_mask_key(keys, "lower lip")] if has_lower_lip else None
    larynx_masks = data[_find_mask_key(keys, "larynx")] if has_larynx else None

    # Determine T from whichever mask is present
    for m in [velum_masks, tongue_masks, palate_masks, lower_lip_masks, larynx_masks]:
        if m is not None:
            T = m.shape[0]
            break
    else:
        raise ValueError(f"No recognized masks in {mask_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"    Warning: could not read FPS from {video_path.name}, defaulting to 30")
    cap.release()

    # Initialize columns as NaN
    vx = np.full(T, np.nan, dtype=np.float32)
    vy = np.full(T, np.nan, dtype=np.float32)
    tt_dist = np.full(T, np.nan, dtype=np.float32)
    la_dist = np.full(T, np.nan, dtype=np.float32)
    tb_dist = np.full(T, np.nan, dtype=np.float32)
    tr_dist = np.full(T, np.nan, dtype=np.float32)
    larynx_y = np.full(T, np.nan, dtype=np.float32)

    tracked_points = {}

    # Velum
    velum_pc1 = np.full(T, np.nan, dtype=np.float32)
    if has_velum:
        vx, vy = compute_velum_kinematics(velum_masks)
        tracked_points["velum_centroids"] = np.stack([vx, vy], axis=1)
        if VELUM_PROCESSING == "pca":
            velum_pc1 = _velum_pca_project(vx, vy)

    # Tongue tip (needs tongue + upper lip / palate)
    tt_points = None
    if has_tongue and has_upper_lip:
        tt_dist, tt_points, alveolar_ridge = compute_tt_kinematics(tongue_masks, palate_masks)
        tracked_points["tt_points"] = tt_points
        tracked_points["alveolar_ridge"] = alveolar_ridge

    # Lip aperture (needs lower lip + upper lip)
    if has_lower_lip and has_upper_lip:
        la_dist, la_upper_pts, la_lower_pts = compute_lip_aperture(lower_lip_masks, palate_masks)
        tracked_points["la_upper_pts"] = la_upper_pts
        tracked_points["la_lower_pts"] = la_lower_pts

    # Tongue body (needs tongue + velum)
    if has_tongue and has_velum:
        tb_dist, tb_tongue_pts, tb_velum_pts = compute_tongue_body(tongue_masks, velum_masks)
        tracked_points["tb_tongue_pts"] = tb_tongue_pts
        tracked_points["tb_velum_pts"] = tb_velum_pts

    # Tongue root (needs tongue)
    if has_tongue:
        tr_dist, tr_root_pts = compute_tongue_root(tongue_masks)
        tracked_points["tr_root_pts"] = tr_root_pts

    # Larynx
    if has_larynx:
        larynx_y, larynx_pts = compute_larynx_kinematics(larynx_masks)
        tracked_points["larynx_pts"] = larynx_pts

    if VELUM_PROCESSING == "pca":
        kinematics = np.stack([velum_pc1, tt_dist, la_dist, tb_dist, tr_dist, larynx_y], axis=1).astype(np.float32)
    else:
        kinematics = np.stack([vx, vy, tt_dist, la_dist, tb_dist, tr_dist, larynx_y], axis=1).astype(np.float32)

    return kinematics, fps, tracked_points, data


# ── Diagnostic video ────────────────────────────────────────────────────────

def _scale_point(x: float, y: float, mask_h: int, mask_w: int, frame_h: int, frame_w: int):
    sx = int(round(x * frame_w / mask_w))
    sy = int(round(y * frame_h / mask_h))
    return sx, sy


def write_diagnostic_video(
    mask_path: Path,
    video_path: Path,
    out_path: Path,
    tracked_points: dict,
    mask_data: dict,
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    # Get mask dimensions from whichever mask is available
    all_keys = list(mask_data.keys())
    mask_h = mask_w = None
    for k in all_keys:
        arr = mask_data[k]
        if hasattr(arr, 'shape') and arr.ndim == 3:
            mask_h, mask_w = arr.shape[1], arr.shape[2]
            break
    if mask_h is None:
        cap.release()
        writer.release()
        return

    # Build region color list only for present masks
    region_colors = []
    if _has_mask_key(all_keys, "velum"):
        region_colors.append((_find_mask_key(all_keys, "velum"), PURPLE))
    if _has_mask_key(all_keys, "tongue"):
        region_colors.append((_find_mask_key(all_keys, "tongue"), BLUE))
    if _has_mask_key(all_keys, "upper lip"):
        region_colors.append((_find_mask_key(all_keys, "upper lip"), GREEN))
    if _has_mask_key(all_keys, "lower lip"):
        region_colors.append((_find_mask_key(all_keys, "lower lip"), ORANGE))
    if _has_mask_key(all_keys, "larynx"):
        region_colors.append((_find_mask_key(all_keys, "larynx"), CYAN))

    def _draw_dot(frame, pts, t, color):
        if pts is None:
            return
        if t < len(pts):
            x, y = pts[t]
            if not (np.isnan(x) or np.isnan(y)):
                px, py = _scale_point(x, y, mask_h, mask_w, frame_h, frame_w)
                cv2.circle(frame, (px, py), DOT_RADIUS, color, -1)

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay masks
        overlay = frame.copy()
        for region, color in region_colors:
            if t >= mask_data[region].shape[0]:
                continue
            m = mask_data[region][t]
            if not m.any():
                continue
            m_resized = cv2.resize(
                m.astype(np.uint8) * 255, (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST
            )
            colored = np.zeros_like(frame)
            colored[m_resized > 0] = color
            cv2.addWeighted(colored, MASK_ALPHA, overlay, 1.0, 0, overlay)
            frame = overlay.copy()

        # Velum centroid
        _draw_dot(frame, tracked_points.get("velum_centroids"), t, RED)

        # Tongue tip closest point
        _draw_dot(frame, tracked_points.get("tt_points"), t, RED)

        # Lip aperture: upper and lower closest points + connecting line
        la_upper_pts = tracked_points.get("la_upper_pts")
        la_lower_pts = tracked_points.get("la_lower_pts")
        if la_upper_pts is not None and la_lower_pts is not None:
            if t < len(la_upper_pts) and t < len(la_lower_pts):
                ux, uy = la_upper_pts[t]
                lx, ly = la_lower_pts[t]
                if not (np.isnan(ux) or np.isnan(uy) or np.isnan(lx) or np.isnan(ly)):
                    p1 = _scale_point(ux, uy, mask_h, mask_w, frame_h, frame_w)
                    p2 = _scale_point(lx, ly, mask_h, mask_w, frame_h, frame_w)
                    cv2.circle(frame, p1, DOT_RADIUS, RED, -1)
                    cv2.circle(frame, p2, DOT_RADIUS, RED, -1)
                    cv2.line(frame, p1, p2, RED, 1)

        # Tongue body: closest tongue pixel to velum reference
        _draw_dot(frame, tracked_points.get("tb_tongue_pts"), t, RED)

        # Tongue root: rightmost tongue point
        _draw_dot(frame, tracked_points.get("tr_root_pts"), t, RED)

        # Larynx: center point
        _draw_dot(frame, tracked_points.get("larynx_pts"), t, RED)

        writer.write(frame)

    cap.release()
    writer.release()


# ── Per-speaker processing ──────────────────────────────────────────────────

def _discover_speakers() -> list:
    """List speaker directories matching SPK_BASE* under DATA_DIR."""
    if not SPK_BASE:
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith(SPK_BASE)
    )


def process_speaker(spk: str | None):
    """
    Process a single speaker. If spk is None (single-speaker mode),
    paths are directly under DATA_DIR with no speaker subdirectory.
    """
    if spk is not None:
        base = DATA_DIR / spk
        label = spk
    else:
        base = DATA_DIR
        label = DATA_DIR.name

    mask_dir = base / "sam_seg" / "masks"
    video_dir = base / VIDEO_DIR
    kin_dir = base / "sam_seg" / "kinematics"
    diag_dir = base / "sam_seg" / "diagnostic"

    # In multi-speaker mode, masks are prefixed with speaker name
    if spk is not None:
        mask_files = sorted(mask_dir.glob(f"{spk}_*.npz"))
    else:
        mask_files = sorted(mask_dir.glob("*.npz"))

    if not mask_files:
        print(f"  No mask files found in {mask_dir}")
        return

    kin_dir.mkdir(parents=True, exist_ok=True)

    # Select random files for diagnostic videos (reproducible)
    seed = sum(ord(c) for c in label)
    rng = random.Random(seed)
    diag_files = rng.sample(mask_files, min(N_DIAGNOSTIC, len(mask_files)))
    diag_set = set(f.name for f in diag_files)

    results = {}  # basename → (kinematics, fps, tracked_points, mask_data)

    for mask_path in tqdm(mask_files, desc=f"  {label} kinematics"):
        if spk is not None:
            # e.g. spk1_spk1_1.npz  →  basename = spk1_1
            basename = mask_path.stem[len(spk) + 1:]
        else:
            basename = mask_path.stem
        video_path = video_dir / f"{basename}.avi"

        if not video_path.exists():
            print(f"    Missing video: {video_path}")
            continue

        try:
            kin, fps, tracked_pts, mask_data = extract_kinematics(mask_path, video_path)
        except Exception as e:
            print(f"    ERROR extracting {mask_path.name}: {e}")
            continue

        out_npy = kin_dir / f"{basename}.npy"
        np.save(out_npy, kin)

        if mask_path.name in diag_set:
            results[basename] = (kin, fps, tracked_pts, mask_data, video_path)

    # Write diagnostic videos
    if results:
        diag_dir.mkdir(parents=True, exist_ok=True)
        for basename, (kin, fps, tracked_pts, mask_data, video_path) in tqdm(
            results.items(), desc=f"  {label} diagnostic videos"
        ):
            out_mp4 = diag_dir / f"{basename}_diagnostic.mp4"
            if spk is not None:
                mask_path = mask_dir / f"{spk}_{basename}.npz"
            else:
                mask_path = mask_dir / f"{basename}.npz"
            try:
                write_diagnostic_video(mask_path, video_path, out_mp4, tracked_pts, mask_data)
            except Exception as e:
                print(f"    ERROR writing diagnostic video {basename}: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    single_speaker = not SPK_BASE

    parser = argparse.ArgumentParser(description="Extract mask kinematics from SAM2 NPZ files.")
    if not single_speaker:
        parser.add_argument(
            "--spk", nargs="+", type=int, default=None, metavar="N",
            help=f"Speaker numbers to process, e.g. --spk 2 3 (prefix: '{SPK_BASE}'). Default: all."
        )
    args = parser.parse_args()

    if single_speaker:
        print(f"\n[{DATA_DIR.name}] (single speaker)")
        process_speaker(None)
    else:
        all_speakers = _discover_speakers()
        if not all_speakers:
            print(f"No speaker directories matching '{SPK_BASE}*' found in {DATA_DIR}")
            sys.exit(1)

        if args.spk is not None:
            speakers = [f"{SPK_BASE}{n}" for n in args.spk]
            for s in speakers:
                if s not in all_speakers:
                    print(f"Unknown speaker: {s} (valid: {all_speakers})")
                    sys.exit(1)
        else:
            speakers = all_speakers

        for spk in speakers:
            print(f"\n[{spk}]")
            process_speaker(spk)

    print("\nDone.")


if __name__ == "__main__":
    main()
