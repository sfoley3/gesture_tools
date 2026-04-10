#!/usr/bin/env python3
# ABOUTME: Extracts articulatory kinematics (velum, tongue tip, lip aperture, tongue body, tongue root) from SAM2 mask NPZ files.
# ABOUTME: Outputs per-video NPY timeseries (T, 6) and diagnostic MP4 overlays with color-coded tracked points.
"""
Processes SAM2 segmentation masks to extract raw articulatory kinematic timeseries.
Smoothing and velocity computation are handled downstream in get_sam_gestures.py.

For each speaker × video:
  - Velum: tip position (x, y) in 104×104 pixel space — right-half centroid of largest component
  - Tongue tip: distance to alveolar ridge reference point
  - Lip aperture: distance between closest points on lower lip and upper lip masks
  - Tongue body: distance from closest tongue pixel to leftmost velum pixel (per-frame)
  - Tongue root: x-coordinate of rightmost tongue pixel

Output NPY shape: (T, 6) — [velum_x, velum_y, tt_dist, lip_aperture, tongue_body_dist, tongue_root_x]

Usage:
    conda run -n myenv python extract_mask_kinematics.py [--spk spk1 spk2 ...]
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label, binary_erosion
from tqdm import tqdm

DATA_DIR     = Path("/data1/span_data/prompt/data/mri")
ALL_SPEAKERS = ["spk2", "spk3", "spk4", "spk5", "spk6", "spk7", "spk8", "spk9", "spk10"]
N_DIAGNOSTIC = 5

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


def compute_tongue_root(tongue_masks: np.ndarray):
    """
    tongue_masks: (T, H, W) bool
    Tracks the rightmost point on the tongue mask each frame.
    Returns (x_arr, root_points) — x_arr is (T,), root_points is (T, 2).
    """
    T = tongue_masks.shape[0]
    x_arr = np.full(T, np.nan, dtype=np.float32)
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

        x_arr[t] = root_x
        root_pts[t] = [root_x, root_y]

    x_arr = _fill_nans(x_arr)
    root_pts = _fill_nans_2d(root_pts)
    return x_arr, root_pts


def extract_kinematics(mask_path: Path, video_path: Path) -> tuple:
    """
    Returns (kinematics_array, fps, tracked_points, mask_data).
    kinematics_array shape: (T, 6) — [velum_x, velum_y, tt_dist, lip_aperture,
                                        tongue_body_dist, tongue_root_x]
    tracked_points: dict of per-feature point arrays for diagnostic drawing.
    """
    data = np.load(mask_path)
    keys = list(data.keys())
    velum_masks = data["velum"]
    tongue_masks = data["tongue"]
    palate_key = _find_mask_key(keys, "upper lip")
    lower_lip_key = _find_mask_key(keys, "lower lip")
    palate_masks = data[palate_key]
    lower_lip_masks = data[lower_lip_key]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"    Warning: could not read FPS from {video_path.name}, defaulting to 30")
    cap.release()

    vx, vy = compute_velum_kinematics(velum_masks)
    tt_dist, tt_points, alveolar_ridge = compute_tt_kinematics(tongue_masks, palate_masks)
    la_dist, la_upper_pts, la_lower_pts = compute_lip_aperture(lower_lip_masks, palate_masks)
    tb_dist, tb_tongue_pts, tb_velum_pts = compute_tongue_body(tongue_masks, velum_masks)
    tr_x, tr_root_pts = compute_tongue_root(tongue_masks)

    kinematics = np.stack([vx, vy, tt_dist, la_dist, tb_dist, tr_x], axis=1).astype(np.float32)
    velum_centroids = np.stack([vx, vy], axis=1)

    tracked_points = {
        "velum_centroids": velum_centroids,
        "tt_points": tt_points,
        "alveolar_ridge": alveolar_ridge,
        "la_upper_pts": la_upper_pts,
        "la_lower_pts": la_lower_pts,
        "tb_tongue_pts": tb_tongue_pts,
        "tb_velum_pts": tb_velum_pts,
        "tr_root_pts": tr_root_pts,
    }

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

    mask_h, mask_w = mask_data["velum"].shape[1], mask_data["velum"].shape[2]

    velum_centroids = tracked_points["velum_centroids"]
    tt_points       = tracked_points["tt_points"]
    alveolar_ridge  = tracked_points["alveolar_ridge"]
    la_upper_pts    = tracked_points["la_upper_pts"]
    la_lower_pts    = tracked_points["la_lower_pts"]
    tb_tongue_pts   = tracked_points["tb_tongue_pts"]
    tb_velum_pts    = tracked_points["tb_velum_pts"]
    tr_root_pts     = tracked_points["tr_root_pts"]

    all_keys = list(mask_data.keys())
    # Build ordered (key, color) pairs; match by substring for lip masks
    region_colors = [
        ("velum", PURPLE),
        ("tongue", BLUE),
        (_find_mask_key(all_keys, "upper lip"), GREEN),
        (_find_mask_key(all_keys, "lower lip"), ORANGE),
    ]

    def _draw_dot(frame, pts, t, color):
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
            if region not in mask_data:
                continue
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
        _draw_dot(frame, velum_centroids, t, RED)

        # Tongue tip closest point
        _draw_dot(frame, tt_points, t, RED)

        # Lip aperture: upper and lower closest points + connecting line
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
        _draw_dot(frame, tb_tongue_pts, t, RED)

        # Tongue root: rightmost tongue point
        _draw_dot(frame, tr_root_pts, t, RED)

        writer.write(frame)

    cap.release()
    writer.release()


# ── Per-speaker processing ──────────────────────────────────────────────────

def process_speaker(spk: str):
    mask_dir = DATA_DIR / spk / "sam_seg" / "masks"
    video_dir = DATA_DIR / spk / "video_split"
    kin_dir = DATA_DIR / spk / "sam_seg" / "kinematics"
    diag_dir = DATA_DIR / spk / "sam_seg" / "diagnostic"

    mask_files = sorted(mask_dir.glob(f"{spk}_*.npz"))
    if not mask_files:
        print(f"  No mask files found for {spk}")
        return

    kin_dir.mkdir(parents=True, exist_ok=True)

    # Select 5 random files for diagnostic videos (reproducible per speaker)
    rng = random.Random(sum(ord(c) for c in spk))
    diag_files = rng.sample(mask_files, min(N_DIAGNOSTIC, len(mask_files)))
    diag_set = set(f.name for f in diag_files)

    results = {}  # basename → (kinematics, fps, tracked_points, mask_data)

    for mask_path in tqdm(mask_files, desc=f"  {spk} kinematics"):
        # e.g. spk1_spk1_1.npz  →  basename = spk1_1
        basename = mask_path.stem[len(spk) + 1:]  # strip "{spk}_" prefix
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
            results.items(), desc=f"  {spk} diagnostic videos"
        ):
            out_mp4 = diag_dir / f"{basename}_diagnostic.mp4"
            mask_path = mask_dir / f"{spk}_{basename}.npz"
            try:
                write_diagnostic_video(mask_path, video_path, out_mp4, tracked_pts, mask_data)
            except Exception as e:
                print(f"    ERROR writing diagnostic video {basename}: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract mask kinematics from SAM2 NPZ files.")
    parser.add_argument("--spk", nargs="+", default=ALL_SPEAKERS, metavar="SPK",
                        help="Speakers to process (default: all)")
    args = parser.parse_args()

    for spk in args.spk:
        if spk not in ALL_SPEAKERS:
            print(f"Unknown speaker: {spk} (valid: {ALL_SPEAKERS})")
            sys.exit(1)

    for spk in args.spk:
        print(f"\n[{spk}]")
        process_speaker(spk)

    print("\nDone.")


if __name__ == "__main__":
    main()
