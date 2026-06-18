#!/usr/bin/env python3
# ABOUTME: Extracts upper tongue contours from SAM2 tongue mask NPZ files using contour tracing and upper-half splitting.
# ABOUTME: Outputs per-video NPY contour arrays (T, 2, 100) and diagnostic MP4 overlays with red contour lines.
"""
Extracts the oral-cavity-facing (upper) tongue surface contour from SAM2 segmentation masks.

For each speaker × video:
  - Loads the tongue and lower lip masks from the SAM2 NPZ file (T, 104, 104 bool)
  - Finds the largest connected component per frame
  - Traces the outer contour via cv2.findContours
  - Derives a fixed jaw reference point from the median per-frame tongue-to-lower-lip
    junction across all frames (two-pass approach for stability)
  - Splits the contour at two anchors: the tongue contour point closest to the
    jaw reference (anterior) and the rightmost point (root/posterior)
  - Keeps the upper path (oral-cavity-facing surface, including retroflexed tips)
  - Resamples to 100 equidistant points via arc-length parameterization

Output NPY shape: (T, 2, 100) — [x_coords, y_coords] in 104×104 pixel space

Usage:
    conda run -n myenv python get_tongue_contours.py [--spk spk2 spk3 ...]
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
from tqdm import tqdm

# ── Load config ─────────────────────────────────────────────────────────────
# Defaults let the module import cleanly when pip-installed (no repo-root
# config.json present), so importing the extraction functions Just Works.
_DEFAULT_CFG = {
    "data_dir":     ".",
    "n_diagnostic": 10,
    "spk_base":     "",
    "video_dir":    "video",
}


def _load_config() -> dict:
    """Load config.json, falling back gracefully so imports never crash.

    Search order:
      1. Path in the GESTURE_TOOLS_CONFIG environment variable, if set.
      2. config.json at the repo root (next to the package) — dev/clone layout.
    Missing keys are filled from _DEFAULT_CFG; if no file is found the defaults
    are used in full.
    """
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

DATA_DIR     = Path(_cfg["data_dir"])
N_DIAGNOSTIC = int(_cfg.get("n_diagnostic", 10))
SPK_BASE     = _cfg.get("spk_base", "")
VIDEO_DIR    = _cfg.get("video_dir", "video")
N_POINTS     = 100

# BGR colors
BLUE       = (200, 100, 0)
RED        = (0, 0, 255)
MASK_ALPHA = 1.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask containing only the largest connected component."""
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == sizes.argmax()


def _scale_point(x: float, y: float, mask_h: int, mask_w: int, frame_h: int, frame_w: int):
    sx = int(round(x * frame_w / mask_w))
    sy = int(round(y * frame_h / mask_h))
    return sx, sy


# ── Contour extraction ──────────────────────────────────────────────────────

def _find_jaw_anchor(tongue_masks: np.ndarray, lower_lip_masks: np.ndarray) -> tuple:
    """
    Derive a fixed jaw anchor from per-frame tongue-to-lower-lip junction points.

    First pass: for each frame, trace the tongue contour and find the contour point
    closest to the lower lip mask (the natural tongue-jaw junction). Collect all
    junction points across frames.
    Second pass: take the median (x, y) of those junction points as a fixed anchor.

    This gives an anatomically accurate anchor (derived from actual tongue-lip
    proximity) that is stable across frames.

    Returns (x_ref, y_ref) in mask pixel space, or None if insufficient data.
    """
    T = tongue_masks.shape[0]
    junction_pts = []

    for t in range(T):
        tmask = tongue_masks[t]
        llmask = lower_lip_masks[t]
        if not tmask.any() or not llmask.any():
            continue

        core = _largest_component(tmask)
        binary = (core.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=len)
        pts = contour.squeeze()
        if pts.ndim != 2 or len(pts) < 4:
            continue

        # Find tongue contour point closest to any lower lip pixel
        lip_ys, lip_xs = np.where(llmask)
        dx = pts[:, 0:1].astype(np.float32) - lip_xs[None, :].astype(np.float32)
        dy = pts[:, 1:2].astype(np.float32) - lip_ys[None, :].astype(np.float32)
        min_dist_per_pt = (dx ** 2 + dy ** 2).min(axis=1)
        idx = int(min_dist_per_pt.argmin())
        junction_pts.append(pts[idx].astype(np.float32))

    if not junction_pts:
        return None

    junction_arr = np.stack(junction_pts)  # (N_valid, 2)
    x_ref = float(np.median(junction_arr[:, 0]))
    y_ref = float(np.median(junction_arr[:, 1]))
    return x_ref, y_ref


def extract_upper_contour(mask_2d: np.ndarray, jaw_ref: tuple) -> np.ndarray | None:
    """
    Extract the upper (oral-cavity-facing) contour from a single-frame tongue mask.

    Uses cv2.findContours to trace the outer boundary of the largest connected
    component, then splits the contour at two anchors:
      - Anterior anchor: the tongue contour point closest to the fixed jaw
        reference point (derived from time-averaged lower lip mask)
      - Posterior anchor: the rightmost contour point (tongue root)
    The upper path (lower mean y) between these anchors is the oral-cavity surface,
    including the underside of a retroflexed tongue tip during /r/.

    jaw_ref: (x, y) fixed reference point derived from median per-frame
        tongue-to-lower-lip junction, or None to fall back to leftmost point.

    Returns an (M, 2) float32 array of [x, y] points ordered anterior-to-posterior,
    or None if the mask is empty.
    """
    if not mask_2d.any():
        return None

    core = _largest_component(mask_2d)
    binary = (core.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Take the longest contour (should be the only external one after largest-component)
    contour = max(contours, key=len)
    pts = contour.squeeze()  # (N, 2) — columns are [x, y]
    if pts.ndim != 2 or len(pts) < 4:
        return None

    # Posterior anchor: rightmost contour point (tongue root)
    idx_root = int(pts[:, 0].argmax())

    # Anterior anchor: tongue contour point closest to the fixed jaw reference
    if jaw_ref is not None:
        ref_x, ref_y = jaw_ref
        dists = (pts[:, 0] - ref_x) ** 2 + (pts[:, 1] - ref_y) ** 2
        idx_junction = int(dists.argmin())
    else:
        # Fallback: leftmost contour point
        idx_junction = int(pts[:, 0].argmin())

    # Order indices so idx_a < idx_b for consistent slicing
    idx_a = min(idx_junction, idx_root)
    idx_b = max(idx_junction, idx_root)

    # Split contour into two paths between the two anchors
    path_a = pts[idx_a:idx_b + 1]                                    # direct slice
    path_b = np.concatenate([pts[idx_b:], pts[:idx_a + 1]])           # wraparound

    # Upper path has lower mean y (closer to top of image = oral cavity)
    if path_a[:, 1].mean() <= path_b[:, 1].mean():
        upper = path_a
    else:
        upper = path_b

    # Ensure anterior-to-posterior ordering (ascending x overall)
    if upper[0, 0] > upper[-1, 0]:
        upper = upper[::-1]

    return upper.astype(np.float32)


def resample_contour(contour: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """
    Resample a variable-length contour to exactly n_points equidistant points.

    Uses arc-length parameterization with linear interpolation.
    contour: (M, 2) array of [x, y] points.
    Returns: (2, n_points) float32 array [x_coords, y_coords].
    """
    # Cumulative arc length
    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len == 0:
        # Degenerate contour (all same point)
        pt = contour[0]
        return np.tile(pt, (n_points, 1)).T.astype(np.float32)

    # Evenly spaced positions along the arc
    target_positions = np.linspace(0, total_len, n_points)

    x_resampled = np.interp(target_positions, cum_len, contour[:, 0])
    y_resampled = np.interp(target_positions, cum_len, contour[:, 1])

    return np.stack([x_resampled, y_resampled], axis=0).astype(np.float32)


def _find_mask_key(keys, substring: str) -> str:
    """Find the first key in *keys* that contains *substring* (case-insensitive)."""
    sub = substring.lower()
    for k in keys:
        if sub in k.lower():
            return k
    raise KeyError(f"No mask key containing '{substring}' in {list(keys)}")


def extract_contours(mask_path: Path, video_path: Path | None) -> tuple:
    """
    Extract resampled upper tongue contours for all frames in a video.

    video_path may be None if the video file is missing; FPS defaults to 30
    and no video-derived information is used.

    Returns (contours_array, fps, tongue_masks).
    contours_array shape: (T, 2, N_POINTS) float32
    tongue_masks: (T, H, W) bool — for diagnostic video overlay
    """
    data = np.load(mask_path)
    tongue_masks = data["tongue"]
    lower_lip_key = _find_mask_key(list(data.keys()), "lower lip")
    lower_lip_masks = data[lower_lip_key]
    T = tongue_masks.shape[0]

    # Compute fixed jaw anchor from median per-frame tongue-to-lower-lip junction
    jaw_ref = _find_jaw_anchor(tongue_masks, lower_lip_masks)

    fps = 50.0
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        _fps = cap.get(cv2.CAP_PROP_FPS)
        if _fps <= 0:
            print(f"    Warning: could not read FPS from {video_path.name}, defaulting to 50")
        else:
            fps = _fps
        cap.release()

    contours = np.full((T, 2, N_POINTS), np.nan, dtype=np.float32)

    for t in range(T):
        upper = extract_upper_contour(tongue_masks[t], jaw_ref)
        if upper is not None and len(upper) >= 2:
            contours[t] = resample_contour(upper, N_POINTS)

    # Forward-fill then backward-fill frames with NaN contours
    for t in range(1, T):
        if np.isnan(contours[t, 0, 0]):
            contours[t] = contours[t - 1]
    for t in range(T - 2, -1, -1):
        if np.isnan(contours[t, 0, 0]):
            contours[t] = contours[t + 1]

    return contours, fps, tongue_masks


# ── Diagnostic video ────────────────────────────────────────────────────────

def write_diagnostic_video(
    video_path: Path | None,
    out_path: Path,
    contours: np.ndarray,
    tongue_masks: np.ndarray,
):
    """Write a diagnostic MP4 with tongue mask overlay (blue) and contour line (red).

    contours: (T, 2, N_POINTS) float32 in mask pixel space (104×104)
    tongue_masks: (T, H, W) bool

    If video_path is None (video file missing), synthesizes black frames from the
    mask dimensions and uses FPS=50, so diagnostics are always produced.
    """
    mask_h, mask_w = tongue_masks.shape[1], tongue_masks.shape[2]

    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap = None
        fps = 50.0
        frame_h, frame_w = mask_h, mask_w
        n_frames = tongue_masks.shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    for t in range(n_frames):
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Overlay tongue mask in blue
        if t < tongue_masks.shape[0] and tongue_masks[t].any():
            m_resized = cv2.resize(
                tongue_masks[t].astype(np.uint8) * 255, (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST
            )
            overlay = frame.copy()
            colored = np.zeros_like(frame)
            colored[m_resized > 0] = BLUE
            cv2.addWeighted(colored, MASK_ALPHA, overlay, 1.0, 0, overlay)
            frame = overlay

        # Draw contour as bright red polyline
        if t < contours.shape[0] and not np.isnan(contours[t, 0, 0]):
            xs = contours[t, 0]
            ys = contours[t, 1]
            scaled_pts = np.array([
                _scale_point(x, y, mask_h, mask_w, frame_h, frame_w)
                for x, y in zip(xs, ys)
            ], dtype=np.int32)
            cv2.polylines(frame, [scaled_pts], isClosed=False, color=RED, thickness=2)

        writer.write(frame)

    if cap is not None:
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
    contour_dir = base / "tongue_contours"
    diag_dir = contour_dir / "diagnostic"

    if not video_dir.exists():
        print(f"  Missing video_dir: {video_dir}")

    if spk is not None:
        mask_files = sorted(mask_dir.glob(f"{spk}_*.npz"))
    else:
        mask_files = sorted(mask_dir.glob("*.npz"))

    if not mask_files:
        print(f"  No mask files found in {mask_dir}")
        return

    contour_dir.mkdir(parents=True, exist_ok=True)

    # Select random files for diagnostic videos (reproducible)
    seed = sum(ord(c) for c in label)
    rng = random.Random(seed)
    diag_files = rng.sample(mask_files, min(N_DIAGNOSTIC, len(mask_files)))
    diag_set = set(f.name for f in diag_files)

    diag_results = {}

    for mask_path in tqdm(mask_files, desc=f"  {label} contours"):
        if spk is not None:
            basename = mask_path.stem[len(spk) + 1:]
        else:
            basename = mask_path.stem
        video_path = video_dir / f"{basename}.avi"

        video_missing = not video_path.exists()
        if video_missing:
            print(f"    Missing video: {video_path}")

        try:
            contours, fps, tongue_masks = extract_contours(
                mask_path, None if video_missing else video_path
            )
        except Exception as e:
            print(f"    ERROR extracting {mask_path.name}: {e}")
            continue

        out_npy = contour_dir / f"{basename}.npy"
        np.save(out_npy, contours)

        if mask_path.name in diag_set:
            diag_results[basename] = (contours, tongue_masks, None if video_missing else video_path)

    # Write diagnostic videos
    if diag_results:
        diag_dir.mkdir(parents=True, exist_ok=True)
        for basename, (contours, tongue_masks, video_path) in tqdm(
            diag_results.items(), desc=f"  {label} diagnostic videos"
        ):
            out_mp4 = diag_dir / f"{basename}_contour_diagnostic.mp4"
            try:
                write_diagnostic_video(video_path, out_mp4, contours, tongue_masks)
            except Exception as e:
                print(f"    ERROR writing diagnostic video {basename}: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    single_speaker = not SPK_BASE

    parser = argparse.ArgumentParser(description="Extract tongue contours from SAM2 NPZ files.")
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