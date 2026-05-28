#!/usr/bin/env python3
# ABOUTME: Plots a randomly selected MRI frame as a side-by-side PDF panel — left: frame + region mask overlays + tracked articulator points; right: frame + region mask overlays + upper tongue contour.
# ABOUTME: Reads config.json for data_dir / spk_base / video_dir; loads pre-extracted masks (NPZ), contours (NPY), and tracked points (JSON).
"""
For one randomly selected video × frame in a speaker's data directory, render a
two-panel PDF over the original MRI frame:

  Left  : MRI frame + raw region-mask overlays + bright-red tracked points
          (velum centroid, TT, TB-tongue, TR-root, and the upper/lower lip
           reference pair connected by a line)
  Right : MRI frame + raw region-mask overlays + the upper tongue contour

No axes, ticks, or titles. Coordinates live in 104×104 mask pixel space; the
MRI frame is loaded from the corresponding .avi and resized to 104×104.

A new random video × frame is chosen on every run (unless --seed is passed),
and any existing PDFs in the output directory are deleted first.

Single-speaker mode is used when `spk_base` is empty in config.json.
Otherwise pass `--spk N` to choose speaker `{spk_base}{N}`.

Usage:
    conda run -n myenv python plot_pts_contour.py [--output-dir DIR] [--spk N] [--seed S]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ── Load config ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

DATA_DIR  = Path(_cfg["data_dir"])
SPK_BASE  = _cfg.get("spk_base", "")
VIDEO_DIR = _cfg.get("video_dir", "video")

STANDARD_SIZE = 104  # mask pixel space


# ── Rendering constants ─────────────────────────────────────────────────────
# Colors assigned in order from the default matplotlib cycle (tab10) so they
# match the prompting notebook's `show_mask` overlay.
_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
REGION_DEFS = [
    {"substring": "upper lip", "color": _CYCLE[0]},
    {"substring": "lower lip", "color": _CYCLE[1]},
    {"substring": "tongue",    "color": _CYCLE[2]},
    {"substring": "velum",     "color": _CYCLE[3]},
    # {"substring": "larynx",  "color": _CYCLE[4]},  # disabled
]

MASK_ALPHA  = 0.5
POINT_COLOR = "#ff1744"   # bright red for overlays
POINT_SIZE  = 60
LINE_WIDTH  = 5.0
CONTOUR_LW  = 5.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _find_mask_key(keys, substring):
    sub = substring.lower()
    for k in keys:
        if sub in k.lower():
            return k
    return None


def overlay_masks(ax, region_frames):
    """
    Imshow each raw mask as a translucent solid-color layer on top of `ax`'s
    current image. No smoothing, no contour extraction.

    region_frames: list of (color_hex, 2-D bool mask).
    """
    for color, mask in region_frames:
        if mask is None or not mask.any():
            continue
        h, w = mask.shape
        rgb = mcolors.to_rgb(color)
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3]  = mask.astype(np.float32) * MASK_ALPHA
        ax.imshow(rgba, interpolation="nearest")


def _valid_xy(arr, t):
    """Return (x, y) at frame t if both are finite, else None."""
    if arr is None or t >= len(arr):
        return None
    x, y = float(arr[t][0]), float(arr[t][1])
    if np.isnan(x) or np.isnan(y):
        return None
    return x, y


def draw_left_overlay(ax, pts, t):
    """Draw tracked-point dots and lip-pair line at frame t."""
    for key in ("velum_centroids", "tt_points", "tb_tongue_pts", "tr_root_pts"):
        xy = _valid_xy(np.asarray(pts[key]) if key in pts else None, t)
        if xy is not None:
            ax.scatter([xy[0]], [xy[1]], s=POINT_SIZE,
                       c=POINT_COLOR, zorder=5, linewidths=0)

    up = _valid_xy(np.asarray(pts["la_upper_pts"]) if "la_upper_pts" in pts else None, t)
    lo = _valid_xy(np.asarray(pts["la_lower_pts"]) if "la_lower_pts" in pts else None, t)
    if up is not None and lo is not None:
        ax.plot([up[0], lo[0]], [up[1], lo[1]],
                color=POINT_COLOR, linewidth=LINE_WIDTH, zorder=4)
        ax.scatter([up[0], lo[0]], [up[1], lo[1]], s=POINT_SIZE,
                   c=POINT_COLOR, zorder=5, linewidths=0)


def draw_right_overlay(ax, contour_frame):
    """Draw the upper tongue contour as a red polyline. contour_frame: (2, N)."""
    if contour_frame is None:
        return
    xs = np.asarray(contour_frame[0])
    ys = np.asarray(contour_frame[1])
    valid = ~(np.isnan(xs) | np.isnan(ys))
    if valid.sum() < 2:
        return
    ax.plot(xs[valid], ys[valid], color=POINT_COLOR,
            linewidth=CONTOUR_LW, zorder=5)


# ── Data discovery ─────────────────────────────────────────────────────────

def resolve_base_dir(args):
    """Return (base_dir, spk_or_None) based on config + CLI."""
    if SPK_BASE:
        if args.spk is None:
            print(f"--spk N is required when spk_base='{SPK_BASE}' is set", file=sys.stderr)
            sys.exit(2)
        spk = f"{SPK_BASE}{args.spk}"
        base = DATA_DIR / spk
        return base, spk
    return DATA_DIR, None


def discover_basenames(base, spk):
    mask_dir    = base / "sam_seg" / "masks"
    contour_dir = base / "tongue_contours"
    pts_dir     = base / "kinematics" / "raw"

    if not mask_dir.is_dir():
        print(f"Missing mask dir: {mask_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = f"{spk}_*.npz" if spk else "*.npz"
    mask_files = sorted(mask_dir.glob(pattern))

    basenames = []
    for mf in mask_files:
        basename = mf.stem[len(spk) + 1:] if spk else mf.stem
        if (contour_dir / f"{basename}.npy").exists() and \
           (pts_dir / f"{basename}.json").exists():
            basenames.append((basename, mf))

    if not basenames:
        print(f"No basenames with masks + contours + pts found under {base}",
              file=sys.stderr)
        sys.exit(1)
    return basenames, contour_dir, pts_dir


# ── Main ────────────────────────────────────────────────────────────────────

def _load_mri_frame(video_path, t):
    """Load frame `t` from `video_path`, resize to STANDARD_SIZE×STANDARD_SIZE,
    return grayscale uint8 (H, W). Raises if frame can't be read."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {t} from {video_path}")
    frame = cv2.resize(frame, (STANDARD_SIZE, STANDARD_SIZE),
                       interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a random MRI frame as a side-by-side PDF "
                    "(frame + regions + points | frame + regions + tongue contour)."
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.path.join(DATA_DIR, "sam_seg")),
                        help="Directory to write the output PDF.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed (default: nondeterministic).")
    if SPK_BASE:
        parser.add_argument("--spk", type=int, required=True, metavar="N",
                            help=f"Speaker number (prefix: '{SPK_BASE}').")
    else:
        parser.add_argument("--spk", type=int, default=None,
                            help=argparse.SUPPRESS)
    args = parser.parse_args()

    base, spk = resolve_base_dir(args)
    basenames, contour_dir, pts_dir = discover_basenames(base, spk)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any old PDFs so the directory only ever holds the current run.
    for old in output_dir.glob("*.pdf"):
        try:
            old.unlink()
        except OSError as e:
            print(f"Warning: could not remove {old}: {e}", file=sys.stderr)

    rng = random.Random(args.seed)
    basename, mask_path = rng.choice(basenames)

    # Load mask NPZ and pick a random frame
    npz = np.load(mask_path)
    keys = list(npz.keys())

    T = None
    for k in keys:
        a = npz[k]
        if a.ndim == 3:
            T = a.shape[0]
            break
    if T is None:
        print(f"No 3-D mask arrays in {mask_path}", file=sys.stderr)
        sys.exit(1)

    t = rng.randrange(T)

    region_frames = []
    for rd in REGION_DEFS:
        key = _find_mask_key(keys, rd["substring"])
        if key is None:
            continue
        region_frames.append((rd["color"], npz[key][t]))

    contours_all  = np.load(contour_dir / f"{basename}.npy")
    contour_frame = contours_all[t] if t < len(contours_all) else None

    with open(pts_dir / f"{basename}.json") as f:
        pts = json.load(f)

    # MRI frame from the corresponding video, resized to mask space
    video_path = base / VIDEO_DIR / f"{basename}.avi"
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    mri = _load_mri_frame(video_path, t)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    for ax in (ax_l, ax_r):
        ax.imshow(mri, cmap="gray", interpolation="nearest")
        overlay_masks(ax, region_frames)

    draw_left_overlay(ax_l, pts, t)
    draw_right_overlay(ax_r, contour_frame)

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.5, STANDARD_SIZE - 0.5)
        ax.set_ylim(STANDARD_SIZE - 0.5, -0.5)  # image coords

    plt.tight_layout()
    out_path = output_dir / f"{basename}_frame{t:04d}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
