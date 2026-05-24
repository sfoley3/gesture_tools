#!/usr/bin/env python3
# ABOUTME: Computes per-frame MCI (Dawson 2016) and NINFL (Preston 2019) from tongue contour NPY files.
# ABOUTME: Outputs per-video NPY arrays (T,) saved in mci/ and ninfl/ directories per speaker.
"""
Extracts lingual complexity measures from pre-extracted tongue contours.

For each speaker × video:
  - Loads the tongue contour NPY file (T, 2, 100) from tongue_contours/
  - For each frame, computes:
      MCI   — Modified Curvature Index (Dawson 2016): Butterworth-filtered
              absolute curvature integrated over arc length via Simpson's rule
      NINFL — Number of inflections (Preston 2019): sign changes in
              trimmed curvature after fix_curl preprocessing

Output NPY shape: (T,) float32 — one scalar per frame

Usage:
    conda run -n myenv python extract_tongue_complexity.py [--spk spk2 spk3 ...]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ── Load config ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

DATA_DIR  = Path(_cfg["data_dir"])
SPK_BASE  = _cfg.get("spk_base", "")


# ── MCI (Dawson 2016) ──────────────────────────────────────────────────────

def curvature_index(data: np.ndarray) -> float:
    """
    Modified Curvature Index (Dawson 2016).

    data: (N, 2) array of [x, y] contour points.
    Returns scalar MCI value (integral of |filtered curvature| over arc length).
    Returns NaN if the contour has fewer than 2 points.
    """
    if len(data) < 2:
        return np.nan
    dx = np.gradient(data[:, 0])
    dy = np.gradient(data[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    cur = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    s = np.cumsum(np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1)))
    s = np.insert(s, 0, 0)

    b, a = butter(5, 1.0 / 4.0)
    n = len(data)
    r = cur[::-1]
    fcur = filtfilt(b, a, np.concatenate((r, cur, r)))
    fcur = fcur[n:-n]

    fcurA = np.abs(fcur)
    mci = simpson(fcurA, x=s)

    return float(mci)


# ── NINFL (Preston 2019) ───────────────────────────────────────────────────

def _resample_contour(xy: np.ndarray, n_pts: int = 100) -> np.ndarray:
    """Resample contour to n_pts equally spaced points along arc length."""
    dist = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)))])
    t = np.linspace(0, dist[-1], n_pts)
    return np.column_stack([np.interp(t, dist, xy[:, i]) for i in range(2)])


def _fix_curl(xy: np.ndarray) -> np.ndarray:
    """
    Remove or flip non-monotonic (curl-over) points in the x-direction.
    Truncates leading/trailing overcurl where the open end is higher in y.
    """
    def nonmono(arr):
        return np.where(np.sign(np.diff(arr[:, 0])) < 1)[0]

    q = nonmono(xy)

    # All negative: flip left-to-right
    if len(q) + 1 == len(xy):
        xy = xy[::-1].copy()
        q = nonmono(xy)

    if len(q) == 0:
        return xy

    # More than half negative: flip
    if len(q) > len(xy) / 2:
        xy = xy[::-1].copy()
        q = nonmono(xy)

    if len(q) == 0:
        return xy

    # Leading overcurl: q starts at index 0
    if q[0] == 0:
        gaps = np.where(np.diff(q) > 1)[0]
        n_run = gaps[0] + 1 if len(gaps) > 0 else len(q)
        if xy[0, 1] < xy[q[n_run - 1], 1]:
            xy = np.delete(xy, q[:n_run], axis=0)
            q = nonmono(xy)

    if len(q) == 0:
        return xy

    # Trailing overcurl
    if xy[-1, 1] < xy[q[0], 1] and xy[-1, 0] < xy[q.max(), 0]:
        xy = xy[:q[0]].copy()

    return xy


def compute_curvature(
    xy: np.ndarray,
    trim: float = 0.3,
    n_pts: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Compute signed curvature and inflection count of a 2D contour.

    Parameters
    ----------
    xy     : (N, 2) array of [x, y] contour points
    trim   : fraction of arc length used as radius threshold for trimming
    n_pts  : resample to this many points; 0 = skip resampling

    Returns
    -------
    k      : signed curvature at each point
    n_infl : number of inflections in the trimmed curvature signal
    """
    xy = np.array(xy, dtype=float)
    if xy.shape[0] < xy.shape[1]:
        xy = xy.T

    if n_pts > 0:
        xy = _resample_contour(xy, n_pts)

    xy = _fix_curl(xy)

    if len(xy) < 2:
        return np.array([]), 0

    dx  = np.gradient(xy[:, 0])
    dy  = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    arc = np.sum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))) * trim
    fk = k.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fk[np.abs(1.0 / k) > arc] = 0.0
    fk[k == 0] = 0.0

    sfk = np.sign(fk)
    xfk = sfk[sfk != 0]

    if len(xfk) == 0:
        sk = np.sign(k)
        sk = sk[sk != 0]
        n_infl = 0 if len(sk) == 0 else 1
    else:
        n_infl = int(np.sum(np.diff(xfk) != 0)) + 1

    return k, n_infl


# ── Per-frame processing ───────────────────────────────────────────────────

def process_contour_frame(contour_2x100: np.ndarray) -> tuple[float, float]:
    """
    Compute MCI and NINFL for a single frame's tongue contour.

    contour_2x100: (2, 100) array [x_coords, y_coords].
    Returns (mci, n_infl) — both float; NaN if the contour is degenerate.
    """
    if np.isnan(contour_2x100).all():
        return np.nan, np.nan

    xy = contour_2x100.T  # (100, 2)

    # Check for degenerate contour (all same point)
    if np.ptp(xy[:, 0]) == 0 and np.ptp(xy[:, 1]) == 0:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mci = curvature_index(xy)
        _, n_infl = compute_curvature(xy, n_pts=0)

    return float(mci), float(n_infl)


# ── Per-video processing ───────────────────────────────────────────────────

def process_video(contour_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MCI and NINFL for all frames in a tongue contour file.

    contour_path: path to NPY file of shape (T, 2, 100).
    Returns (mci_arr, ninfl_arr) — both shape (T,) float32.
    """
    contours = np.load(contour_path)  # (T, 2, 100)
    T = contours.shape[0]
    mci_arr = np.full(T, np.nan, dtype=np.float32)
    ninfl_arr = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):
        mci_arr[t], ninfl_arr[t] = process_contour_frame(contours[t])

    return mci_arr, ninfl_arr


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

    contour_dir = base / "tongue_contours"
    mci_dir     = base / "mci"
    ninfl_dir   = base / "ninfl"

    if spk is not None:
        contour_files = sorted(contour_dir.glob(f"{spk}_*.npy"))
    else:
        contour_files = sorted(contour_dir.glob("*.npy"))

    if not contour_files:
        print(f"  No contour files found in {contour_dir}")
        return

    mci_dir.mkdir(parents=True, exist_ok=True)
    ninfl_dir.mkdir(parents=True, exist_ok=True)

    for contour_path in tqdm(contour_files, desc=f"  {label}"):
        basename = contour_path.stem
        try:
            mci_arr, ninfl_arr = process_video(contour_path)
        except Exception as e:
            print(f"    ERROR processing {contour_path.name}: {e}")
            continue

        np.save(mci_dir / f"{basename}.npy", mci_arr)
        np.save(ninfl_dir / f"{basename}.npy", ninfl_arr)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    single_speaker = not SPK_BASE

    parser = argparse.ArgumentParser(description="Extract MCI and NINFL from tongue contours.")
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
