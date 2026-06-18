#!/usr/bin/env python3
"""
get_tt_region.py — Derive the tongue-tip (TT) region from raw tracked kinematics.

Loads every raw TT (x, y) point from {data_dir}/kinematics/raw/{stem}.json,
pools them across all frames and all utterances, fits a 95% confidence ellipse
to the resulting 2-D pixel-space distribution, and reports:

  * the distribution centre  — the pixel-space mean of all TT points,
  * the 95% confidence ellipse geometry (semi-axes + rotation),
  * a single circular radius (in pixels) that captures 95% of the points.

The circular centre + radius can be dropped straight into config.json as
tt_cx / tt_cy / tt_radius for the tongue-tip region loss in src/losses.py.

A diagnostic figure (TT points + derived ellipse overlaid on the per-pixel
mask-std heatmap, same format as visualize_mask_std.py) is written next to the
kinematics directory as {data_dir}/kinematics/tt_region.pdf.

Raw JSON layout (produced by extract_mask_kinematics.py)
--------------------------------------------------------
Each {stem}.json holds tracked-point arrays keyed by region, e.g.
  "tt_points":      [[x, y], [x, y], ...]   # (T, 2) — the tongue tip
  "tb_tongue_pts":  ...
  "tr_root_pts":    ...
  ...
Frames with no detection are stored as NaN and are dropped here.

Usage
-----
python src/get_tt_region.py --config config.json
python src/get_tt_region.py --config config.json --output tt_region.json
python src/get_tt_region.py --raw-dir /path/to/kinematics/raw --output tt_region.json
"""

import argparse
import glob
import json
import os

import numpy as np

try:
    # scipy is already a dependency of the repo (see get_corr.py / requirements).
    from scipy.stats import chi2
    _CHI2_95_DF2 = float(chi2.ppf(0.95, df=2))
except Exception:  # pragma: no cover - fallback if scipy is unavailable
    # chi-square value for 2 dof at the 0.95 quantile.
    _CHI2_95_DF2 = 5.991464547107979


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tt_points_from_json(json_path: str, tt_key: str = "tt_points") -> np.ndarray:
    """
    Load the (T, 2) array of TT (x, y) points from one raw kinematics JSON.

    Returns an (N, 2) float array of finite points (NaN frames removed). If the
    key is missing or empty, returns an empty (0, 2) array.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if tt_key not in data or data[tt_key] is None:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.asarray(data[tt_key], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)

    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def collect_tt_points(raw_dir: str, tt_key: str = "tt_points"):
    """
    Pool TT points from every {stem}.json in raw_dir.

    Returns:
        all_pts:  (N, 2) array of pooled finite TT points
        n_files:  number of JSON files that contributed at least one point
    """
    json_paths = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in raw dir: {raw_dir}")

    chunks = []
    n_files = 0
    for p in json_paths:
        pts = load_tt_points_from_json(p, tt_key=tt_key)
        if len(pts):
            chunks.append(pts)
            n_files += 1

    if not chunks:
        raise ValueError(
            f"No valid '{tt_key}' points found across {len(json_paths)} files in {raw_dir}"
        )

    return np.concatenate(chunks, axis=0), n_files


# ---------------------------------------------------------------------------
# Ellipse / radius fitting
# ---------------------------------------------------------------------------

def fit_confidence_ellipse(points: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Fit a confidence ellipse to a 2-D point cloud.

    The ellipse is the level set of the fitted Gaussian containing `confidence`
    of the probability mass: (p - mu)^T Sigma^-1 (p - mu) = s, with
    s = chi2.ppf(confidence, df=2). Semi-axis lengths are sqrt(s * eigenvalue)
    along each covariance eigenvector.

    Returns a dict with the centre, semi-axes, rotation angle, and the
    chi-square scale used.
    """
    mu = points.mean(axis=0)
    # Sample covariance (ddof=1); columns are x, y.
    cov = np.cov(points, rowvar=False)

    # Eigen-decomposition of a symmetric 2x2 -> orthonormal eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)  # guard tiny negatives from round-off

    # Order largest-first so axis 0 is the major axis.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if confidence >= 1.0:
        scale = np.inf
    else:
        scale = float(chi2.ppf(confidence, df=2)) if "chi2" in globals() else _CHI2_95_DF2
        if not np.isfinite(scale):
            scale = _CHI2_95_DF2

    semi_major = float(np.sqrt(scale * eigvals[0]))
    semi_minor = float(np.sqrt(scale * eigvals[1]))

    # Rotation of the major axis, measured from +x toward +y (image y points down).
    major_vec = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))

    return {
        "center_x": float(mu[0]),
        "center_y": float(mu[1]),
        "semi_major_axis": semi_major,
        "semi_minor_axis": semi_minor,
        "angle_deg": angle_deg,
        "confidence": float(confidence),
        "chi2_scale": float(scale),
        "covariance": cov.tolist(),
    }


def radius_capturing(points: np.ndarray, center: np.ndarray, fraction: float = 0.95) -> float:
    """
    Smallest circular radius about `center` that contains `fraction` of points.

    This is the empirical `fraction`-quantile of the Euclidean distance from the
    centre — a distribution-free counterpart to the Gaussian ellipse, and the
    value that best matches the circular TT mask used in src/losses.py.
    """
    d = np.linalg.norm(points - center[None, :], axis=1)
    return float(np.quantile(d, fraction))


# ---------------------------------------------------------------------------
# Mask-std background (same computation as visualize_mask_std.py)
# ---------------------------------------------------------------------------

def compute_pixel_std(dataset, max_utts: int | None = None):
    """
    For every utterance, compute the per-pixel std-dev across time, then average
    those std maps across all utterances. Returns mean_std (H, W).
    """
    n = len(dataset) if max_utts is None else min(max_utts, len(dataset))
    sum_std = None
    for i in range(n):
        mask = dataset[i]["mask"].numpy()      # (T, H, W) float32 0/1
        pixel_std = mask.std(axis=0)           # (H, W)
        if sum_std is None:
            sum_std = np.zeros_like(pixel_std)
        sum_std += pixel_std
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Processed {i + 1}/{n} utterances")
    return sum_std / n


def load_mask_std_background(cfg, max_utts: int | None = None):
    """
    Load the training set and compute the per-pixel mask-std heatmap used as the
    plot background. Returns (H, W) array, or None if the dataset can't be loaded.
    """
    try:
        from dataset import LSSSpeechDataset, PROMPTSpeechDataset
    except Exception as e:  # pragma: no cover - dataset module/torch unavailable
        print(f"  WARNING: could not import dataset module ({e}); "
              f"plotting points on a blank background.")
        return None

    dataset_type = cfg.get("dataset", "lss")
    data_dir = cfg["data_dir"]
    print(f"Loading {dataset_type} training set from {data_dir} for mask-std background ...")
    if dataset_type == "prompt":
        split_dir = os.path.join(data_dir, "data")
        dataset = PROMPTSpeechDataset.from_split_file(
            data_dir, os.path.join(split_dir, "train.txt"))
    else:
        dataset = LSSSpeechDataset.from_split_file(
            data_dir, os.path.join(data_dir, "train.txt"))
    print(f"  {len(dataset)} utterances")
    print("Computing per-pixel temporal std ...")
    return compute_pixel_std(dataset, max_utts=max_utts)


# ---------------------------------------------------------------------------
# Plotting — TT points + derived ellipse over the mask-std heatmap
# (same format as visualize_mask_std.py)
# ---------------------------------------------------------------------------

def save_plot(points, ellipse, center, radius95, confidence, out_path,
              mean_std=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Background: mask-std heatmap (hot) with a cyan coordinate grid, matching
    # visualize_mask_std.py. imshow uses image coordinates (y increases downward),
    # so no axis inversion is needed and the TT pixel points align directly.
    if mean_std is not None:
        im = ax.imshow(mean_std, cmap="hot", interpolation="nearest")
        H, W = mean_std.shape
        ax.set_xticks(range(0, W, 10))
        ax.set_yticks(range(0, H, 10))
        ax.grid(True, color="cyan", linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.set_aspect("equal")
        ax.invert_yaxis()

    # Overlay the pooled TT points (subsample for a readable scatter).
    pts = points
    if len(pts) > 20000:
        idx = np.random.default_rng(0).choice(len(pts), 20000, replace=False)
        pts = pts[idx]
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.12, color="deepskyblue",
               label=f"TT points (N={len(points)})")

    # Derived confidence ellipse + its centre.
    ell = Ellipse(
        (ellipse["center_x"], ellipse["center_y"]),
        width=2 * ellipse["semi_major_axis"],
        height=2 * ellipse["semi_minor_axis"],
        angle=ellipse["angle_deg"],
        fill=False, edgecolor="lime", lw=2,
        label=f"{int(confidence * 100)}% ellipse",
    )
    ax.add_patch(ell)
    ax.scatter([ellipse["center_x"]], [ellipse["center_y"]],
               color="lime", marker="+", s=140, lw=2,
               label=f"centre ({center[0]:.1f}, {center[1]:.1f})")

    # Final circular tt_region radius.
    circ = Circle(
        (center[0], center[1]), radius95,
        fill=False, edgecolor="white", lw=1.5, linestyle="--",
        label=f"r{int(confidence * 100)} = {radius95:.1f}px",
    )
    ax.add_patch(circ)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json",
                   help="Config JSON providing data_dir (default: config.json).")
    p.add_argument("--raw-dir", default=None,
                   help="Directory of raw kinematics JSON files. Overrides "
                        "{data_dir}/kinematics/raw.")
    p.add_argument("--tt-key", default="tt_points",
                   help="JSON key holding the (T,2) TT points (default: tt_points).")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level for the ellipse / radius (default: 0.95).")
    p.add_argument("--output", default="tt_region.json",
                   help="Output JSON path (default: tt_region.json).")
    p.add_argument("--max-utts", type=int, default=None,
                   help="Cap utterances used for the mask-std plot background "
                        "(faster preview).")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config if present (gives the default raw dir + the plot background).
    cfg = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    data_dir = cfg["data_dir"] if cfg is not None else None

    # Resolve the raw directory.
    raw_dir = args.raw_dir
    if raw_dir is None:
        if data_dir is None:
            raise FileNotFoundError(
                f"--raw-dir not given and config not found at {args.config}")
        raw_dir = os.path.join(data_dir, "kinematics", "raw")

    print(f"Loading raw TT points from: {raw_dir}")
    all_pts, n_files = collect_tt_points(raw_dir, tt_key=args.tt_key)
    print(f"  Pooled {len(all_pts)} finite TT points from {n_files} utterances.")

    center = all_pts.mean(axis=0)
    ellipse = fit_confidence_ellipse(all_pts, confidence=args.confidence)
    r_capture = radius_capturing(all_pts, center, fraction=args.confidence)

    pct = int(round(args.confidence * 100))
    result = {
        "tt_key": args.tt_key,
        "n_points": int(len(all_pts)),
        "n_utterances": int(n_files),
        "confidence": float(args.confidence),
        "center": {"x": float(center[0]), "y": float(center[1])},
        # Full confidence ellipse geometry.
        "ellipse": ellipse,
        # Distribution-free circular radius capturing `confidence` of points.
        f"radius_{pct}_pixels": float(r_capture),
        "distribution": {
            "x_mean": float(all_pts[:, 0].mean()),
            "y_mean": float(all_pts[:, 1].mean()),
            "x_std": float(all_pts[:, 0].std()),
            "y_std": float(all_pts[:, 1].std()),
            "x_min": float(all_pts[:, 0].min()),
            "x_max": float(all_pts[:, 0].max()),
            "y_min": float(all_pts[:, 1].min()),
            "y_max": float(all_pts[:, 1].max()),
        },
        # Ready-to-paste circular TT region for config.json / losses.py.
        "tt_region": {
            "tt_cx": int(round(center[0])),
            "tt_cy": int(round(center[1])),
            "tt_radius": int(round(r_capture)),
        },
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print("\n── TT region ─────────────────────────────────────────────────────")
    print(f"  center           : x={center[0]:.2f}, y={center[1]:.2f}")
    print(f"  {pct}% ellipse axes : major={ellipse['semi_major_axis']:.2f}, "
          f"minor={ellipse['semi_minor_axis']:.2f}, angle={ellipse['angle_deg']:.1f} deg")
    print(f"  {pct}% radius (px)  : {r_capture:.2f}")
    print(f"  config tt_region : tt_cx={result['tt_region']['tt_cx']}, "
          f"tt_cy={result['tt_region']['tt_cy']}, "
          f"tt_radius={result['tt_region']['tt_radius']}")
    print("──────────────────────────────────────────────────────────────────")
    print(f"Saved → {args.output}")

    # Diagnostic figure: points + ellipse over the mask-std heatmap. The
    # background is computed from the dataset when a config is available;
    # otherwise the points/ellipse are drawn on a blank background.
    plot_dir = (os.path.join(data_dir, "kinematics") if data_dir
                else os.path.dirname(os.path.normpath(raw_dir)))
    plot_path = os.path.join(plot_dir, "tt_region.pdf")
    mean_std = load_mask_std_background(cfg, max_utts=args.max_utts) if cfg else None
    save_plot(all_pts, ellipse, center, r_capture, args.confidence,
              plot_path, mean_std=mean_std)
    print(f"Plot  → {plot_path}")


if __name__ == "__main__":
    main()