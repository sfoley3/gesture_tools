#!/usr/bin/env python3
# ABOUTME: Plots per-speaker temporal VTD variability and wall-relative arc-ratio variability.
"""
Plot variability in extracted VTD targets, one speaker at a time.

The extractor saves two complementary arrays:

  pts/{stem}.npy       raw VTD values, shape (T, L)
  lines/{stem}.npz      roof/floor endpoints, shape (T, L, 2)

For every utterance this script computes a temporal spread at each grid point,
then makes a boxplot across utterances for that speaker. The first plot uses
raw VTD values and answers "how much does this measured distance vary?" The
second reconstructs the cumulative arc fraction of each endpoint along the
saved roof and floor grid points and answers "does this point move around along
the wall, relative to the rest of the tract?"

The default point metric is temporal standard deviation. Use --point-metric
diff_std when the specific question is frame-to-frame jitter/noise rather than
the total articulatory range. The arc-ratio plot always reports the temporal
standard deviation of the normalized arc fraction.

Examples
--------
Current gesture_tools output, with data_dir/session read from config.json:

    python src/plot_vtd_variability.py

Explicit dataset root and extraction directory:

    python src/plot_vtd_variability.py \
        --data-root /data1/span_data/longitudinal \
        --session D1A --vtd-dir-name vtd_arc_minmax --spk ID16 ID17

Outputs are written to each speaker's vtd*/diagnostic directory:

    vtd_point_variability.pdf
    vtd_arc_ratio_variability.pdf
    vtd_variability.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def _has_vtd_files(path: Path) -> bool:
    return (path / "pts").is_dir() and any((path / "pts").glob("*.npy"))


def resolve_vtd_dir(base: Path, requested: str | None) -> Path | None:
    """Find a VTD output directory below one speaker/session directory."""
    if requested and requested != "auto":
        path = Path(requested)
        if not path.is_absolute():
            path = base / path
        return path if _has_vtd_files(path) else None

    candidates = [base / "vtd"]
    candidates.extend(sorted(p for p in base.glob("vtd_*") if p.is_dir()))
    usable = [p for p in candidates if _has_vtd_files(p)]
    if not usable:
        return None
    # Prefer the current arc output when several historical extractions exist.
    usable.sort(key=lambda p: (0 if p.name.startswith("vtd_arc_") else 1, p.name))
    return usable[0]


def speaker_bases(
    data_root: Path, speakers: list[str] | None, session: str | None
) -> list[tuple[str, Path]]:
    """Return (speaker label, speaker/session directory) pairs."""
    # Also allow --data-root to point directly at a single speaker or session.
    if _has_vtd_files(data_root / "vtd") or any(data_root.glob("vtd_*/pts/*.npy")):
        return [(data_root.name, data_root)]

    if speakers:
        names = speakers
    else:
        names = sorted(p.name for p in data_root.iterdir() if p.is_dir())

    out = []
    for name in names:
        root = data_root / name
        if not root.is_dir():
            print(f"Warning: speaker directory not found: {root}")
            continue
        base = root / session if session else root
        if base.is_dir():
            out.append((name, base))
    return out


def finite_values(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def temporal_metric(x: np.ndarray, metric: str) -> float:
    """Compute one temporal variability statistic from a 1-D sequence."""
    raw = np.asarray(x, dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size < 2:
        return float("nan")
    if metric == "std":
        return float(np.std(finite))
    if metric == "cv":
        mean = float(np.mean(finite))
        return float(np.std(finite) / abs(mean)) if abs(mean) > 1e-8 else float("nan")
    if metric == "diff_std":
        adjacent = np.isfinite(raw[:-1]) & np.isfinite(raw[1:])
        diffs = np.diff(raw)[adjacent]
        return float(np.std(diffs)) if diffs.size else float("nan")
    raise ValueError(f"Unknown temporal metric: {metric}")


def arc_fractions(points: np.ndarray) -> np.ndarray:
    """Return each saved endpoint's cumulative arc fraction, shape (T, L).

    The extractor does not persist the dense wall polyline, so the saved VTD
    endpoints are used as an ordered polyline. This is the relevant
    correspondence diagnostic: it tests whether grid points preserve their
    relative position along each measured wall from frame to frame.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expected (T, L, 2) wall points, got {points.shape}")
    out = np.full(points.shape[:2], np.nan, dtype=float)
    for t, row in enumerate(points):
        if not np.all(np.isfinite(row)):
            continue
        seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
        total = float(seg.sum())
        if total > 1e-8:
            out[t] = np.concatenate(([0.0], np.cumsum(seg))) / total
    return out


def _boxplot(ax, values: list[np.ndarray], positions: np.ndarray, color: str) -> None:
    usable = [i for i, v in enumerate(values) if finite_values(v).size]
    if not usable:
        ax.text(0.5, 0.5, "No finite measurements", ha="center", va="center")
        return
    data = [finite_values(values[i]) for i in usable]
    bp = ax.boxplot(
        data,
        positions=positions[usable],
        widths=0.65 / max(len(values) - 1, 1),
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")


def style_position_axis(ax) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0\n(Lips)", "0.5\n(Velum)", "1\n(Pharynx)"])
    ax.set_xlabel("Normalized VTD grid position")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)


def plot_point_variability(
    values: list[np.ndarray], speaker: str, metric: str, out_path: Path
) -> None:
    L = len(values)
    pos = np.arange(L) / max(L - 1, 1)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    _boxplot(ax, values, pos, "#4c78a8")
    style_position_axis(ax)
    labels = {
        "std": "Temporal SD",
        "cv": "Coefficient of variation",
        "diff_std": "Frame-difference SD",
    }
    unit = "pixels" if metric in ("std", "diff_std") else "unitless"
    ax.set_ylabel(f"{labels[metric]} ({unit})")
    ax.set_title(f"{speaker}: per-point VTD variability ({labels[metric].lower()})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_arc_variability(
    roof_values: list[np.ndarray],
    floor_values: list[np.ndarray],
    speaker: str,
    out_path: Path,
) -> None:
    L = max(len(roof_values), len(floor_values))
    pos = np.arange(L) / max(L - 1, 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, values, label, color in (
        (axes[0], roof_values, "Roof", "#59a14f"),
        (axes[1], floor_values, "Floor", "#e15759"),
    ):
        _boxplot(ax, values, pos, color)
        ax.set_ylabel(f"SD of {label.lower()} arc fraction")
        ax.set_ylim(bottom=0.0)
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    style_position_axis(axes[1])
    fig.suptitle(f"{speaker}: relative grid-point movement along the wall", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_meta(vtd_dir: Path) -> dict:
    path = vtd_dir / "grid_meta.json"
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def analyze_speaker(speaker: str, vtd_dir: Path, point_metric: str) -> bool:
    pts_dir = vtd_dir / "pts"
    lines_dir = vtd_dir / "lines"
    diagnostic_dir = vtd_dir / "diagnostic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    pts_files = sorted(pts_dir.glob("*.npy"))
    if not pts_files:
        print(f"  No VTD point files in {pts_dir}")
        return False

    first = np.load(pts_files[0], mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"Expected (T, L) VTD in {pts_files[0]}, got {first.shape}")
    L = first.shape[1]
    point_values = [[] for _ in range(L)]
    roof_values = [[] for _ in range(L)]
    floor_values = [[] for _ in range(L)]
    csv_rows = []
    n_lines = 0

    for pts_path in pts_files:
        pts = np.asarray(np.load(pts_path), dtype=float)
        if pts.ndim != 2 or pts.shape[1] != L:
            print(f"  Warning: skipping shape-mismatched file {pts_path.name}: {pts.shape}")
            continue
        stem = pts_path.stem
        lines_path = lines_dir / f"{stem}.npz"
        roof_arc = floor_arc = None
        if lines_path.is_file():
            with np.load(lines_path) as lines:
                roof = np.asarray(lines["roof"], dtype=float)
                floor = np.asarray(lines["floor"], dtype=float)
            if roof.shape[:2] == pts.shape and floor.shape[:2] == pts.shape:
                roof_arc = arc_fractions(roof)
                floor_arc = arc_fractions(floor)
            else:
                print(f"  Warning: line/VTD shape mismatch for {stem}; arc ratios skipped")

        row = {"speaker": speaker, "stem": stem, "frames": int(pts.shape[0])}
        for l in range(L):
            p_metric = temporal_metric(pts[:, l], point_metric)
            point_values[l].append(p_metric)
            row[f"point_{l:03d}_{point_metric}"] = p_metric
            if roof_arc is not None and floor_arc is not None:
                r_metric = temporal_metric(roof_arc[:, l], "std")
                f_metric = temporal_metric(floor_arc[:, l], "std")
                roof_values[l].append(r_metric)
                floor_values[l].append(f_metric)
                row[f"roof_arc_{l:03d}_std"] = r_metric
                row[f"floor_arc_{l:03d}_std"] = f_metric
            else:
                row[f"roof_arc_{l:03d}_std"] = float("nan")
                row[f"floor_arc_{l:03d}_std"] = float("nan")
        csv_rows.append(row)
        n_lines += 1

    if not csv_rows:
        return False

    plot_point_variability(
        point_values,
        speaker,
        point_metric,
        diagnostic_dir / "vtd_point_variability.pdf",
    )
    if any(finite_values(v).size for v in roof_values + floor_values):
        plot_arc_variability(
            roof_values,
            floor_values,
            speaker,
            diagnostic_dir / "vtd_arc_ratio_variability.pdf",
        )
    else:
        print("  Warning: no usable lines/*.npz files; arc-ratio plot skipped")

    fields = list(csv_rows[0])
    with (diagnostic_dir / "vtd_variability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    meta = load_meta(vtd_dir)
    anchors = meta.get("anchor_indices")
    anchor_note = f", anchors={anchors}" if anchors else ""
    print(
        f"  {speaker}: {n_lines} utterances, L={L}{anchor_note}\n"
        f"    point variability -> {diagnostic_dir / 'vtd_point_variability.pdf'}\n"
        f"    arc variability   -> {diagnostic_dir / 'vtd_arc_ratio_variability.pdf'}"
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root containing speaker directories; defaults to config data_dir.",
    )
    ap.add_argument(
        "--spk", nargs="+", default=None, help="Speaker directory names; default is all speakers."
    )
    ap.add_argument(
        "--session",
        default=None,
        help="Optional speaker subdirectory such as D1A; defaults to config session.",
    )
    ap.add_argument(
        "--vtd-dir-name",
        default="auto",
        help="VTD directory below each speaker/session (default: auto; accepts vtd or vtd_arc_minmax).",
    )
    ap.add_argument(
        "--point-metric",
        choices=["std", "cv", "diff_std"],
        default="std",
        help="Point variability: temporal SD, coefficient of variation, or frame-difference SD.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg.get("data_dir", "."))
    session = args.session if args.session is not None else cfg.get("session") or None
    bases = speaker_bases(data_root, args.spk, session)
    if not bases:
        raise FileNotFoundError(f"No speaker directories found below {data_root}")

    processed = 0
    for speaker, base in bases:
        vtd_dir = resolve_vtd_dir(base, args.vtd_dir_name)
        if vtd_dir is None:
            print(f"{speaker}: no VTD output found below {base}")
            continue
        if analyze_speaker(speaker, vtd_dir, args.point_metric):
            processed += 1
    if not processed:
        raise FileNotFoundError(
            "No speakers were processed. Check --data-root, --session, and --vtd-dir-name."
        )


if __name__ == "__main__":
    main()
