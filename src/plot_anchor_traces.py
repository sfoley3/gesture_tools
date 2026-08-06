#!/usr/bin/env python3
# ABOUTME: QA plot of raw-vs-smoothed VTD anchor trajectories (velum split fraction,
# ABOUTME: tongue-bottom x/y) to confirm temporal stabilization removes jitter without flattening real motion.
"""
plot_anchor_traces.py — sanity plot for the VTD anchor stabilization.

For a few utterances, traces per frame:
  * velum split fraction on the roof, from THREE sources so you can see both
    improvements at once:
      - bottom-edge center (the old `_velum_lower_center`, noisiest),
      - mask centroid (the new `_velum_centroid`, intrinsically stabler),
      - centroid + temporal smoothing (`stabilize`, what training will use);
  * tongue-bottom x and y: raw vs temporally smoothed.

Prints a jitter table (mean |Δ| between consecutive frames) so the reduction is
quantified, not just eyeballed. Reuses the extractor's own functions so the plot
matches exactly what `extract_vtd.py` feeds the grid.

Usage:
    python src/plot_anchor_traces.py [--spk N] [--n-utts 4] \
        [--median 5] [--sigma 2.5] [--seed 0]

Outputs: {base}/anchor_traces/{basename}_anchors.pdf
"""

import argparse
import importlib.util
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import extract_vtd as a module (reuse its config + tracing helpers).
_EV = Path(__file__).resolve().parent / "extract_vtd.py"
_spec = importlib.util.spec_from_file_location("extract_vtd", _EV)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _mad(x):
    """Mean absolute first difference (a simple jitter metric) over finite frames."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    d = np.abs(np.diff(x))
    return float(d.mean()) if len(d) else float("nan")


def _measure(regions, T, jaw_ref):
    """Per-frame raw anchors: velum fraction from bottom-edge and from centroid,
    plus tongue-bottom x/y."""
    f_edge = np.full(T, np.nan, np.float32)
    f_cent = np.full(T, np.nan, np.float32)
    tbx = np.full(T, np.nan, np.float32)
    tby = np.full(T, np.nan, np.float32)
    for t in range(T):
        roof, floor, _, reg_up = ev._frame_walls(regions, t, jaw_ref)
        if roof is not None and len(roof) >= 3:
            edge = ev._velum_lower_center(reg_up)
            if edge is not None:
                _, f_edge[t], _ = ev._project_to_polyline(roof, edge)
            cent = ev._velum_centroid(reg_up)
            if cent is not None:
                _, f_cent[t], _ = ev._project_to_polyline(roof, cent)
        if floor is not None and len(floor) >= 2:
            tbx[t], tby[t] = floor[-1]
    return f_edge, f_cent, tbx, tby


def _plot(basename, f_edge, f_cent, f_smooth, tbx, tbx_s, tby, tby_s, out_path):
    T = len(f_cent)
    fr = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(fr, f_edge, color="0.75", lw=1.0, label="bottom-edge (old)")
    ax.plot(fr, f_cent, color="tab:orange", lw=1.0, alpha=0.8, label="centroid (raw)")
    ax.plot(fr, f_smooth, color="tab:blue", lw=2.2, label="centroid + smoothed")
    f_median = np.nanmedian(f_cent)
    ax.axhline(f_median, color="tab:red", lw=2.0, ls="--", label=f"median (firm)={f_median:.3f}")
    ax.set_ylabel("velum split\n(roof arc frac)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(basename)

    for ax, raw, sm, lab in (
        (axes[1], tbx, tbx_s, "tongue-bottom x"),
        (axes[2], tby, tby_s, "tongue-bottom y"),
    ):
        ax.plot(fr, raw, color="0.7", lw=1.0, label="raw")
        ax.plot(fr, sm, color="tab:green", lw=2.2, label="smoothed")
        ax.set_ylabel(lab + "\n(px)")
        ax.legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("frame")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot raw-vs-smoothed VTD anchor traces.")
    p.add_argument("--spk", default=None, help="Speaker dir name (e.g. ID16) or number.")
    p.add_argument("--session", default=ev.SESSION, help="Session subdir (longitudinal).")
    p.add_argument("--n-utts", type=int, default=4, help="Number of utterances to plot.")
    p.add_argument("--median", type=int, default=int(ev.ANCHOR_SMOOTH.get("median", 5)))
    p.add_argument("--sigma", type=float, default=float(ev.ANCHOR_SMOOTH.get("sigma", 2.5)))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spk = args.spk
    if spk is not None:
        # accept a full dir name (ID16) or a bare number with the SPK_BASE prefix.
        name = spk if (ev.DATA_DIR / spk).is_dir() else f"{ev.SPK_BASE}{spk}"
        base = ev.DATA_DIR / name
    else:
        base = ev.DATA_DIR
    session = args.session or ""
    sess_base = base / session
    mask_dir = sess_base / "sam_seg" / "masks"
    mask_files = sorted(mask_dir.glob("*.npz"))
    if not mask_files:
        print(f"No mask files in {mask_dir}")
        return
    rng = random.Random(args.seed)
    chosen = rng.sample(mask_files, min(args.n_utts, len(mask_files)))
    out_dir = sess_base / "anchor_traces"

    print(f"anchors: median={args.median} sigma={args.sigma}  ->  {out_dir}")
    print(f"{'utterance':<28} {'velum edge':>10} {'velum cent':>10} "
          f"{'velum smth':>10} {'tb-y raw':>9} {'tb-y smth':>9}")
    for mp in chosen:
        regions, T = ev._load_regions(mp)
        if T == 0:
            continue
        jaw_ref = (
            ev._find_jaw_anchor(regions[ev.TONGUE_SUB], regions[ev.LOWER_LIP_SUB])
            if regions[ev.TONGUE_SUB] is not None and regions[ev.LOWER_LIP_SUB] is not None
            else None
        )
        f_edge, f_cent, tbx, tby = _measure(regions, T, jaw_ref)
        f_smooth = ev.stabilize(f_cent, args.median, args.sigma)
        tbx_s = ev.stabilize(tbx, args.median, args.sigma)
        tby_s = ev.stabilize(tby, args.median, args.sigma)
        _plot(mp.stem, f_edge, f_cent, f_smooth, tbx, tbx_s, tby, tby_s,
              out_dir / f"{mp.stem}_anchors.pdf")
        print(f"{mp.stem:<28} {_mad(f_edge):>10.4f} {_mad(f_cent):>10.4f} "
              f"{_mad(f_smooth):>10.4f} {_mad(tby):>9.3f} {_mad(tby_s):>9.3f}")
    print("\n(lower = less frame-to-frame jitter)")


if __name__ == "__main__":
    main()
