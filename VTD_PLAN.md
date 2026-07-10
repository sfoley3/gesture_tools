# Vocal Tract Distance (VTD) from SAM2 masks — `src/extract_vtd.py`

Measures the airway width across the whole vocal tract by tracing **two walls**
from the SAM2 tissue masks and measuring the distance between them along `L`
grid lines. Deliberately simple: **no lingual origin, no semipolar / Proctor
grid, no larynx.** Just two lines and the distances between them.

## Masks

Five segmented regions (case-insensitive substring match):
`upper lip - palate`, `lower lip - jaw`, `tongue`, `velum`, `pharyngeal wall`.
There is **no larynx/glottis mask**.

Face-left convention: front of mouth = low x (left), back/pharynx = high x
(right), roof = low y (top), floor = high y (bottom).

## The two lines (per frame)

**Roof (upper wall)** — one continuous line, front → back:

1. bottom (airway-facing) edge of `upper lip - palate` (lips → hard palate)
2. straight bridge → bottom edge of `velum`
3. straight bridge from the **velum's bottom-right point to its closest point on
   the pharyngeal wall**
4. pharyngeal-wall (airway-facing / left) edge from that junction **down to the
   bottom** — the wall is *not* traced upward past the junction, and its bottom
   horizontal curl is trimmed.

The velum is included and joined gracefully to both the palate and the wall with
straight bridges, so there are no loop-arounds — one direct line from the lips to
the near-bottom of the pharyngeal wall.

**Floor (lower wall)** — one continuous line, front → back:

1. top (airway-facing) edge of `lower lip - jaw`
2. straight bridge to the tongue's front (left-most) point
3. tongue **upper surface** via the existing contour method
   (`extract_upper_contour`: split the tongue contour at the jaw junction and
   the root, keep the upper path) down to the tongue root.

## Mask smoothing

Because staircase pixelation inflates distances, each mask is anti-alias
smoothed before tracing: keep the largest component, upsample `--upscale`× with
cubic interpolation, Gaussian blur (`--pre-sigma`), threshold at 0.5. Edges are
traced on the upscaled mask and divided back to original resolution; the
assembled line is lightly Gaussian-smoothed along its path (`--sigma-path`).

## VTD

Both walls are arc-length resampled to `L = --n-gridlines` corresponding points
(front → back); `VTD[t, l] = ‖roof_l − floor_l‖`. Because both lines follow the
tract, connecting equal-arc-length points makes the grid follow the natural
bend. The anterior-most line is the lip aperture; the **posterior-most line
connects the tongue root/bottom to the pharyngeal wall** — the requested
posterior endpoint. Per-speaker global min-max per grid line gives the
normalized `[0,1]` VTD (0 = max constriction), aggregated into `(L, bins)`
histograms (Shi 2024 style).

## Outputs — `{data_dir}/[spk/]vtd/`

```
pts/{basename}.npy    (T, L)      raw VTD in pixels
norm/{basename}.npy   (T, L)      per-speaker min-max normalized
hist/{basename}.npy   (L, bins)   per-gridline histogram of normalized VTD
lines/{basename}.npz  roof,floor  (T, L, 2) resampled wall points (QA)
diagnostic/{spk}_frame.png        one MRI frame + masks + lines + VTD points
diagnostic/{basename}_vtd.mp4     per-frame MRI overlay (--n-videos videos)
```

Both diagnostics render **over the MRI frame** with translucent masks, the roof
line (green), floor line (red), grid lines (cyan) and VTD points (yellow). If a
video is missing they fall back to a blank canvas.

## CLI

```
conda run -n myenv python src/extract_vtd.py [--spk N ...] \
    --n-gridlines 40 --n-videos 5 --bins 20 \
    --upscale 8 --pre-sigma 1.5 --sigma-path 1.5
```

Config keys mirror the other scripts (`data_dir`, `spk_base`, `video_dir`,
`n_diagnostic`) plus `n_gridlines`, `n_bins`, `upscale`, `pre_sigma`,
`sigma_path`. Registered as `extract-vtd` in `pyproject.toml`.

## Validation

Verified on synthetic masks: parallel-line VTD is constant and exact; the roof
reaches the pharyngeal wall and the floor reaches the tongue root; VTD has `L`
finite lines with no gaps; per-speaker normalization is in `[0,1]`; and the
diagnostic image/video render the two lines, grid and points over the frame.

## Follow-ups

- Shared arc-tracing helpers are duplicated inside `extract_vtd.py` to avoid
  disturbing the other scripts; factoring them into a `mask_geometry.py` is a
  clean follow-up.
- Correspondence is by equal arc length. If tract-normal pairing is preferred in
  curved regions, a midline-perpendicular variant can be added behind a flag.
