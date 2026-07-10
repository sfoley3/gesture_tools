# Plan: Vocal Tract Distance (VTD) from SAM2 masks

A new `gesture_tools` script that measures the airway width across the whole
vocal tract along a Proctor/Kim‑style **semipolar grid**, reproducing the VTD
feature used in Shi et al. (2024) — but driven entirely by the SAM2 tissue
masks instead of raw MRI intensity.

---

## 1. What we are building

For each speaker × video the tool will:

1. Build one **fixed, speaker‑specific semipolar grid** that follows the natural
   bend of the vocal tract, from the **lips (anterior anchor)** to the
   **tongue‑root / pharyngeal‑wall region (posterior anchor)**, with an
   **adjustable number of evenly spaced grid lines** in between.
2. For every frame, intersect each grid line with the **upper VT boundary**
   (roof: palate + velum + pharyngeal wall) and the **lower VT boundary**
   (floor: lower lip + tongue), and record the Euclidean **VTD** on that line.
3. Save a per‑video `(T, L)` VTD array, an optional min‑max‑normalized version
   and per‑gridline histograms (Shi‑style), and a diagnostic overlay
   (grid lines in cyan, VTD segments in yellow — matching Shi Fig. 1b).

Because we want to compare grid‑line *l* across frames and tokens (the whole
point of the Shi histograms), the grid is computed **once per speaker** from the
time‑aggregated masks and then applied unchanged to every frame — this is the
"speaker‑specific grid lines" step in the paper, and matches Proctor's fixed
analysis grid.

---

## 2. Key simplification: every landmark is already segmented

Proctor 2010 spends most of its machinery on problems we **do not have**:

| Proctor problem | Their solution | Our situation |
|---|---|---|
| Airway not segmented | Intensity thresholding along grid lines (Eqs. 2–4) | Tissue is already segmented — boundaries are mask edges |
| Centerline unknown | Dijkstra over graph of intensity minima (Eq. 1) | Centerline = medial line between segmented roof & floor |
| Hard palate / teeth don't image | Manual palate reference, DFT head‑motion correction | `upper lip - palate` mask already contains the palate |
| Rear pharyngeal wall faint | Average tissue boundary across frames as reference | Pharyngeal‑wall mask is provided |
| Glottis/larynx reference | Segment the larynx | No larynx mask — posterior end is tongue‑bottom → closest pharyngeal‑wall point |
| Tongue noise | DCT low‑pass smoothing | Optional LOESS reuse; masks are already clean |

So grid construction becomes **fully deterministic from mask geometry** — no
intensity profiles, no graph search, no manual anchors. The four Proctor
landmarks and the lingual origin are all read straight off the masks:

- **Posterior end** → **bottom of the `tongue` mask** and its **closest point on
  the `pharyngeal wall`**. There is **no larynx/glottis mask**; this pair defines
  the posterior‑most grid line and the back end of the tract.
- **Highest point on the palate** → min‑y pixel of `upper lip - palate`.
- **Alveolar ridge** → existing `_find_alveolar_ridge()` (leftmost stable palate
  column from the time‑averaged mask).
- **Lips** → lip‑aperture midpoint from `compute_lip_aperture()`.
- **Lingual origin** → point equidistant from the palate roof and the rear
  pharyngeal wall, near the time‑averaged tongue centroid (Proctor's
  definition), computed directly from the aggregate masks.

---

## 3. Inputs & conventions (from the existing code)

- **Masks** live in `{data_dir}/[spk/]sam_seg/masks/*.npz`, shape `(T, 104, 104)`
  bool, matched by case‑insensitive substring (`_find_mask_key`). Regions we use:
  - `upper lip - palate` (roof: hard palate + upper lip) — key substring `"upper lip"`
  - `velum` (soft palate)
  - `pharyngeal wall` — **new region; exact NPZ key to confirm** (auto‑detect via
    substring `"pharyn"`, configurable)
  - `tongue`
  - `lower lip - jaw` — key substring `"lower lip"`
  - `larynx` (glottis reference)
- **Coordinate frame**: face‑left midsagittal. Front of mouth = **low x (left)**;
  back/pharynx = **high x (right)**; roof = **low y (top)**; floor = **high y
  (bottom)**. (Confirmed in `get_tongue_contours.py` and `_find_alveolar_ridge`.)
- **Config** (`config.json`): reuse `data_dir`, `n_diagnostic`, `spk_base`,
  `video_dir`, `smooth`, `loess_span`; single‑ vs multi‑speaker handled exactly
  like the other scripts.

---

## 4. Method

### 4.1 Boundary contours (reuse existing tracing)

Per frame, build two airway‑facing polylines, oriented front→back:

- **Roof (upper boundary)** = anterior palate arc ⊕ velum arc ⊕ pharyngeal‑wall
  arc, joined into one continuous curve. Reuse `_bottom_arc_ordered()` and
  `_join_front_back_arcs()` from `plot_pts_contour.py` (they already trace the
  bottom/airway‑facing edge of `upper lip - palate` and `velum` and stitch
  them); extend the join to include the pharyngeal‑wall mask as the posterior
  segment.
- **Floor (lower boundary)** = upper tongue surface ⊕ inner lower‑lip edge.
  Reuse `extract_upper_contour()` + `resample_contour()` from
  `get_tongue_contours.py` for the tongue surface, and the lower‑lip inner edge
  logic from `_draw_lower_lip_jaw()` in `plot_pts_contour.py`.

These are the air–tissue boundaries Shi measures between; we get them for free
because the tissue is segmented.

### 4.2 Semipolar grid construction (Proctor §2.2, adapted)

Build the grid **once** from time‑averaged masks so it is fixed per speaker.
Four contiguous sections, following the tract bend:

1. **Labial (anterior)** — vertical lines over the region anterior to the teeth,
   extending through the lips. Anterior‑most line is anchored at the
   **lip‑aperture** point (the user's requested anterior start).
2. **Anterior‑oral fan** — radial lines from a second origin above the incisors
   (from the palate apex / alveolar‑ridge landmarks) through the anterior oral
   cavity.
3. **Mid‑oral / palatal fan** — equi‑spaced radial lines projected from the
   **lingual origin**, spanning from the alveolar ridge back through the mid
   pharynx. This is the fan that makes the grid hug the palatal bend.
4. **Pharyngeal (posterior)** — parallel (horizontal) lines at regular intervals
   from the **lingual‑origin level down to the posterior end**. The posterior‑most
   line is pinned to the **bottom of the tongue → closest pharyngeal‑wall point**
   (the user's requested posterior end; no larynx/glottis is used).

Each grid line is a directed segment from a point on the roof toward the floor.
Grid‑line orientation is what makes VTD "tract‑normal" and gives the natural
bend; the fan origins are derived from the segmented palate/pharynx rather than
hand‑placed.

**Adjustable density.** Proctor uses fixed spacings (5–8 mm linear, 4–8° radial).
We expose a single **`--n-gridlines N`** (total lines, or intermediate‑lines
count) that is distributed by arc length along the roof centerline between the
anterior (lip) and posterior (pharyngeal) anchors, so "n evenly spaced points in
between" is satisfied while each line keeps its section‑appropriate orientation
(vertical / radial / horizontal). A `--grid-spacing` alternative can be offered
later.

### 4.3 VTD measurement (per frame, per grid line)

For grid line *l*:

1. Find its intersection with the **roof** polyline → point `U_l = (xu, yu)`.
2. Find its intersection with the **floor** polyline → point `L_l = (xl, yl)`.
3. `VTD[t, l] = ||U_l − L_l||₂` (pixel Euclidean distance, as in Shi §3.2).
4. If a grid line misses a boundary (e.g. closed constriction where roof/floor
   overlap), clamp to 0 (maximum constriction) and flag it.

Intersections use segment/polyline crossing (shapely or a small numpy
line‑intersection helper — no new heavy deps; numpy is enough).

### 4.4 Normalization & histograms (Shi §5.1)

- **Per‑gridline global min‑max** across all frames/tokens of a speaker →
  `D̃[t,l] ∈ [0,1]`, 0 = max constriction, 1 = max opening (Shi Eq. 3).
- Aggregate `D̃` over time into a `(L, B)` histogram, default **B = 20** bins,
  for the token/phoneme‑level analysis Shi uses (optional, flag‑gated).

---

## 5. Outputs (mirrors `extract_mask_kinematics.py` layout)

```
{data_dir}/[spk/]vtd/
  pts/{basename}.npy            # (T, L) raw VTD in pixels
  norm/{basename}.npy           # (T, L) min-max normalized (optional)
  hist/{basename}.npy           # (L, B) histogram (optional)
  grid/{spk}_grid.json          # fixed grid line endpoints + landmarks (reuse)
  raw/{basename}.json           # per-frame U_l / L_l points for QA
  diagnostic/{basename}_vtd.mp4 # cyan grid + yellow VTD segments (Shi Fig 1b)
```

`grid.json` stores the speaker‑specific grid (landmarks + each line's roof/floor
anchor and orientation) so it is auditable and reused across all that speaker's
videos.

---

## 6. New files & API

**`src/extract_vtd.py`** — same skeleton as `extract_mask_kinematics.py`:

- `build_landmarks(masks_aggregate) -> dict` — glottis, palate apex, alveolar
  ridge, lips, lingual origin (all from masks).
- `build_semipolar_grid(landmarks, roof_arc, n_gridlines) -> GridLines`
  (fixed, speaker‑specific).
- `trace_roof(masks_t) / trace_floor(masks_t) -> polyline` (wrap existing
  tracing helpers).
- `compute_vtd(grid, roof_t, floor_t) -> (vtd_l, U_pts, L_pts)`.
- `extract_vtd(mask_path, video_path, grid) -> (vtd_array, fps, pts, mask_data)`.
- `write_diagnostic_video(...)` — reuse the mask‑overlay + `_scale_point`
  pattern; draw grid lines (cyan) and per‑line VTD segments (yellow).
- `process_speaker(spk)` / `main()` with `--spk`, `--n-gridlines`,
  `--normalize`, `--histogram`, `--bins` — matching the other CLIs.

**`pyproject.toml`** — add `extract-vtd = "extract_vtd:main"` to
`[project.scripts]` and `extract_vtd` to `py-modules`. Only new dependency
consideration: `shapely` for robust line/polyline intersection (optional; a
numpy fallback avoids it).

Reuse, don't duplicate: factor the shared arc‑tracing helpers
(`_bottom_arc_ordered`, `_join_front_back_arcs`, `extract_upper_contour`,
`resample_contour`, `_find_alveolar_ridge`, `_find_mask_key`, `_scale_point`)
into a small `src/mask_geometry.py` imported by the existing scripts and the new
one, so there is a single source of truth.

---

## 7. Edge cases

- **Missing pharyngeal‑wall mask** in a given NPZ → fall back to Proctor's
  time‑averaged mean pharyngeal contour as the posterior roof reference; warn.
- **Closed constriction** (roof/floor cross) → VTD = 0, flagged, not NaN.
- **Grid line with no intersection** (boundary gap) → bridge with the
  front/back arc join logic already in `_join_front_back_arcs`; else forward/
  back‑fill like the other extractors.
- **Retroflexed tongue tip / velic port open** → the upper‑contour splitter in
  `get_tongue_contours.py` already keeps the oral‑cavity‑facing surface; verify
  on a /r/ token.
- **Frame count / FPS** — default 50, read from video if present, exactly as the
  existing scripts.

---

## 8. Verification

- **Unit**: synthetic two‑arc "tube" mask with known constant gap → VTD must
  equal the gap on every line; a linearly tapering tube → linear VTD ramp.
- **Grid invariance**: assert the grid is identical across frames of a video
  (fixed‑grid guarantee) and that grid‑line count == `--n-gridlines`.
- **Visual**: render the diagnostic on a few VCV/bVt tokens and confirm cyan
  lines are tract‑normal through the palatal bend and yellow VTD segments span
  roof→floor (compare against Shi Fig. 1b).
- **Sanity vs. existing kinematics**: at the tongue‑tip and lip‑aperture grid
  lines, VTD should track `tt_dist` and `lip_aperture` from
  `extract_mask_kinematics.py` (correlation check).
- **Cross‑frame histograms**: reproduce a Shi‑style `(L,B)` histogram for an
  /IY/ vs /EY/ token and confirm the velar‑region (posterior grid lines)
  difference the paper reports.

---

## 9. Resolved decisions (confirmed)

1. **Pharyngeal‑wall mask** is named `"pharyngeal wall"`; auto‑detected via the
   `"pharyn"` substring. `larynx` marks the glottis.
2. **`n` = total number of grid lines**, distributed evenly by arc length along
   the roof from the lip anchor to the glottis.
3. **Fan / lingual origin fully mask‑derived** (no hand tuning).
4. **Normalization = per‑speaker** global min‑max per grid line.
5. **Diagnostics**: (a) one static PNG per speaker — a random frame with masks
   shown translucent, solid roof/floor edge lines, cyan grid lines and yellow
   VTD points; (b) an adjustable number of per‑speaker overlay **videos**
   (`--n-videos`) rendering masks + edges + grid + VTD points for every frame.

## 10. Status — implemented

`src/extract_vtd.py` implements the above and is wired into `pyproject.toml`
(`extract-vtd`) and `config.json` (`n_gridlines`, `n_bins`). It was validated
end‑to‑end on synthetic curved‑tube masks (ray/polyline intersection unit test,
fixed‑grid line count, VTD ≈ known airway gap, per‑speaker normalization in
[0,1], and both diagnostic outputs). Run with:

```
conda run -n myenv python src/extract_vtd.py [--spk N ...] \
    --n-gridlines 40 --n-videos 5 --bins 20
```

Outputs land in `{data_dir}/[spk/]vtd/` as described in §5.

Note: shared arc‑tracing helpers are currently duplicated inside
`extract_vtd.py` (rather than refactored into a shared `mask_geometry.py`) to
avoid touching the working scripts before running on real data; the refactor in
§6 remains a clean follow‑up.
