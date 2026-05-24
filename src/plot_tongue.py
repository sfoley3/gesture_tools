# ABOUTME: Diagnostic script to plot tongue contours directly from numpy files.
# ABOUTME: Produces 3x3 grid plots (per speaker) with 2x2 facets (phone x position), raw, smoothed, and ssanova.
"""
Plot tongue contours from .npy files for /l/ and /r/ liquids.

Reads per-utterance contour arrays (T, 2, 100) from each speaker's
tongue_contours/ directory, extracts the midpoint frame using TextGrid
phone alignments, and produces three 3x3 panel PDFs:
  - liquids_raw_py.pdf      raw contours
  - liquids_smoothed_py.pdf Gaussian-smoothed contours (sigma=5)
  - liquids_ssanova_py.pdf  smoothing spline fit + 95% CI

Each outer panel = one speaker with a rectangular border and a bold title.
Inner 2x2 = phone (R top, L bottom) × position (onset left, coda right).
R/L labels appear outside the left border of every speaker panel.
Colors: onset = red (#e41a1c), coda = blue (#377eb8).
Y-axis is inverted to match pixel space.

Usage:
    python analysis/plot_tongue_contours.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d

# ssanova.py lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from ssanova import compute_all as ssanova_compute_all

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

DATA_DIR = Path(_cfg["data_dir"])
FIGS_DIR = DATA_DIR / "figs"

# ── Constants ─────────────────────────────────────────────────────────────────

FPS       = 99
SPK_ORDER = [f'spk{i}' for i in range(2, 11)]   # spk2–spk10

L_WORDS      = {'llama', 'leaf', 'loop', 'peel', 'pool', 'ball'}
R_WORDS      = {'reef', 'rob', 'roof', 'bar', 'fear'}
TARGET_WORDS = L_WORDS | R_WORDS
ONSET_WORDS  = {'llama', 'leaf', 'loop', 'reef', 'rob', 'roof'}

PHONES    = ('R', 'L')        # row order: R top, L bottom
POSITIONS = ('onset', 'coda') # column order: onset left, coda right

SIGMA      = 5
POS_COLORS = {'onset': '#e41a1c', 'coda': '#377eb8'}

# Per inner-cell (ph_idx, pos_idx): which spines face outward.
# With hspace=wspace=0 the four panels share edges, forming one outer rectangle.
BORDER_SPINES = {
    (0, 0): ('top', 'left'),
    (0, 1): ('top', 'right'),
    (1, 0): ('bottom', 'left'),
    (1, 1): ('bottom', 'right'),
}


# ── TextGrid parser ───────────────────────────────────────────────────────────

def parse_textgrid_tier(path, tier_name):
    """Return list of (xmin, xmax, label) for the named tier."""
    intervals = []
    with open(path, 'r') as f:
        lines = f.readlines()

    in_tier = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if f'name = "{tier_name}"' in line:
            in_tier = True
            i += 1
            continue
        if in_tier and line.startswith('item ['):
            break
        if in_tier and line.startswith('xmin'):
            xmin = float(line.split('=')[1].strip())
            xmax = float(lines[i + 1].strip().split('=')[1].strip())
            text = lines[i + 2].strip().split('=')[1].strip().strip('" ')
            if text:
                intervals.append((xmin, xmax, text))
            i += 3
            continue
        i += 1
    return intervals


# ── Data loading ──────────────────────────────────────────────────────────────

def load_liquid_contours():
    """
    Load midpoint-frame tongue contours for all lwr tokens across speakers.

    Returns a list of dicts with keys:
        spk, word, phone, position, contour_id, x (100,), y (100,)
    """
    records = []
    counts  = {}

    for spk in SPK_ORDER:
        cnt_dir = DATA_DIR / spk / 'tongue_contours'
        tg_dir  = DATA_DIR / spk / 'textgrids_split'

        if not cnt_dir.exists() or not tg_dir.exists():
            print(f'  [{spk}] missing tongue_contours or textgrids_split — skipping')
            continue

        npy_files = sorted(cnt_dir.glob('*lwr*.npy'))
        if not npy_files:
            print(f'  [{spk}] no lwr*.npy files found')
            continue

        spk_count = 0
        for npy_path in npy_files:
            fname    = npy_path.stem
            parts    = fname.split('_')
            if len(parts) < 3:
                continue
            word_rep = parts[2]
            word     = word_rep.rstrip('0123456789')

            if word not in TARGET_WORDS:
                continue

            phone    = 'L' if word in L_WORDS else 'R'
            position = 'onset' if word in ONSET_WORDS else 'coda'

            tg_path = tg_dir / f'{fname}.TextGrid'
            if not tg_path.exists():
                continue

            phones_tier     = parse_textgrid_tier(str(tg_path), 'phones')
            phone_intervals = [(xmn, xmx) for xmn, xmx, lbl in phones_tier
                               if lbl == phone]
            if not phone_intervals:
                continue

            xmin, xmax = phone_intervals[0]
            mid_sec    = (xmin + xmax) / 2.0

            arr = np.load(npy_path)   # (T, 2, 100)
            T   = arr.shape[0]
            mid_frame = max(0, min(int(round(mid_sec * FPS)), T - 1))

            x = arr[mid_frame, 0, :]  # (100,) x-coords
            y = arr[mid_frame, 1, :]  # (100,) y-coords

            if np.isnan(x).any() or np.isnan(y).any():
                continue

            records.append(dict(
                spk=spk, word=word, phone=phone, position=position,
                contour_id=fname, x=x, y=y,
            ))
            spk_count += 1
            counts[(spk, phone, position)] = counts.get((spk, phone, position), 0) + 1

        print(f'  [{spk}] {spk_count} contours loaded')

    print('\nContours per speaker × phone × position:')
    for (spk, ph, pos), n in sorted(counts.items()):
        print(f'  {spk}  {ph}  {pos:5s}  n={n}')

    return records


# ── Plotting ──────────────────────────────────────────────────────────────────

def _make_3x3(records, smooth, ssanova_fits=None):
    """
    3x3 figure; each outer cell = one speaker with a rectangular border.
    Inner 2x2 = phone (R/L) × position (onset/coda).

    smooth=False        raw contours
    smooth=True         Gaussian-smoothed contours
    ssanova_fits dict   ribbon + fitted line (from ssanova.compute_all)
    """
    data = defaultdict(list)
    for rec in records:
        data[(rec['spk'], rec['phone'], rec['position'])].append(rec)

    # Global limits → every speaker box identical size
    all_x = np.concatenate([r['x'] for r in records])
    all_y = np.concatenate([r['y'] for r in records])
    pad   = 2.0
    xlim  = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim  = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    fig = plt.figure(figsize=(14, 15))
    outer_gs = GridSpec(
        3, 3, figure=fig,
        hspace=0.18, wspace=0.15,
        left=0.07, right=0.97, top=0.94, bottom=0.05,
    )

    for cell_idx, spk in enumerate(SPK_ORDER):
        row = cell_idx // 3
        col = cell_idx  % 3
        inner_gs = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_gs[row, col],
            hspace=0.0, wspace=0.0,
        )

        for ph_idx, ph in enumerate(PHONES):
            for pos_idx, pos in enumerate(POSITIONS):
                ax = fig.add_subplot(inner_gs[ph_idx, pos_idx])

                if ssanova_fits is not None:
                    fit = ssanova_fits.get((spk, ph, pos))
                    if fit is not None:
                        ax.fill_between(fit.x_grid, fit.y_lo, fit.y_hi,
                                        color=POS_COLORS[pos], alpha=0.15)
                        ax.plot(fit.x_grid, fit.y_fit,
                                color=POS_COLORS[pos], lw=1.5)
                else:
                    for rec in data.get((spk, ph, pos), []):
                        x = gaussian_filter1d(rec['x'], sigma=SIGMA) if smooth else rec['x'].copy()
                        y = gaussian_filter1d(rec['y'], sigma=SIGMA) if smooth else rec['y'].copy()
                        ax.plot(x, y, color=POS_COLORS[pos], alpha=0.6, lw=0.5)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

                # Outer border: only the two outward-facing spines per cell
                for spine in ax.spines.values():
                    spine.set_visible(False)
                for side in BORDER_SPINES[(ph_idx, pos_idx)]:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(0.8)
                    ax.spines[side].set_color('black')

                # R/L label outside the left border — every speaker
                if pos_idx == 0 and col == 0:
                    ax.text(
                        -0.14, 0.5, ph,
                        transform=ax.transAxes,
                        fontsize=22, fontweight='bold',
                        ha='right', va='center',
                        clip_on=False,
                    )

        # Speaker title centered above the 2x2 block
        ss   = outer_gs[row, col]
        bbox = ss.get_position(fig)
        fig.text(
            (bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.005, spk,
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            transform=fig.transFigure,
        )

    # Shared legend at bottom
    handles = [
        mpatches.Patch(color='#e41a1c', label='onset'),
        mpatches.Patch(color='#377eb8', label='coda'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.005))

    return fig


def make_plots(records):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print('\nGenerating raw plot...')
    fig_raw = _make_3x3(records, smooth=False)
    fig_raw.savefig(FIGS_DIR / 'liquids_raw_py.pdf', bbox_inches='tight')
    plt.close(fig_raw)
    print(f'  Saved: {FIGS_DIR / "liquids_raw_py.pdf"}')

    print('Generating smoothed plot...')
    fig_sm = _make_3x3(records, smooth=True)
    fig_sm.savefig(FIGS_DIR / 'liquids_smoothed_py.pdf', bbox_inches='tight')
    plt.close(fig_sm)
    print(f'  Saved: {FIGS_DIR / "liquids_smoothed_py.pdf"}')

    print('\nFitting ssanova models (this takes ~30s)...')
    fits = ssanova_compute_all(records, sigma=SIGMA)
    print('Generating ssanova plot...')
    fig_ss = _make_3x3(records, smooth=True, ssanova_fits=fits)
    fig_ss.savefig(FIGS_DIR / 'liquids_ssanova_py.pdf', bbox_inches='tight')
    plt.close(fig_ss)
    print(f'  Saved: {FIGS_DIR / "liquids_ssanova_py.pdf"}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading contours...')
    records = load_liquid_contours()
    print(f'\nTotal contours: {len(records)}')

    if not records:
        print('No contours loaded — check data paths.')
    else:
        make_plots(records)

    print('\nDone.')
