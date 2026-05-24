#!/usr/bin/env python3
"""Print per-speaker mean ± std for MCI and NINFL."""

import json
from pathlib import Path
import numpy as np

# ── Load config ─────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

DATA_DIR  = Path(_cfg["data_dir"])
SPK_BASE  = _cfg.get("spk_base", "")


def _discover_speakers() -> list:
    if not SPK_BASE:
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith(SPK_BASE)
    )


if SPK_BASE:
    speakers = _discover_speakers()
else:
    speakers = [None]  # single-speaker mode

print(f"{'speaker':<8}  {'MCI mean':>10} {'MCI std':>10}  {'NINFL mean':>12} {'NINFL std':>11}")
print("-" * 60)

for spk in speakers:
    base = DATA_DIR / spk if spk else DATA_DIR
    spk_label = spk if spk else DATA_DIR.name

    for metric_label, subdir in [("mci", "mci"), ("ninfl", "ninfl")]:
        d = base / subdir
        files = sorted(d.glob("*.npy")) if d.exists() else []
        vals = np.concatenate([np.load(f) for f in files]) if files else np.array([])
        vals = vals[~np.isnan(vals)]
        if metric_label == "mci":
            mci_mean, mci_std = (vals.mean(), vals.std()) if len(vals) else (float("nan"), float("nan"))
        else:
            ninfl_mean, ninfl_std = (vals.mean(), vals.std()) if len(vals) else (float("nan"), float("nan"))

    print(f"{spk_label:<8}  {mci_mean:10.4f} {mci_std:10.4f}  {ninfl_mean:12.4f} {ninfl_std:11.4f}")
