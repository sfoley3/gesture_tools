#!/usr/bin/env python3
"""Print per-speaker mean ± std for MCI and NINFL."""

from pathlib import Path
import numpy as np

DATA_DIR     = Path("/data1/span_data/prompt/data/mri")
ALL_SPEAKERS = ["spk2", "spk3", "spk4", "spk5", "spk6", "spk7", "spk8", "spk9", "spk10"]

print(f"{'speaker':<8}  {'MCI mean':>10} {'MCI std':>10}  {'NINFL mean':>12} {'NINFL std':>11}")
print("-" * 60)

for spk in ALL_SPEAKERS:
    for label, subdir in [("mci", "mci"), ("ninfl", "ninfl")]:
        d = DATA_DIR / spk / subdir
        files = sorted(d.glob(f"{spk}_*.npy"))
        vals = np.concatenate([np.load(f) for f in files]) if files else np.array([])
        vals = vals[~np.isnan(vals)]
        if label == "mci":
            mci_mean, mci_std = (vals.mean(), vals.std()) if len(vals) else (float("nan"), float("nan"))
        else:
            ninfl_mean, ninfl_std = (vals.mean(), vals.std()) if len(vals) else (float("nan"), float("nan"))

    print(f"{spk:<8}  {mci_mean:10.4f} {mci_std:10.4f}  {ninfl_mean:12.4f} {ninfl_std:11.4f}")
