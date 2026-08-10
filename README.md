# gesture_tools
Analysis tools for working with rtMRI-derived video features for speech production research.

## VTD variability diagnostics

`src/plot_vtd_variability.py` makes two per-speaker boxplots from the saved VTD
targets, writing them and a long-form CSV into each VTD `diagnostic/` folder:

- `vtd_point_variability.pdf`: temporal variability of raw VTD at each point.
- `vtd_arc_ratio_variability.pdf`: variability of each endpoint's relative arc
  position along the roof and floor.
- `vtd_variability.csv`: the per-utterance values used by both plots.

The script auto-detects both `vtd/` and current `vtd_arc_minmax/`-style output
directories and reads `data_dir`/`session` from `config.json`:

```text
python src/plot_vtd_variability.py
python src/plot_vtd_variability.py --data-root /path/to/data --session D1A \
    --vtd-dir-name vtd_arc_minmax --spk ID16 ID17
```

The default point statistic is temporal standard deviation in pixels. Use
`--point-metric diff_std` to emphasize frame-to-frame jitter/noise, or
`--point-metric cv` for scale-normalized variability.

## VTD variability report

`src/report_vtd_variability.py` writes a machine-readable JSON report and a
compact Markdown report for train, validation, and test. It follows the
`speech_inv` split logic, including `adult_only`, PROMPT `adults.txt` testing,
and optional LSS/longitudinal training augmentation. It reports pooled,
within-speaker, between-speaker, and frame-difference variability per VTD point,
plus roof/floor arc-ratio variability when `lines/*.npz` is available.

```text
python src/report_vtd_variability.py \
    --config ../speech_inv/src/config.json \
    --out /tmp/vtd_variability_report
```

The default target is `norm`, matching the model input target. Use
`--value-kind pts` for raw-pixel VTD.
