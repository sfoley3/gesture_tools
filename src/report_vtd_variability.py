#!/usr/bin/env python3
# ABOUTME: Reports VTD target variability across train, validation, and test splits.
"""
Create a machine-readable VTD variability report for speech inversion.

The report is intentionally text/JSON rather than plot-oriented so it can be
returned to an analyst or to Codex. It follows the split construction in
speech_inv/src/train.py and eval.py, including adult-only runs and optional LSS
or longitudinal training augmentation.

For each split and grid point it reports:

  pooled_sd                  variation across every valid target frame
  within_speaker_sd          pooled variation after removing each speaker mean
  between_speaker_sd_weighted variation caused by different speaker means
  speaker_mean_sd_unweighted  SD of speaker means, giving every speaker equal weight
  frame_diff_sd              frame-to-frame variation, sensitive to jitter/noise

The variance decomposition is population-weighted, so approximately:

  pooled_sd^2 = within_speaker_sd^2 + between_speaker_sd_weighted^2

The default target is ``norm`` because that is what the VTD inversion model
trains on. Use ``--value-kind pts`` to report raw pixel VTD instead. If
``lines/*.npz`` is available, the JSON also contains roof/floor arc-ratio
variability using the same decomposition.

Example
-------

    python src/report_vtd_variability.py \
        --config ../speech_inv/src/config.json \
        --vtd-dir-name auto \
        --out /tmp/vtd_variability_report

This writes ``/tmp/vtd_variability_report.json`` and
``/tmp/vtd_variability_report.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "speech_inv" / "src" / "config.json"


def read_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return json.load(f)


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open() as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def json_number(value):
    value = float(value)
    return round(value, 8) if math.isfinite(value) else None


def finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def frame_differences(x: np.ndarray) -> np.ndarray:
    """Differences only across adjacent finite frames; do not bridge gaps."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] < 2:
        return np.empty(0, dtype=float)
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    return np.diff(x)[ok]


def arc_fractions(points: np.ndarray) -> np.ndarray:
    """Cumulative arc fraction of saved ordered VTD endpoints, shape (T, L)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expected wall points with shape (T,L,2), got {points.shape}")
    out = np.full(points.shape[:2], np.nan, dtype=float)
    for t, row in enumerate(points):
        if not np.all(np.isfinite(row)):
            continue
        seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
        total = float(seg.sum())
        if total > 1e-8:
            out[t] = np.concatenate(([0.0], np.cumsum(seg))) / total
    return out


def has_value_files(vtd_dir: Path, value_kind: str) -> bool:
    return any((vtd_dir / value_kind).glob("*.npy"))


def resolve_vtd_dir(
    base: Path,
    requested: str,
    value_kind: str,
    preferred: str | None = None,
) -> Path | None:
    """Resolve a speaker/session VTD directory with norm-only compatibility."""
    if requested != "auto":
        path = Path(requested)
        if not path.is_absolute():
            path = base / path
        return path if has_value_files(path, value_kind) else None

    names = []
    if preferred:
        names.append(preferred)
    names.extend(["vtd"])
    names.extend(sorted(p.name for p in base.glob("vtd_*") if p.is_dir()))
    candidates = []
    for name in names:
        path = base / name
        if path not in candidates:
            candidates.append(path)
    usable = [p for p in candidates if has_value_files(p, value_kind)]
    if not usable:
        return None
    # Prefer the configured directory, then the current arc extraction.
    usable.sort(
        key=lambda p: (
            0 if preferred and p.name == preferred else 1,
            0 if p.name.startswith("vtd_arc_") else 1,
            p.name,
        )
    )
    return usable[0]


def discover_speakers(root: Path, session: str | None) -> list[str]:
    names = []
    if not root.is_dir():
        return names
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        base = child / session if session else child
        if base.is_dir():
            names.append(child.name)
    return names


def load_record(
    value_path: Path, lines_path: Path | None, expected_L: int | None
) -> dict:
    value = np.asarray(np.load(value_path), dtype=float)
    if value.ndim != 2:
        raise ValueError(f"Expected (T,L) target, got {value.shape}: {value_path}")
    if expected_L is not None and value.shape[1] != expected_L:
        raise ValueError(
            f"Grid count mismatch in {value_path}: got L={value.shape[1]}, expected {expected_L}"
        )
    result = {"value": value, "roof_arc": None, "floor_arc": None}
    if lines_path is None or not lines_path.is_file():
        return result
    try:
        with np.load(lines_path) as lines:
            roof = np.asarray(lines["roof"], dtype=float)
            floor = np.asarray(lines["floor"], dtype=float)
        if roof.shape[:2] == value.shape and floor.shape[:2] == value.shape:
            result["roof_arc"] = arc_fractions(roof)
            result["floor_arc"] = arc_fractions(floor)
    except (KeyError, ValueError):
        pass
    return result


def speaker_records(
    root: Path,
    speakers: list[str],
    session: str | None,
    vtd_dir_name: str,
    value_kind: str,
    preferred_vtd_dir: str | None,
    split: str,
    source: str,
    expected_L: int | None,
    speaker_subdir: str | None = None,
) -> tuple[list[dict], dict]:
    records = []
    info = {
        "source": source,
        "search_root": str(root / speaker_subdir) if speaker_subdir else str(root),
        "session": session,
        "requested_speakers": list(speakers),
        "missing_speakers": [],
        "missing_vtd_directories": [],
        "n_utterances": 0,
    }
    speaker_root = root / speaker_subdir if speaker_subdir else root
    for speaker in speakers:
        base = speaker_root / speaker / session if session else speaker_root / speaker
        if not base.is_dir():
            info["missing_speakers"].append(speaker)
            continue
        vtd_dir = resolve_vtd_dir(base, vtd_dir_name, value_kind, preferred_vtd_dir)
        if vtd_dir is None:
            info["missing_vtd_directories"].append(speaker)
            continue
        value_dir = vtd_dir / value_kind
        for value_path in sorted(value_dir.glob("*.npy")):
            stem = value_path.stem
            lines_path = vtd_dir / "lines" / f"{stem}.npz"
            record = load_record(value_path, lines_path, expected_L)
            records.append(
                {
                    "split": split,
                    "source": source,
                    "speaker": speaker,
                    "stem": stem,
                    "path": str(value_path),
                    **record,
                }
            )
        info["n_utterances"] += sum(1 for p in value_dir.glob("*.npy"))
    info["n_speakers_found"] = len({r["speaker"] for r in records})
    return records, info


def lss_records(
    root: Path,
    stems: list[str],
    vtd_dir_name: str,
    value_kind: str,
    preferred_vtd_dir: str | None,
    split: str,
    source: str,
    expected_L: int | None,
) -> tuple[list[dict], dict]:
    """Read LSS flat targets; LSS is one effective speaker for decomposition."""
    base = root
    vtd_dir = resolve_vtd_dir(base, vtd_dir_name, value_kind, preferred_vtd_dir)
    info = {
        "source": source,
        "search_root": str(root),
        "requested_stems": len(stems),
        "missing_stems": [],
        "n_utterances": 0,
        "n_speakers_found": 0,
    }
    if vtd_dir is None:
        info["missing_stems"] = list(stems)
        return [], info
    records = []
    for stem in stems:
        value_path = vtd_dir / value_kind / f"{stem}.npy"
        if not value_path.is_file():
            info["missing_stems"].append(stem)
            continue
        record = load_record(
            value_path, vtd_dir / "lines" / f"{stem}.npz", expected_L
        )
        records.append(
            {
                "split": split,
                "source": source,
                "speaker": "LSS",
                "stem": stem,
                "path": str(value_path),
                **record,
            }
        )
    info["n_utterances"] = len(records)
    info["n_speakers_found"] = 1 if records else 0
    return records, info


def all_speaker_records(
    root: Path,
    session: str | None,
    vtd_dir_name: str,
    value_kind: str,
    preferred_vtd_dir: str | None,
    split: str,
    source: str,
    expected_L: int | None,
    speaker_subdir: str | None = None,
) -> tuple[list[dict], dict]:
    speaker_root = root / speaker_subdir if speaker_subdir else root
    speakers = discover_speakers(speaker_root, session)
    records, info = speaker_records(
        root,
        speakers,
        session,
        vtd_dir_name,
        value_kind,
        preferred_vtd_dir,
        split,
        source,
        expected_L,
        speaker_subdir=speaker_subdir,
    )
    info["discovered_speakers"] = speakers
    return records, info


def _group_values(records: list[dict], key: str) -> dict[str, list[np.ndarray]]:
    groups = defaultdict(list)
    for record in records:
        values = record.get(key)
        if values is not None:
            groups[record["speaker"]].append(values)
    return dict(groups)


def summarize_groups(groups: dict[str, list[np.ndarray]]) -> dict:
    """Compute per-point total/within/between variability for speaker groups."""
    if not groups:
        return {
            "n_speakers": 0,
            "n_frames": 0,
            "n_utterances": 0,
            "points": [],
            "speakers": [],
            "overall": {},
        }
    L = next(iter(groups.values()))[0].shape[1]
    point_rows = []
    speaker_rows = []
    for speaker, arrays in sorted(groups.items()):
        concat = np.concatenate(arrays, axis=0)
        p_mean = np.full(L, np.nan)
        p_sd = np.full(L, np.nan)
        p_diff = np.full(L, np.nan)
        n_frames = np.zeros(L, dtype=int)
        for l in range(L):
            vals = finite(concat[:, l])
            diffs = np.concatenate(
                [frame_differences(a[:, l]) for a in arrays]
            )
            n_frames[l] = vals.size
            if vals.size:
                p_mean[l] = vals.mean()
                p_sd[l] = vals.std()
            if diffs.size:
                p_diff[l] = diffs.std()
        speaker_rows.append(
            {
                "speaker": speaker,
                "n_utterances": len(arrays),
                "n_frames": int(concat.shape[0]),
                "mean": json_number(np.nanmean(p_mean)),
                "mean_point_sd": json_number(np.nanmean(p_sd)),
                "max_point_sd": json_number(np.nanmax(p_sd)),
                "mean_frame_diff_sd": json_number(np.nanmean(p_diff)),
                "point_mean": [json_number(x) for x in p_mean],
                "point_sd": [json_number(x) for x in p_sd],
                "frame_diff_sd": [json_number(x) for x in p_diff],
                "n_frames_per_point": n_frames.tolist(),
            }
        )

    for l in range(L):
        speaker_data = []
        for speaker, arrays in sorted(groups.items()):
            vals = finite(np.concatenate([a[:, l] for a in arrays]))
            if vals.size:
                speaker_data.append((speaker, vals))
        if not speaker_data:
            point_rows.append({"point": l, "position": json_number(l / max(L - 1, 1))})
            continue
        all_vals = np.concatenate([vals for _, vals in speaker_data])
        grand_mean = float(all_vals.mean())
        within_ss = 0.0
        between_ss = 0.0
        means = []
        speaker_sd = []
        for _, vals in speaker_data:
            mean = float(vals.mean())
            means.append(mean)
            speaker_sd.append(float(vals.std()))
            within_ss += float(((vals - mean) ** 2).sum())
            between_ss += vals.size * (mean - grand_mean) ** 2
        total_n = all_vals.size
        point_rows.append(
            {
                "point": l,
                "position": json_number(l / max(L - 1, 1)),
                "n_frames": int(total_n),
                "n_speakers": len(speaker_data),
                "mean": json_number(grand_mean),
                "pooled_sd": json_number(all_vals.std()),
                "within_speaker_sd": json_number(math.sqrt(within_ss / total_n)),
                "between_speaker_sd_weighted": json_number(math.sqrt(between_ss / total_n)),
                "speaker_mean_sd_unweighted": json_number(np.std(means)),
                "mean_speaker_sd": json_number(np.mean(speaker_sd)),
            }
        )

    all_flat = [finite(a) for arrays in groups.values() for a in arrays]
    all_flat = np.concatenate(all_flat) if all_flat else np.empty(0)
    all_diffs = [frame_differences(a[:, l]) for arrays in groups.values() for a in arrays for l in range(L)]
    all_diffs = np.concatenate([d for d in all_diffs if d.size]) if any(d.size for d in all_diffs) else np.empty(0)
    overall = {
        "mean": json_number(all_flat.mean()) if all_flat.size else None,
        "pooled_sd": json_number(all_flat.std()) if all_flat.size else None,
        "frame_diff_sd": json_number(all_diffs.std()) if all_diffs.size else None,
        "mean_point_pooled_sd": json_number(np.nanmean([p.get("pooled_sd", np.nan) for p in point_rows])),
        "mean_point_within_speaker_sd": json_number(np.nanmean([p.get("within_speaker_sd", np.nan) for p in point_rows])),
        "mean_point_between_speaker_sd": json_number(np.nanmean([p.get("between_speaker_sd_weighted", np.nan) for p in point_rows])),
    }
    return {
        "n_speakers": len(groups),
        "n_frames": int(sum(a.shape[0] for arrays in groups.values() for a in arrays)),
        "n_utterances": int(sum(len(arrays) for arrays in groups.values())),
        "points": point_rows,
        "speakers": speaker_rows,
        "overall": overall,
    }


def summarize_split(records: list[dict], value_key: str) -> dict:
    groups = _group_values(records, value_key)
    summary = summarize_groups(groups)
    arc = {}
    if value_key == "value":
        for name in ("roof_arc", "floor_arc"):
            if any(record.get(name) is not None for record in records):
                arc[name.replace("_arc", "")] = summarize_groups(_group_values(records, name))
    summary["arc_ratio"] = arc
    summary["sources"] = dict(
        sorted(
            (source, sum(1 for r in records if r["source"] == source))
            for source in {r["source"] for r in records}
        )
    )
    return summary


def collect_report(cfg: dict, args) -> dict:
    value_kind = args.value_kind
    if value_kind == "auto":
        target_parts = Path(cfg.get("vtd_subdir", "vtd/norm")).parts
        value_kind = target_parts[-1] if target_parts else "norm"
    preferred = None
    target_parts = Path(cfg.get("vtd_subdir", "")).parts
    if target_parts and not Path(cfg.get("vtd_subdir", "")).is_absolute():
        preferred = target_parts[0]
    expected_L = args.n_gridlines or cfg.get("n_gridlines")
    dataset = cfg.get("dataset", "prompt").lower()
    data_root = Path(cfg["data_root"])
    longitudinal_root = Path(cfg.get("longitudinal_data_root", data_root))
    lss_root = Path(cfg.get("lss_data_root", data_root))
    sessions = cfg.get("longitudinal_sessions") or [cfg.get("session", "D1A")]
    session = args.session if args.session is not None else (sessions[0] if sessions else None)
    adult_only = bool(cfg.get("adult_only", False))
    collection = {}
    all_records = {}

    def add(name, records, info):
        all_records.setdefault(name, []).extend(records)
        collection.setdefault(name, []).append(info)

    if adult_only:
        for split in ("train", "val"):
            split_path = longitudinal_root / f"{split}.txt"
            speakers = read_split(split_path)
            records, info = speaker_records(
                longitudinal_root,
                speakers,
                session,
                args.vtd_dir_name,
                value_kind,
                preferred,
                split,
                "longitudinal",
                expected_L,
            )
            info["split_file"] = str(split_path)
            add(split, records, info)
        test_path = data_root / "adults.txt"
        speakers = read_split(test_path)
        records, info = speaker_records(
            data_root,
            speakers,
            None,
            args.vtd_dir_name,
            value_kind,
            preferred,
            "test",
            "prompt_adults",
            expected_L,
            speaker_subdir="mri",
        )
        info["split_file"] = str(test_path)
        add("test", records, info)
        if cfg.get("add_lss", False):
            for split in ("train",):
                for split_name in ("train.txt", "val.txt", "test.txt"):
                    path = lss_root / split_name
                    if not path.is_file():
                        continue
                    records, info = lss_records(
                        lss_root,
                        read_split(path),
                        args.vtd_dir_name,
                        value_kind,
                        preferred,
                        split,
                        "lss_all",
                        expected_L,
                    )
                    info["split_file"] = str(path)
                    add(split, records, info)
    elif dataset == "prompt":
        for split in ("train", "val", "test"):
            path = data_root / f"{split}.txt"
            speakers = read_split(path)
            records, info = speaker_records(
                data_root,
                speakers,
                None,
                args.vtd_dir_name,
                value_kind,
                preferred,
                split,
                "prompt",
                expected_L,
                speaker_subdir="mri",
            )
            info["split_file"] = str(path)
            add(split, records, info)
        if cfg.get("add_lss", False):
            for split_name in ("train.txt", "val.txt", "test.txt"):
                path = lss_root / split_name
                if path.is_file():
                    records, info = lss_records(
                        lss_root,
                        read_split(path),
                        args.vtd_dir_name,
                        value_kind,
                        preferred,
                        "train",
                        "lss_all",
                        expected_L,
                    )
                    info["split_file"] = str(path)
                    add("train", records, info)
        if cfg.get("add_longitudinal", False):
            records, info = all_speaker_records(
                longitudinal_root,
                session,
                args.vtd_dir_name,
                value_kind,
                preferred,
                "train",
                "longitudinal_all",
                expected_L,
            )
            add("train", records, info)
    else:
        for split in ("train", "val", "test"):
            path = data_root / f"{split}.txt"
            stems = read_split(path)
            records, info = lss_records(
                data_root,
                stems,
                args.vtd_dir_name,
                value_kind,
                preferred,
                split,
                "lss",
                expected_L,
            )
            info["split_file"] = str(path)
            add(split, records, info)

    report = {
        "metadata": {
            "config": str(args.config),
            "dataset": dataset,
            "adult_only": adult_only,
            "value_kind": value_kind,
            "vtd_dir_name": args.vtd_dir_name,
            "expected_gridlines": expected_L,
            "longitudinal_session": session,
            "definitions": {
                "pooled_sd": "SD across all valid target frames and points within a grid point",
                "within_speaker_sd": "pooled residual SD after subtracting each speaker's point mean",
                "between_speaker_sd_weighted": "frame-weighted SD of speaker point means",
                "frame_diff_sd": "SD of adjacent-frame differences, excluding missing-frame gaps",
                "arc_ratio": "cumulative fraction along the saved ordered VTD endpoints",
            },
        },
        "collection": collection,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        records = all_records.get(split, [])
        report["splits"][split] = summarize_split(records, "value")
    return report


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_report(report: dict) -> str:
    meta = report["metadata"]
    lines = [
        "# VTD variability report",
        "",
        f"- Dataset: `{meta['dataset']}`; adult_only={meta['adult_only']}",
        f"- Target: `{meta['value_kind']}`; expected gridlines={meta['expected_gridlines']}",
        f"- Config: `{meta['config']}`",
        "",
        "`within_speaker_sd` is the pooled residual variation after removing each speaker's mean. "
        "`between_speaker_sd_weighted` is variation in speaker means. "
        "`frame_diff_sd` is the noise-sensitive frame-to-frame statistic.",
        "",
        "## Set overview",
        "",
        "| Set | Utterances | Speakers | Frames | Pooled SD | Within-speaker SD | Between-speaker SD | Frame-diff SD | Sources |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "val", "test"):
        s = report["splits"].get(split, {})
        o = s.get("overall", {})
        lines.append(
            f"| {split} | {s.get('n_utterances', 0)} | {s.get('n_speakers', 0)} | "
            f"{s.get('n_frames', 0)} | {fmt(o.get('pooled_sd'))} | "
            f"{fmt(o.get('mean_point_within_speaker_sd'))} | "
            f"{fmt(o.get('mean_point_between_speaker_sd'))} | "
            f"{fmt(o.get('frame_diff_sd'))} | {', '.join(f'{k}: {v}' for k, v in s.get('sources', {}).items())} |"
        )

    for split in ("train", "val", "test"):
        s = report["splits"].get(split, {})
        lines += ["", f"## {split}: per speaker", "", "| Speaker | Utterances | Frames | Mean | Mean point SD | Max point SD | Mean frame-diff SD |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in s.get("speakers", []):
            lines.append(
                f"| {row['speaker']} | {row['n_utterances']} | {row['n_frames']} | {fmt(row.get('mean'))} | "
                f"{fmt(row.get('mean_point_sd'))} | {fmt(row.get('max_point_sd'))} | {fmt(row.get('mean_frame_diff_sd'))} |"
            )
        lines += ["", f"## {split}: per point", "", "| Point | Position | Mean | Pooled SD | Within-speaker SD | Between-speaker SD |", "|---:|---:|---:|---:|---:|---:|"]
        for row in s.get("points", []):
            lines.append(
                f"| {row['point']} | {fmt(row.get('position'))} | {fmt(row.get('mean'))} | {fmt(row.get('pooled_sd'))} | "
                f"{fmt(row.get('within_speaker_sd'))} | {fmt(row.get('between_speaker_sd_weighted'))} |"
            )
        arc = s.get("arc_ratio", {})
        if arc:
            lines += ["", f"### {split}: arc-ratio summary", "", "| Wall | Mean pooled SD | Mean within-speaker SD | Mean between-speaker SD |", "|---|---:|---:|---:|"]
            for wall in ("roof", "floor"):
                if wall in arc:
                    o = arc[wall].get("overall", {})
                    lines.append(
                        f"| {wall} | {fmt(o.get('mean_point_pooled_sd'))} | {fmt(o.get('mean_point_within_speaker_sd'))} | {fmt(o.get('mean_point_between_speaker_sd'))} |"
                    )
    lines += ["", "## Collection details", ""]
    for split, infos in report.get("collection", {}).items():
        for info in infos:
            missing = info.get("missing_speakers", []) + info.get("missing_vtd_directories", []) + info.get("missing_stems", [])
            status = f"missing={len(missing)}" if missing else "complete"
            lines.append(f"- `{split}` / `{info.get('source')}`: {info.get('n_utterances', 0)} utterances, {status}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=Path("vtd_variability_report"), help="Output prefix for .json and .md files.")
    ap.add_argument("--vtd-dir-name", default="auto", help="VTD directory below each speaker/session; default auto.")
    ap.add_argument("--value-kind", choices=["auto", "norm", "pts"], default="auto", help="Analyze model targets (norm) or raw VTD (pts).")
    ap.add_argument("--n-gridlines", type=int, default=None, help="Expected L; defaults to config n_gridlines.")
    ap.add_argument("--session", default=None, help="Override the first configured longitudinal session.")
    args = ap.parse_args()
    cfg = read_config(args.config)
    report = collect_report(cfg, args)
    json_path = args.out.with_suffix(".json")
    md_path = args.out.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(markdown_report(report))
    print(f"JSON report -> {json_path}")
    print(f"Markdown report -> {md_path}")
    for split in ("train", "val", "test"):
        s = report["splits"].get(split, {})
        print(
            f"{split:>5}: {s.get('n_utterances', 0):5d} utterances, "
            f"{s.get('n_speakers', 0):3d} speakers, {s.get('n_frames', 0):8d} frames"
        )
    if not any(report["splits"].get(split, {}).get("n_utterances", 0) for split in ("train", "val", "test")):
        print("No VTD targets were found. Collection diagnostics:")
        for split, infos in report.get("collection", {}).items():
            for info in infos:
                missing = (
                    info.get("missing_speakers", [])
                    + info.get("missing_vtd_directories", [])
                    + info.get("missing_stems", [])
                )
                print(
                    f"  {split}/{info.get('source')}: search_root={info.get('search_root')} "
                    f"session={info.get('session')} missing={len(missing)}"
                )


if __name__ == "__main__":
    main()
