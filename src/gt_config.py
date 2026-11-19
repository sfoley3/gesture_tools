# ABOUTME: Shared config loading for all gesture_tools scripts.
# ABOUTME: One JSON per dataset lives in configs/ and is chosen with --config.
"""Every script accepts ``--config NAME_OR_PATH``: a dataset name in ``configs/``
(``prompt``, ``longitudinal``, ``lss``) or a path to a JSON file.

Scripts read their config at import time, so ``load_config`` looks at the command
line itself instead of waiting for the script's own argument parser."""

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_HELP = (
    "Dataset config: a name in configs/ (prompt, longitudinal, lss) or a path "
    "to a JSON file. Default: $GESTURE_TOOLS_CONFIG, then config.json."
)


def resolve_config(value) -> Path:
    """Dataset name -> configs/<name>.json; anything else is taken as a path."""
    named = REPO / "configs" / f"{value}.json"
    return named if named.is_file() else Path(value)


def load_config(defaults: dict | None = None) -> dict:
    """Config dict from, in order: --config on the command line, the
    GESTURE_TOOLS_CONFIG environment variable, config.json at the repo root.
    Missing keys come from `defaults`. A config that was asked for by name or path
    must exist; the repo-root fallback is optional so imports never crash."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config")
    value = pre.parse_known_args()[0].config or os.environ.get("GESTURE_TOOLS_CONFIG")
    path = resolve_config(value) if value else REPO / "config.json"
    cfg = dict(defaults or {})
    if value or path.is_file():
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg
