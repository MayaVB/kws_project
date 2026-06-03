"""
config.py

Provides YAML configuration loading
with basic validation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a flat YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a top-level dictionary")

    return cfg