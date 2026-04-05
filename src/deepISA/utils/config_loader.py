"""YAML configuration loader for motif_filter."""
import os
from pathlib import Path
from typing import Any

import yaml


def _find_config() -> Path:
    """Locate config file. Searches in order: env var > ./config/ > src/deepISA/../config/."""
    # 1. Environment variable
    env_path = os.environ.get("MOTIF_FILTER_CONFIG", "")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Relative to current working directory
    cwd_config = Path("config/default_config.yaml")
    if cwd_config.exists():
        return cwd_config

    # 3. Relative to this file's location
    this_dir = Path(__file__).resolve().parents[1]  # utils → deepISA → src
    src_config = this_dir.parent.parent / "config" / "default_config.yaml"
    if src_config.exists():
        return src_config

    raise FileNotFoundError(
        "config/default_config.yaml not found. "
        "Set MOTIF_FILTER_CONFIG env var or run from project root."
    )


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML file. If None, auto-detects location.

    Returns
    -------
    dict
        Configuration as nested dict.
    """
    if config_path is None:
        config_path = _find_config()
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to the config file's directory
    cfg_dir = config_path.resolve().parent
    cfg = _resolve_relative_paths(cfg, cfg_dir)

    return cfg


def _resolve_relative_paths(cfg: dict, base: Path) -> dict:
    """Resolve relative path values in a config dict, in-place."""
    path_keys = {"genome_dir", "model", "regions", "motifs", "jaspar", "output_dir"}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key in path_keys:
            if key in values and isinstance(values[key], str):
                p = Path(values[key])
                if not p.is_absolute():
                    values[key] = str(base / p)
    return cfg


def get_param(cfg: dict, *keys: str, default: Any = None) -> Any:
    """
    Get a nested configuration value.

    Example:
        device = get_param(cfg, "runtime", "device", default="cuda")
    """
    val = cfg
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return default
    return val
