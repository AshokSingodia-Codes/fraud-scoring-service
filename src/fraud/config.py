from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict containing project configuration.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config
