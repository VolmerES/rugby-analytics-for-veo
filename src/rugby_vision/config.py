from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else DEFAULT_CONFIG
    if not source.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {source}")
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"La configuración debe ser un objeto YAML: {source}")
    for section in ("detection", "tracking", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Falta la sección '{section}' en {source}")
    confidence = config["detection"].get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("detection.confidence debe estar entre 0 y 1")
    return config
