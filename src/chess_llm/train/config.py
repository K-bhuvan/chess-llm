from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def resolve_adapter(preferred: Path | None = None) -> Path | None:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(
        [
            Path("outputs/grpo"),
            Path("outputs/dpo"),
            Path("outputs/sft"),
        ]
    )
    for path in candidates:
        if (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists():
            return path
    return None


def attach_adapter(model, adapter_path: Path | None):
    if adapter_path is None or not adapter_path.exists():
        return model
    try:
        model.load_adapter(str(adapter_path))
    except Exception:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, str(adapter_path))
    return model
