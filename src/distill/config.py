"""Typed access to configs/default.yaml. Keep flat and boring."""
from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any
import yaml


@dataclasses.dataclass
class Cfg:
    raw: dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        try:
            v = self.raw[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, key: str, default: Any = None) -> Any:
        v = self.raw.get(key, default)
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | Path = "configs/default.yaml") -> Cfg:
    with open(path) as f:
        return Cfg(yaml.safe_load(f))
