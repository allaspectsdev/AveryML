"""Base configuration with YAML I/O and merge semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base for all AveryML configs. Provides YAML serialization and override merging."""

    model_config = {"extra": "forbid"}

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data or {})

    def merge(self, overrides: dict[str, Any]) -> Self:
        """Return a new config with overrides applied. Only non-None values are merged."""
        current = self.model_dump()
        for key, value in overrides.items():
            if value is not None:
                current[key] = value
        return self.__class__.model_validate(current)
