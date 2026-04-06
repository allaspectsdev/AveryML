"""Simple registry pattern for backends, benchmarks, and prompt sources."""

from __future__ import annotations

from typing import Any


class Registry:
    """Name -> class mapping. Avoids import-time dependencies on heavy packages."""

    def __init__(self, kind: str = "component"):
        self._kind = kind
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a class under a name."""

        def wrapper(cls: type) -> type:
            self._registry[name] = cls
            return cls

        return wrapper

    def get(self, name: str) -> type:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {available}")
        return self._registry[name]

    def list_names(self) -> list[str]:
        return sorted(self._registry.keys())


# Global registries
synthesis_backend_registry = Registry("synthesis backend")
training_backend_registry = Registry("training backend")
benchmark_registry = Registry("benchmark")
prompt_source_registry = Registry("prompt source")
