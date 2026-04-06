"""Optional experiment tracking (W&B). No-ops gracefully when disabled."""

from __future__ import annotations

from typing import Any


class ExperimentTracker:
    """Optional W&B integration. No-ops if project is None or wandb not installed."""

    def __init__(self, project: str | None = None, run_name: str | None = None, config: dict | None = None):
        self._run = None
        if project is not None:
            try:
                import wandb

                self._run = wandb.init(project=project, name=run_name, config=config or {})
            except ImportError:
                pass

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            import wandb

            wandb.finish()
            self._run = None

    @property
    def enabled(self) -> bool:
        return self._run is not None
