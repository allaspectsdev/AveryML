"""Temperature grid construction and T_eff analysis.

The paper shows performance is well-governed by T_eff = T_train * T_eval,
with a quadratic peak near T_eff ~ 1.2 (R^2=0.75, Section 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from averyml.config.search import SearchConfig
from averyml.config.synthesis import DecodingConfig


@dataclass
class TemperaturePoint:
    """A single point in the (T_train, T_eval) grid."""

    t_train: float
    t_eval: float
    rho_train: DecodingConfig
    rho_eval: DecodingConfig

    @property
    def t_eff(self) -> float:
        """Effective temperature: T_eff = T_train * T_eval."""
        return self.t_train * self.t_eval

    def __repr__(self) -> str:
        return f"TemperaturePoint(T_train={self.t_train}, T_eval={self.t_eval}, T_eff={self.t_eff:.2f})"


def build_grid(config: SearchConfig) -> list[TemperaturePoint]:
    """Build the full Cartesian product of temperature configurations.

    If config.truncation_configs is provided, each truncation config is also
    swept (3D grid). Otherwise, default truncation is used (2D grid over temperatures).
    """
    truncation_configs = config.truncation_configs or [DecodingConfig()]

    points = []
    for base_trunc in truncation_configs:
        for t_train in config.t_train_values:
            for t_eval in config.t_eval_values:
                rho_train = base_trunc.model_copy(update={"temperature": t_train})
                rho_eval = base_trunc.model_copy(update={"temperature": t_eval})
                points.append(TemperaturePoint(
                    t_train=t_train,
                    t_eval=t_eval,
                    rho_train=rho_train,
                    rho_eval=rho_eval,
                ))

    return points


def filter_diagonal_band(
    points: list[TemperaturePoint],
    t_eff_min: float = 0.8,
    t_eff_max: float = 1.6,
) -> list[TemperaturePoint]:
    """Filter to the diagonal band where T_eff is in the productive range.

    Paper shows R^2=0.75 with quadratic peak near T_eff ~ 1.2.
    The sweet spot is roughly T_eff in [0.8, 1.6].
    """
    return [p for p in points if t_eff_min <= p.t_eff <= t_eff_max]
