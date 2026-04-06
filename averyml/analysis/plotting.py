"""Plotting utilities for SSD analysis and experiment visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SSDPlotter:
    """Plotting utilities for SSD temperature sweeps, distribution analysis, and metrics."""

    @staticmethod
    def temperature_sweep_heatmap(
        df: pd.DataFrame,
        metric: str = "pass@1",
        output_path: Path | None = None,
    ):
        """Heatmap of a metric over the (T_train, T_eval) grid. Reproduces Figure 3b."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        pivot = df.pivot_table(index="t_train", columns="t_eval", values=metric, aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax)
        ax.set_xlabel("T_eval")
        ax.set_ylabel("T_train")
        ax.set_title(f"{metric} over (T_train, T_eval) grid")

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def t_eff_curve(
        df: pd.DataFrame,
        metric: str = "pass@1",
        output_path: Path | None = None,
    ):
        """Plot metric vs T_eff with quadratic fit. Shows R^2."""
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        t_eff = df["t_eff"].values
        values = df[metric].values

        # Quadratic fit
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        mask = ~np.isnan(values)
        if mask.sum() < 3:
            return None

        popt, _ = curve_fit(quadratic, t_eff[mask], values[mask])
        ss_res = np.sum((values[mask] - quadratic(t_eff[mask], *popt)) ** 2)
        ss_tot = np.sum((values[mask] - values[mask].mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(t_eff, values, alpha=0.7, label="Data")
        x_fit = np.linspace(t_eff.min(), t_eff.max(), 100)
        ax.plot(x_fit, quadratic(x_fit, *popt), "r-", label=f"Quadratic fit (R^2={r_squared:.2f})")
        ax.set_xlabel("T_eff = T_train * T_eval")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Effective Temperature")
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def difficulty_breakdown(
        metrics: dict[str, Any],
        output_path: Path | None = None,
    ):
        """Bar chart of pass@k by difficulty (easy/medium/hard)."""
        import matplotlib.pyplot as plt

        difficulties = ["easy", "medium", "hard"]
        k_values = []

        # Find available k values
        for key in metrics:
            if key.startswith("pass@") and "_" not in key and isinstance(metrics[key], float):
                k_values.append(int(key.split("@")[1]))

        if not k_values:
            return None

        fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5))
        if len(k_values) == 1:
            axes = [axes]

        for ax, k in zip(axes, k_values):
            values = []
            labels = []
            for diff in difficulties:
                key = f"pass@{k}_{diff}"
                if key in metrics:
                    values.append(metrics[key])
                    labels.append(diff.capitalize())

            if values:
                bars = ax.bar(labels, values, color=["#2ecc71", "#f39c12", "#e74c3c"])
                ax.set_ylabel(f"pass@{k}")
                ax.set_title(f"pass@{k} by Difficulty")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.1%}", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig
