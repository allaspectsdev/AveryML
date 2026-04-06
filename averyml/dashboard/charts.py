"""Plotly chart builders for the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def build_comparison_chart(results: list[dict], metric_keys: list[str] | None = None):
    """Grouped bar chart comparing pass@k across selected runs."""
    import plotly.graph_objects as go

    if not results:
        return _empty_figure("Select results to compare")

    if metric_keys is None:
        metric_keys = ["pass@1", "pass@5", "pass@10"]

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

    fig = go.Figure()
    for idx, r in enumerate(results):
        res = r.get("results", {})
        model = r.get("config", {}).get("model_id", "?").split("/")[-1]
        ts = r.get("timestamp", 0)
        label = f"{model} ({datetime.fromtimestamp(ts).strftime('%m/%d')})" if ts else model

        values = [res.get(k, 0) for k in metric_keys if isinstance(res.get(k), float)]
        keys = [k for k in metric_keys if isinstance(res.get(k), float)]

        fig.add_trace(go.Bar(
            name=label, x=keys, y=values,
            text=[f"{v:.1%}" for v in values], textposition="auto",
            marker_color=colors[idx % len(colors)],
        ))

    fig.update_layout(
        barmode="group", title="Pass@k Comparison",
        yaxis_title="Pass Rate", yaxis_tickformat=".0%",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_difficulty_chart(results: list[dict]):
    """Bar chart of pass@1 by difficulty for selected runs."""
    import plotly.graph_objects as go

    if not results:
        return _empty_figure("Select results to compare")

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    difficulties = ["easy", "medium", "hard"]

    fig = go.Figure()
    for idx, r in enumerate(results):
        res = r.get("results", {})
        model = r.get("config", {}).get("model_id", "?").split("/")[-1]
        ts = r.get("timestamp", 0)
        label = f"{model} ({datetime.fromtimestamp(ts).strftime('%m/%d')})" if ts else model

        vals, labels = [], []
        for d in difficulties:
            key = f"pass@1_{d}"
            if isinstance(res.get(key), float):
                vals.append(res[key])
                labels.append(d.capitalize())

        if vals:
            fig.add_trace(go.Bar(
                name=label, x=labels, y=vals,
                text=[f"{v:.1%}" for v in vals], textposition="auto",
                marker_color=colors[idx % len(colors)],
            ))

    fig.update_layout(
        barmode="group", title="pass@1 by Difficulty",
        yaxis_title="Pass Rate", yaxis_tickformat=".0%",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_temperature_heatmap(df: pd.DataFrame | None, metric: str = "pass@1"):
    """Interactive heatmap of metric over (T_train, T_eval) grid."""
    import plotly.graph_objects as go

    if df is None or df.empty or metric not in df.columns:
        return _empty_figure("No search results available")

    pivot = df.pivot_table(index="t_train", columns="t_eval", values=metric, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)
    text = pivot.map(lambda v: f"{v:.1%}" if pd.notna(v) else "")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        text=text.values, texttemplate="%{text}",
        colorscale="YlOrRd",
        colorbar=dict(title=metric, tickformat=".0%"),
        hovertemplate="T_eval=%{x}<br>T_train=%{y}<br>" + metric + "=%{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"{metric} over (T_train, T_eval) Grid",
        xaxis_title="T_eval", yaxis_title="T_train",
        template="plotly_white", height=500,
    )
    return fig


def build_teff_curve(df: pd.DataFrame | None, metric: str = "pass@1"):
    """Scatter + quadratic fit of metric vs T_eff."""
    import plotly.graph_objects as go

    if df is None or df.empty or metric not in df.columns or "t_eff" not in df.columns:
        return _empty_figure("No search results available"), {}

    mask = df[metric].notna()
    t_eff = df.loc[mask, "t_eff"].values
    values = df.loc[mask, metric].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_eff, y=values, mode="markers",
        marker=dict(size=10, color=df.loc[mask, "t_train"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="T_train")),
        hovertemplate="T_eff=%{x:.2f}<br>" + metric + "=%{y:.2%}<extra></extra>",
        name="Data",
    ))

    stats = {}
    if len(t_eff) >= 3:
        coeffs = np.polyfit(t_eff, values, 2)
        x_fit = np.linspace(t_eff.min(), t_eff.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        y_pred = np.polyval(coeffs, t_eff)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - values.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        optimal_teff = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0

        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name=f"Quadratic fit (R²={r_squared:.2f})",
        ))
        stats = {"r_squared": r_squared, "optimal_teff": optimal_teff,
                 "best_value": values.max(), "best_teff": t_eff[values.argmax()]}

    fig.update_layout(
        title=f"{metric} vs Effective Temperature",
        xaxis_title="T_eff = T_train x T_eval", yaxis_title=metric,
        yaxis_tickformat=".0%", template="plotly_white", height=450,
    )
    return fig, stats


def build_length_histogram(lengths: list[int]):
    """Histogram of response lengths for data explorer."""
    import plotly.graph_objects as go

    if not lengths:
        return _empty_figure("No data to plot")

    fig = go.Figure(data=go.Histogram(
        x=lengths, nbinsx=50,
        marker_color="#3b82f6", marker_line_color="#1e40af", marker_line_width=1,
    ))
    fig.update_layout(
        title="Response Length Distribution",
        xaxis_title="Characters", yaxis_title="Count",
        template="plotly_white", height=300,
    )
    return fig


def build_training_plots(log_history: list[dict]):
    """Build loss and LR plots from HF Trainer log history."""
    import plotly.graph_objects as go

    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    lr_steps = [e["step"] for e in log_history if "learning_rate" in e]
    lrs = [e["learning_rate"] for e in log_history if "learning_rate" in e]

    loss_fig = go.Figure()
    if steps:
        loss_fig.add_trace(go.Scatter(
            x=steps, y=losses, mode="lines",
            line=dict(color="#3b82f6", width=2), name="Loss",
        ))
        loss_fig.update_layout(
            title="Training Loss", xaxis_title="Step", yaxis_title="Loss",
            template="plotly_white", height=350,
        )
    else:
        loss_fig = _empty_figure("No loss data yet")

    lr_fig = go.Figure()
    if lr_steps:
        lr_fig.add_trace(go.Scatter(
            x=lr_steps, y=lrs, mode="lines",
            line=dict(color="#f59e0b", width=2), name="LR",
        ))
        lr_fig.update_layout(
            title="Learning Rate Schedule", xaxis_title="Step", yaxis_title="LR",
            template="plotly_white", height=350,
        )
    else:
        lr_fig = _empty_figure("No LR data yet")

    return loss_fig, lr_fig


def _empty_figure(message: str = "No data"):
    """Create a blank figure with a centered message."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(size=16, color="#94a3b8"),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        template="plotly_white", height=300,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
