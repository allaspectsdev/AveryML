"""AveryML web dashboard powered by Gradio.

Launch with: averyml dashboard
Requires: pip install averyml[dashboard]
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import yaml

import averyml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------


class DashboardState:
    """Holds directory paths and shared state for the dashboard."""

    def __init__(self, results_dir: str, configs_dir: str, search_dir: str):
        self.results_dir = Path(results_dir)
        self.configs_dir = Path(configs_dir)
        self.search_dir = Path(search_dir)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_all_results(state: DashboardState) -> list[dict[str, Any]]:
    """Load all result JSON files with full data."""
    results = []
    if not state.results_dir.exists():
        return results
    for json_file in sorted(state.results_dir.rglob("results_*.json"), reverse=True):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["_path"] = str(json_file)
            results.append(data)
        except Exception:
            continue
    return results


def results_to_table(results: list[dict]) -> pd.DataFrame:
    """Convert loaded results into a display table."""
    if not results:
        return pd.DataFrame(columns=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10", "Path"])
    rows = []
    for r in results:
        cfg = r.get("config", {})
        res = r.get("results", {})
        ts = r.get("timestamp", 0)
        rows.append({
            "Model": cfg.get("model_id", "?"),
            "Benchmark": cfg.get("benchmark", "?"),
            "Date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?",
            "pass@1": f"{res.get('pass@1', 0):.1%}" if isinstance(res.get("pass@1"), float) else "-",
            "pass@5": f"{res.get('pass@5', 0):.1%}" if isinstance(res.get("pass@5"), float) else "-",
            "pass@10": f"{res.get('pass@10', 0):.1%}" if isinstance(res.get("pass@10"), float) else "-",
            "Path": r.get("_path", ""),
        })
    return pd.DataFrame(rows)


def load_search_results(state: DashboardState) -> pd.DataFrame | None:
    """Load grid search results DataFrame."""
    results_file = state.search_dir / "search_results.json"
    if not results_file.exists():
        return None
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        rows = data.get("results", [])
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def list_configs(state: DashboardState) -> dict[str, list[str]]:
    """List all YAML configs organized by category."""
    configs: dict[str, list[str]] = {}
    if not state.configs_dir.exists():
        return configs
    for yaml_file in sorted(state.configs_dir.rglob("*.yaml")):
        rel = yaml_file.relative_to(state.configs_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "other"
        configs.setdefault(category, []).append(str(rel))
    return configs


def get_config_class(category: str):
    """Return the Pydantic config class for a category."""
    from averyml.config import (
        EvaluationConfig,
        ExperimentConfig,
        SearchConfig,
        SynthesisConfig,
        TrainingConfig,
    )

    mapping = {
        "synthesis": SynthesisConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "experiments": ExperimentConfig,
        "search": SearchConfig,
    }
    return mapping.get(category)


def validate_config(yaml_text: str, category: str) -> str:
    """Validate YAML text against the appropriate Pydantic model."""
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return f"YAML parse error: {e}"

    cls = get_config_class(category)
    if cls is None:
        return "Valid YAML (no schema validation for this category)"

    try:
        cls.model_validate(data or {})
        return f"Valid {cls.__name__}"
    except Exception as e:
        return f"Validation error: {e}"


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------


def build_comparison_chart(results: list[dict], metric_keys: list[str] | None = None):
    """Grouped bar chart comparing pass@k across selected runs."""
    import plotly.graph_objects as go

    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No results selected", showarrow=False, font=dict(size=18))
        return fig

    if metric_keys is None:
        metric_keys = ["pass@1", "pass@5", "pass@10"]

    fig = go.Figure()
    for r in results:
        res = r.get("results", {})
        model = r.get("config", {}).get("model_id", "?").split("/")[-1]
        ts = r.get("timestamp", 0)
        label = f"{model} ({datetime.fromtimestamp(ts).strftime('%m/%d')})" if ts else model

        values = [res.get(k, 0) for k in metric_keys if isinstance(res.get(k), float)]
        keys = [k for k in metric_keys if isinstance(res.get(k), float)]

        fig.add_trace(go.Bar(name=label, x=keys, y=values, text=[f"{v:.1%}" for v in values], textposition="auto"))

    fig.update_layout(
        barmode="group",
        title="Pass@k Comparison",
        yaxis_title="Pass Rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=400,
    )
    return fig


def build_difficulty_chart(results: list[dict]):
    """Bar chart of pass@1 by difficulty for selected runs."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No results selected", showarrow=False, font=dict(size=18))
        return fig

    difficulties = ["easy", "medium", "hard"]
    colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}

    fig = go.Figure()
    for r in results:
        res = r.get("results", {})
        model = r.get("config", {}).get("model_id", "?").split("/")[-1]
        ts = r.get("timestamp", 0)
        label = f"{model} ({datetime.fromtimestamp(ts).strftime('%m/%d')})" if ts else model

        vals = []
        labels = []
        for d in difficulties:
            key = f"pass@1_{d}"
            if isinstance(res.get(key), float):
                vals.append(res[key])
                labels.append(d.capitalize())

        if vals:
            fig.add_trace(go.Bar(
                name=label, x=labels, y=vals,
                text=[f"{v:.1%}" for v in vals], textposition="auto",
            ))

    fig.update_layout(
        barmode="group",
        title="pass@1 by Difficulty",
        yaxis_title="Pass Rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=400,
    )
    return fig


def build_temperature_heatmap(df: pd.DataFrame, metric: str = "pass@1"):
    """Interactive heatmap of metric over (T_train, T_eval) grid."""
    import plotly.graph_objects as go

    if df is None or df.empty or metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No search results available", showarrow=False, font=dict(size=18))
        return fig

    pivot = df.pivot_table(index="t_train", columns="t_eval", values=metric, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    text = pivot.map(lambda v: f"{v:.1%}" if pd.notna(v) else "")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        text=text.values,
        texttemplate="%{text}",
        colorscale="YlOrRd",
        colorbar=dict(title=metric, tickformat=".0%"),
        hovertemplate="T_eval=%{x}<br>T_train=%{y}<br>" + metric + "=%{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"{metric} over (T_train, T_eval) Grid",
        xaxis_title="T_eval",
        yaxis_title="T_train",
        template="plotly_white",
        height=500,
    )
    return fig


def build_teff_curve(df: pd.DataFrame, metric: str = "pass@1"):
    """Scatter + quadratic fit of metric vs T_eff."""
    import plotly.graph_objects as go

    if df is None or df.empty or metric not in df.columns or "t_eff" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No search results available", showarrow=False, font=dict(size=18))
        return fig, {}

    mask = df[metric].notna()
    t_eff = df.loc[mask, "t_eff"].values
    values = df.loc[mask, metric].values

    fig = go.Figure()

    # Scatter colored by T_train
    fig.add_trace(go.Scatter(
        x=t_eff, y=values, mode="markers",
        marker=dict(size=10, color=df.loc[mask, "t_train"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="T_train")),
        hovertemplate="T_eff=%{x:.2f}<br>" + metric + "=%{y:.2%}<extra></extra>",
        name="Data",
    ))

    # Quadratic fit
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
            line=dict(color="red", width=2, dash="dash"),
            name=f"Quadratic fit (R²={r_squared:.2f})",
        ))

        stats = {
            "r_squared": r_squared,
            "optimal_teff": optimal_teff,
            "best_value": values.max(),
            "best_teff": t_eff[values.argmax()],
        }

    fig.update_layout(
        title=f"{metric} vs Effective Temperature (T_eff = T_train × T_eval)",
        xaxis_title="T_eff",
        yaxis_title=metric,
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=450,
    )
    return fig, stats


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class JobRunner:
    """Manages subprocess execution of pipeline steps."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self.command: str = ""
        self.start_time: float = 0

    def launch(self, cmd: list[str]) -> str:
        if self.process is not None and self.process.poll() is None:
            return "A job is already running. Wait for it to finish."

        self.log_file = Path(tempfile.mktemp(suffix=".log"))
        self.command = " ".join(cmd)
        self.start_time = time.time()

        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
            )
        return f"Launched: {self.command} (PID {self.process.pid})"

    def get_logs(self) -> str:
        if self.log_file is None or not self.log_file.exists():
            return ""
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def get_status(self) -> str:
        if self.process is None:
            return "No jobs running"
        rc = self.process.poll()
        elapsed = time.time() - self.start_time
        if rc is None:
            return f"Running: {self.command} ({elapsed:.0f}s elapsed)"
        elif rc == 0:
            return f"Completed: {self.command} ({elapsed:.0f}s, exit 0)"
        else:
            return f"Failed: {self.command} ({elapsed:.0f}s, exit {rc})"


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def build_home_tab(state: DashboardState, runner: JobRunner):
    """Tab 1: Dashboard overview."""
    import gradio as gr

    gr.Markdown(f"""
# AveryML Dashboard
**v{averyml.__version__}** — Simple Self-Distillation pipeline for LLM code generation

*Make your LLM better at code by feeding it its own homework — no answers required.*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Recent Results")
            results_table = gr.Dataframe(
                value=results_to_table(load_all_results(state)),
                headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10", "Path"],
                interactive=False,
                wrap=True,
            )
        with gr.Column(scale=1):
            gr.Markdown("### Status")
            job_status = gr.Markdown(runner.get_status())
            search_df = load_search_results(state)
            if search_df is not None and not search_df.empty:
                gr.Markdown(f"**Grid search:** {len(search_df)} cells completed")
            else:
                gr.Markdown("**Grid search:** No results yet")

            configs = list_configs(state)
            total = sum(len(v) for v in configs.values())
            gr.Markdown(f"**Configs:** {total} YAML files across {len(configs)} categories")

    refresh_btn = gr.Button("Refresh", size="sm")

    def refresh():
        return results_to_table(load_all_results(state)), runner.get_status()

    refresh_btn.click(fn=refresh, outputs=[results_table, job_status])


def build_pipeline_tab(state: DashboardState, runner: JobRunner):
    """Tab 2: Launch pipeline steps."""
    import gradio as gr

    step_to_configs = {
        "Synthesize": "synthesis",
        "Train": "training",
        "Evaluate": "evaluation",
        "Full Pipeline": "experiments",
    }

    with gr.Row():
        with gr.Column(scale=1):
            step = gr.Radio(
                choices=["Synthesize", "Train", "Evaluate", "Full Pipeline"],
                value="Evaluate",
                label="Pipeline Step",
            )

            all_configs = list_configs(state)
            config_choices = all_configs.get("evaluation", [])
            config_picker = gr.Dropdown(choices=config_choices, label="Config File", interactive=True)

            with gr.Accordion("Overrides", open=False):
                override_model = gr.Textbox(label="Model ID (override)", value="", placeholder="e.g. Qwen/Qwen3-4B-Instruct-2507")
                override_temp = gr.Slider(0.0, 3.0, step=0.1, label="Temperature (override)", value=0)
                override_temp_info = gr.Markdown("*Set to 0 to use config default*")

            launch_btn = gr.Button("Launch", variant="primary")

        with gr.Column(scale=2):
            config_preview = gr.Code(label="Config Preview", language="yaml", interactive=False)
            log_output = gr.Textbox(label="Logs", lines=20, max_lines=40, interactive=False)
            status_text = gr.Markdown("")

    def update_configs(selected_step):
        cat = step_to_configs.get(selected_step, "evaluation")
        choices = all_configs.get(cat, [])
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def show_preview(config_file):
        if not config_file:
            return ""
        path = state.configs_dir / config_file
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "File not found"

    def launch_job(selected_step, config_file, model_override, temp_override):
        if not config_file:
            return "No config selected", ""

        step_cmd_map = {
            "Synthesize": "synthesize",
            "Train": "train",
            "Evaluate": "evaluate",
            "Full Pipeline": "run-pipeline",
        }
        cmd = ["averyml", step_cmd_map[selected_step], "--config", str(state.configs_dir / config_file)]

        if model_override:
            cmd.extend(["--model-id", model_override])
        if temp_override and temp_override > 0:
            cmd.extend(["--temperature", str(temp_override)])

        msg = runner.launch(cmd)
        return msg, ""

    def poll_logs():
        return runner.get_logs(), runner.get_status()

    step.change(fn=update_configs, inputs=[step], outputs=[config_picker])
    config_picker.change(fn=show_preview, inputs=[config_picker], outputs=[config_preview])
    launch_btn.click(fn=launch_job, inputs=[step, config_picker, override_model, override_temp], outputs=[status_text, log_output])

    poll_btn = gr.Button("Refresh Logs", size="sm")
    poll_btn.click(fn=poll_logs, outputs=[log_output, status_text])


def build_results_tab(state: DashboardState):
    """Tab 3: Browse and compare evaluation results."""
    import gradio as gr

    gr.Markdown("### Results Explorer\nSelect rows below, then click **Compare** to visualize.")

    all_results = load_all_results(state)
    table_df = results_to_table(all_results)

    results_table = gr.Dataframe(
        value=table_df,
        headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10", "Path"],
        interactive=False,
        wrap=True,
    )

    with gr.Row():
        select_indices = gr.Textbox(
            label="Row indices to compare (comma-separated, 0-indexed)",
            placeholder="e.g. 0,1,2",
            value="0" if len(all_results) > 0 else "",
        )
        compare_btn = gr.Button("Compare", variant="primary")
        refresh_btn = gr.Button("Refresh", size="sm")

    with gr.Row():
        comparison_chart = gr.Plot(label="Pass@k Comparison")
        difficulty_chart = gr.Plot(label="Difficulty Breakdown")

    def refresh():
        nonlocal all_results
        all_results = load_all_results(state)
        return results_to_table(all_results)

    def compare(indices_str):
        try:
            indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
        except ValueError:
            return build_comparison_chart([]), build_difficulty_chart([])

        selected = [all_results[i] for i in indices if 0 <= i < len(all_results)]
        return build_comparison_chart(selected), build_difficulty_chart(selected)

    refresh_btn.click(fn=refresh, outputs=[results_table])
    compare_btn.click(fn=compare, inputs=[select_indices], outputs=[comparison_chart, difficulty_chart])


def build_search_tab(state: DashboardState):
    """Tab 4: Temperature grid search visualizer."""
    import gradio as gr

    gr.Markdown("### Temperature Search Visualizer\nExplore the (T_train, T_eval) grid — the core insight from the paper.")

    df = load_search_results(state)

    if df is None or df.empty:
        gr.Markdown("""
**No search results found.**

Run a temperature grid search first:
```bash
averyml search --config configs/search/temperature_grid.yaml
```
Results will appear here automatically after refresh.
        """)
        return

    available_metrics = [c for c in df.columns if c.startswith("pass@")]
    default_metric = "pass@1" if "pass@1" in available_metrics else (available_metrics[0] if available_metrics else "")

    with gr.Row():
        metric_selector = gr.Dropdown(choices=available_metrics, value=default_metric, label="Metric")
        diagonal_toggle = gr.Checkbox(label="Show diagonal band only (T_eff 0.8-1.6)", value=False)
        refresh_btn = gr.Button("Refresh", size="sm")

    # Build initial figures for display
    init_hm = build_temperature_heatmap(df, default_metric) if default_metric else None
    init_tc, _ = build_teff_curve(df, default_metric) if default_metric else (None, {})

    with gr.Row():
        heatmap = gr.Plot(value=init_hm, label="Temperature Heatmap")
        teff_plot = gr.Plot(value=init_tc, label="T_eff Curve")

    stats_md = gr.Markdown("")

    def update_plots(metric, diagonal_only):
        data = load_search_results(state)
        if data is None or data.empty:
            empty_fig = build_temperature_heatmap(None, metric)
            return empty_fig, empty_fig, "No data"

        if diagonal_only:
            data = data[(data["t_eff"] >= 0.8) & (data["t_eff"] <= 1.6)]

        hm = build_temperature_heatmap(data, metric)
        tc, stats = build_teff_curve(data, metric)

        stats_text = ""
        if stats:
            stats_text = (
                f"**Best cell:** T_eff={stats.get('best_teff', 0):.2f}, "
                f"{metric}={stats.get('best_value', 0):.2%}  \n"
                f"**Quadratic fit:** R²={stats.get('r_squared', 0):.2f}, "
                f"optimal T_eff={stats.get('optimal_teff', 0):.2f}"
            )

        return hm, tc, stats_text

    metric_selector.change(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])
    diagonal_toggle.change(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])
    refresh_btn.click(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])


def build_config_tab(state: DashboardState):
    """Tab 5: View, validate, and edit YAML configs."""
    import gradio as gr

    all_configs = list_configs(state)
    flat_list = []
    for cat, files in sorted(all_configs.items()):
        for f in files:
            flat_list.append(f)

    with gr.Row():
        with gr.Column(scale=1):
            config_picker = gr.Dropdown(choices=flat_list, label="Config File", interactive=True,
                                        value=flat_list[0] if flat_list else None)
            validation_status = gr.Markdown("")

            with gr.Accordion("Field Documentation", open=False):
                field_docs = gr.Markdown("*Select a config to see field documentation.*")

            save_btn = gr.Button("Save", variant="secondary")
            save_status = gr.Markdown("")

        with gr.Column(scale=2):
            editor = gr.Code(label="Config Editor", language="yaml", interactive=True, lines=30)

    def load_config(config_file):
        if not config_file:
            return "", "", ""

        path = state.configs_dir / config_file
        if not path.exists():
            return "File not found", "", ""

        content = path.read_text(encoding="utf-8")
        category = config_file.split("/")[0] if "/" in config_file else "other"
        status = validate_config(content, category)

        # Generate field docs
        cls = get_config_class(category)
        docs = ""
        if cls:
            docs = f"**{cls.__name__} fields:**\n\n"
            for name, field in cls.model_fields.items():
                default = field.default if field.default is not None else "required"
                desc = field.description or ""
                docs += f"- **`{name}`** ({type(default).__name__}): {desc} *Default: `{default}`*\n"

        return content, status, docs

    def on_edit(yaml_text, config_file):
        if not config_file:
            return ""
        category = config_file.split("/")[0] if "/" in config_file else "other"
        return validate_config(yaml_text, category)

    def save_config(yaml_text, config_file):
        if not config_file:
            return "No config selected"
        path = state.configs_dir / config_file
        # Backup
        backup = path.with_suffix(".yaml.bak")
        if path.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(yaml_text, encoding="utf-8")
        return f"Saved to {path} (backup: {backup.name})"

    config_picker.change(fn=load_config, inputs=[config_picker], outputs=[editor, validation_status, field_docs])
    editor.change(fn=on_edit, inputs=[editor, config_picker], outputs=[validation_status])
    save_btn.click(fn=save_config, inputs=[editor, config_picker], outputs=[save_status])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def create_app(results_dir: str = "./results", configs_dir: str = "./configs",
               search_dir: str = "./search_results") -> Any:
    """Create the Gradio Blocks app."""
    import gradio as gr

    state = DashboardState(results_dir, configs_dir, search_dir)
    runner = JobRunner()

    with gr.Blocks(
        title="AveryML Dashboard",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
    ) as app:
        with gr.Tab("Home"):
            build_home_tab(state, runner)
        with gr.Tab("Pipeline"):
            build_pipeline_tab(state, runner)
        with gr.Tab("Results"):
            build_results_tab(state)
        with gr.Tab("Temperature Search"):
            build_search_tab(state)
        with gr.Tab("Config Editor"):
            build_config_tab(state)

    return app


def launch(results_dir: str = "./results", configs_dir: str = "./configs",
           search_dir: str = "./search_results", port: int = 7860, share: bool = False):
    """Launch the dashboard server."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Gradio is required for the dashboard. Install it with:\n"
            "  pip install averyml[dashboard]"
        )

    app = create_app(results_dir, configs_dir, search_dir)
    app.launch(server_port=port, share=share)
