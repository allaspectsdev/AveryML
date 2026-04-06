"""Dashboard tab builders — one function per tab, all with polished empty states."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import averyml
from averyml.dashboard.charts import (
    build_comparison_chart,
    build_difficulty_chart,
    build_length_histogram,
    build_temperature_heatmap,
    build_teff_curve,
    build_training_plots,
)
from averyml.dashboard.state import (
    DashboardState,
    JobRunner,
    list_configs,
    load_all_results,
    load_search_results,
    results_to_table,
    validate_config,
    get_config_class,
)
from averyml.dashboard.theme import empty_state, metric_card, status_badge


# ========================================================================== #
# Tab 1: Home
# ========================================================================== #

def build_home_tab(state: DashboardState, runner: JobRunner):
    import gradio as gr

    gr.Markdown(f"""
# AveryML Dashboard <span style="font-size:0.5em; color:#94a3b8">v{averyml.__version__}</span>

**Simple Self-Distillation pipeline for LLM code generation**
    """)

    results = load_all_results(state)
    configs = list_configs(state)
    search_df = load_search_results(state)

    # Metric cards row
    with gr.Row():
        gr.HTML(metric_card(str(len(results)), "Evaluation Runs"))
        gr.HTML(metric_card(
            str(len(search_df)) if search_df is not None else "0",
            "Grid Search Cells"
        ))
        gr.HTML(metric_card(
            str(sum(len(v) for v in configs.values())),
            "Config Files"
        ))
        status_text, status_class = runner.get_status()
        gr.HTML(metric_card(status_badge(status_text, status_class), "Job Status"))

    if results:
        gr.Markdown("### Recent Results")
        results_table = gr.Dataframe(
            value=results_to_table(results),
            headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10"],
            interactive=False, wrap=True,
        )
    else:
        gr.HTML(empty_state(
            "No results yet",
            "Run your first evaluation to see results here.",
            "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
        ))
        results_table = gr.Dataframe(visible=False)

    refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")

    def refresh():
        new_results = load_all_results(state)
        st, _ = runner.get_status()
        return results_to_table(new_results), st

    refresh_btn.click(fn=refresh, outputs=[results_table, gr.Markdown(visible=False)])


# ========================================================================== #
# Tab 2: Pipeline
# ========================================================================== #

def build_pipeline_tab(state: DashboardState, runner: JobRunner):
    import gradio as gr

    gr.Markdown("### Pipeline Runner")

    step_to_configs = {
        "Synthesize": "synthesis", "Train": "training",
        "Evaluate": "evaluation", "Full Pipeline": "experiments",
    }
    all_configs = list_configs(state)

    with gr.Row():
        with gr.Column(scale=1):
            step = gr.Radio(
                ["Synthesize", "Train", "Evaluate", "Full Pipeline"],
                value="Evaluate", label="Pipeline Step",
            )
            config_choices = all_configs.get("evaluation", [])
            config_picker = gr.Dropdown(choices=config_choices, label="Config File", interactive=True)

            with gr.Accordion("Parameter Overrides", open=False):
                override_model = gr.Textbox(label="Model ID", value="", placeholder="e.g. Qwen/Qwen3-4B-Instruct-2507")
                override_temp = gr.Slider(0.0, 3.0, step=0.1, label="Temperature", value=0, info="0 = use config default")

            with gr.Row():
                launch_btn = gr.Button("Launch", variant="primary", scale=2)
                poll_btn = gr.Button("Refresh Logs", size="sm", scale=1)

        with gr.Column(scale=2):
            config_preview = gr.Code(label="Config Preview", language="yaml", interactive=False, lines=15)
            status_md = gr.Markdown("")
            log_output = gr.Code(label="Logs", language=None, interactive=False, lines=20)

    def update_configs(selected_step):
        cat = step_to_configs.get(selected_step, "evaluation")
        choices = all_configs.get(cat, [])
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def show_preview(config_file):
        if not config_file:
            return ""
        path = state.configs_dir / config_file
        return path.read_text(encoding="utf-8") if path.exists() else "File not found"

    def launch_job(selected_step, config_file, model_override, temp_override):
        if not config_file:
            return "**No config selected**", ""
        step_cmd = {"Synthesize": "synthesize", "Train": "train", "Evaluate": "evaluate", "Full Pipeline": "run-pipeline"}
        cmd = ["averyml", step_cmd[selected_step], "--config", str(state.configs_dir / config_file)]
        if model_override:
            cmd.extend(["--model-id", model_override])
        if temp_override and temp_override > 0:
            cmd.extend(["--temperature", str(temp_override)])
        msg = runner.launch(cmd)
        return f"**{msg}**", ""

    def poll_logs():
        st, cls = runner.get_status()
        return runner.get_logs(), status_badge(st, cls)

    step.change(fn=update_configs, inputs=[step], outputs=[config_picker])
    config_picker.change(fn=show_preview, inputs=[config_picker], outputs=[config_preview])
    launch_btn.click(fn=launch_job, inputs=[step, config_picker, override_model, override_temp], outputs=[status_md, log_output])
    poll_btn.click(fn=poll_logs, outputs=[log_output, status_md])


# ========================================================================== #
# Tab 3: Results
# ========================================================================== #

def build_results_tab(state: DashboardState):
    import gradio as gr

    all_results_ref = {"data": load_all_results(state)}

    if not all_results_ref["data"]:
        gr.HTML(empty_state(
            "No evaluation results",
            "Run an evaluation to see results here.",
            "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
        ))
        return

    gr.Markdown("### Results Explorer")

    results_table = gr.Dataframe(
        value=results_to_table(all_results_ref["data"]),
        headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10"],
        interactive=False, wrap=True,
    )

    with gr.Row():
        select_indices = gr.Textbox(
            label="Row indices to compare (comma-separated)",
            placeholder="0,1,2", value="0",
        )
        compare_btn = gr.Button("Compare", variant="primary")
        refresh_btn = gr.Button("Refresh", size="sm")

    with gr.Row():
        comparison_chart = gr.Plot(label="Pass@k Comparison")
        difficulty_chart = gr.Plot(label="Difficulty Breakdown")

    def refresh():
        all_results_ref["data"] = load_all_results(state)
        return results_to_table(all_results_ref["data"])

    def compare(indices_str):
        try:
            indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
        except ValueError:
            return build_comparison_chart([]), build_difficulty_chart([])
        selected = [all_results_ref["data"][i] for i in indices if 0 <= i < len(all_results_ref["data"])]
        return build_comparison_chart(selected), build_difficulty_chart(selected)

    refresh_btn.click(fn=refresh, outputs=[results_table])
    compare_btn.click(fn=compare, inputs=[select_indices], outputs=[comparison_chart, difficulty_chart])


# ========================================================================== #
# Tab 4: Temperature Search
# ========================================================================== #

def build_search_tab(state: DashboardState):
    import gradio as gr

    df = load_search_results(state)

    if df is None or df.empty:
        gr.HTML(empty_state(
            "No grid search results",
            "Run a temperature grid search to explore the (T_train, T_eval) space.",
            "averyml search --config configs/search/temperature_grid.yaml",
        ))
        return

    gr.Markdown("### Temperature Search Visualizer")

    available_metrics = [c for c in df.columns if c.startswith("pass@")]
    default_metric = "pass@1" if "pass@1" in available_metrics else (available_metrics[0] if available_metrics else "")

    with gr.Row():
        metric_selector = gr.Dropdown(choices=available_metrics, value=default_metric, label="Metric")
        diagonal_toggle = gr.Checkbox(label="Diagonal band only (T_eff 0.8-1.6)", value=False)
        refresh_btn = gr.Button("Refresh", size="sm")

    init_hm = build_temperature_heatmap(df, default_metric) if default_metric else None
    init_tc, init_stats = (build_teff_curve(df, default_metric) if default_metric else (None, {}))

    with gr.Row():
        heatmap = gr.Plot(value=init_hm, label="Temperature Heatmap")
        teff_plot = gr.Plot(value=init_tc, label="T_eff Curve")

    stats_md = gr.Markdown(
        _format_search_stats(init_stats, default_metric) if init_stats else ""
    )

    def update_plots(metric, diagonal_only):
        data = load_search_results(state)
        if data is None or data.empty:
            from averyml.dashboard.charts import _empty_figure
            empty = _empty_figure("No data")
            return empty, empty, "No data"
        if diagonal_only:
            data = data[(data["t_eff"] >= 0.8) & (data["t_eff"] <= 1.6)]
        hm = build_temperature_heatmap(data, metric)
        tc, stats = build_teff_curve(data, metric)
        return hm, tc, _format_search_stats(stats, metric)

    metric_selector.change(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])
    diagonal_toggle.change(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])
    refresh_btn.click(fn=update_plots, inputs=[metric_selector, diagonal_toggle], outputs=[heatmap, teff_plot, stats_md])


def _format_search_stats(stats: dict, metric: str) -> str:
    if not stats:
        return ""
    return (
        f"**Best cell:** T_eff={stats.get('best_teff', 0):.2f}, "
        f"{metric}={stats.get('best_value', 0):.2%} | "
        f"**Fit:** R²={stats.get('r_squared', 0):.2f}, "
        f"optimal T_eff={stats.get('optimal_teff', 0):.2f}"
    )


# ========================================================================== #
# Tab 5: Data Explorer
# ========================================================================== #

def build_data_explorer_tab(state: DashboardState):
    import gradio as gr

    data_dirs = [state.results_dir.parent / "data", Path("./data")]
    jsonl_files = []
    for d in data_dirs:
        if d.exists():
            jsonl_files.extend([str(f) for f in sorted(d.rglob("*.jsonl")) if "_checkpoint" not in f.name])

    if not jsonl_files:
        gr.HTML(empty_state(
            "No synthesis data found",
            "Generate training data first, then come back here to explore it.",
            "averyml synthesize --config configs/synthesis/default.yaml",
        ))

    gr.Markdown("### Data Explorer")

    with gr.Row():
        file_picker = gr.Dropdown(choices=jsonl_files, label="Dataset File", interactive=True,
                                   value=jsonl_files[0] if jsonl_files else None)
        custom_path = gr.Textbox(label="Or enter path", placeholder="./data/synthesis/synthesis.jsonl")
        load_btn = gr.Button("Load", variant="primary")

    stats_md = gr.Markdown("")
    length_plot = gr.Plot(label="Response Length Distribution")

    sample_table = gr.Dataframe(
        headers=["#", "prompt_id", "prompt_text", "response", "length"],
        interactive=False, wrap=True,
    )

    with gr.Row():
        page_num = gr.Number(value=1, label="Page", precision=0, minimum=1)
        page_size = gr.Number(value=20, label="Page size", precision=0, minimum=1, maximum=100)
        nav_btn = gr.Button("Go", size="sm")

    def load_data(file_path, custom):
        path = custom.strip() if custom and custom.strip() else file_path
        if not path:
            return "No file selected", None, pd.DataFrame()
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}", None, pd.DataFrame()
        try:
            from averyml.utils.io import read_jsonl
            samples = read_jsonl(p)
        except Exception as e:
            return f"Error: {e}", None, pd.DataFrame()
        if not samples:
            return "File is empty", None, pd.DataFrame()

        lengths = [len(s.get("response", "")) for s in samples]
        avg_len = sum(lengths) / len(lengths)
        empty = sum(1 for l in lengths if l == 0)

        stats = (
            f"**{len(samples):,} samples** | "
            f"Avg length: {avg_len:,.0f} chars | "
            f"Min: {min(lengths):,} | Max: {max(lengths):,} | "
            f"Empty: {empty}"
        )

        hist = build_length_histogram(lengths)
        rows = _samples_to_rows(samples, 0, 20)
        return stats, hist, pd.DataFrame(rows)

    def navigate(file_path, custom, page, size):
        path = custom.strip() if custom and custom.strip() else file_path
        if not path:
            return pd.DataFrame()
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        from averyml.utils.io import read_jsonl
        samples = read_jsonl(p)
        page = max(1, int(page))
        size = max(1, min(100, int(size)))
        return pd.DataFrame(_samples_to_rows(samples, (page - 1) * size, size))

    load_btn.click(fn=load_data, inputs=[file_picker, custom_path], outputs=[stats_md, length_plot, sample_table])
    nav_btn.click(fn=navigate, inputs=[file_picker, custom_path, page_num, page_size], outputs=[sample_table])


def _samples_to_rows(samples: list[dict], start: int, count: int) -> list[dict]:
    rows = []
    for i, s in enumerate(samples[start:start + count], start=start):
        resp = s.get("response", "")
        rows.append({
            "#": i,
            "prompt_id": str(s.get("prompt_id", ""))[:30],
            "prompt_text": s.get("prompt_text", "")[:100] + "...",
            "response": resp[:200] + ("..." if len(resp) > 200 else ""),
            "length": len(resp),
        })
    return rows


# ========================================================================== #
# Tab 6: Training Monitor
# ========================================================================== #

def build_training_monitor_tab(state: DashboardState):
    import gradio as gr

    gr.Markdown("### Training Monitor")

    with gr.Row():
        log_path_input = gr.Textbox(label="Checkpoint directory", value="./checkpoints",
                                     placeholder="./checkpoints/sft_instruct")
        load_btn = gr.Button("Load / Refresh", variant="primary")

    with gr.Row():
        loss_plot = gr.Plot(label="Training Loss")
        lr_plot = gr.Plot(label="Learning Rate")

    stats_md = gr.Markdown("")

    def load_logs(log_dir):
        log_path = Path(log_dir)
        if not log_path.exists():
            from averyml.dashboard.charts import _empty_figure
            empty = _empty_figure("Directory not found")
            return empty, empty, f"**Not found:** `{log_dir}`"

        state_files = list(log_path.rglob("trainer_state.json"))
        if not state_files:
            from averyml.dashboard.charts import _empty_figure
            empty = _empty_figure("No trainer_state.json found")
            return empty, empty, empty_state(
                "No training logs",
                "Start a training job — logs appear here automatically.",
                "averyml train --config configs/training/sft_instruct.yaml",
            )

        latest = max(state_files, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        log_history = data.get("log_history", [])

        if not log_history:
            from averyml.dashboard.charts import _empty_figure
            empty = _empty_figure("Training started, no entries yet")
            return empty, empty, "Waiting for first log entry..."

        loss_fig, lr_fig = build_training_plots(log_history)
        step = log_history[-1].get("step", "?")
        losses = [e["loss"] for e in log_history if "loss" in e]
        current_loss = f"{losses[-1]:.4f}" if losses else "?"
        stats = f"**Step {step}** | Loss: {current_loss} | Log: `{latest.name}`"

        return loss_fig, lr_fig, stats

    load_btn.click(fn=load_logs, inputs=[log_path_input], outputs=[loss_plot, lr_plot, stats_md])

    # Auto-refresh note
    gr.Markdown("*Click Refresh to update. Training logs are read from HF Trainer's `trainer_state.json`.*",
                elem_classes=["status-idle"])


# ========================================================================== #
# Tab 7: Export
# ========================================================================== #

def build_export_tab(state: DashboardState):
    import gradio as gr

    all_results = load_all_results(state)

    if not all_results:
        gr.HTML(empty_state(
            "No results to export",
            "Run evaluations first, then export results to LaTeX or CSV.",
            "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
        ))
        return

    gr.Markdown("### Export Results")

    results_table = gr.Dataframe(
        value=results_to_table(all_results),
        headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10"],
        interactive=False,
    )

    with gr.Row():
        select_indices = gr.Textbox(label="Row indices", placeholder="0,1,2", value="0")
        export_btn = gr.Button("LaTeX Table", variant="primary")
        csv_btn = gr.Button("CSV")

    with gr.Row():
        latex_output = gr.Code(label="LaTeX (copy to Overleaf)", language=None, interactive=False, lines=15)
        csv_output = gr.Code(label="CSV", language=None, interactive=False, lines=15)

    def generate_latex(indices_str):
        selected = _select_results(all_results, indices_str)
        if not selected:
            return "No results selected"
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{SSD Evaluation Results}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & pass@1 & pass@5 & pass@10 \\",
            r"\midrule",
        ]
        for r in selected:
            cfg = r.get("config", {})
            res = r.get("results", {})
            model = cfg.get("model_id", "?").split("/")[-1].replace("_", r"\_")
            p1 = f"{res.get('pass@1', 0):.1\\%}" if isinstance(res.get("pass@1"), float) else "-"
            p5 = f"{res.get('pass@5', 0):.1\\%}" if isinstance(res.get("pass@5"), float) else "-"
            p10 = f"{res.get('pass@10', 0):.1\\%}" if isinstance(res.get("pass@10"), float) else "-"
            lines.append(f"{model} & {p1} & {p5} & {p10} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def generate_csv(indices_str):
        selected = _select_results(all_results, indices_str)
        if not selected:
            return "No results selected"
        lines = ["model,benchmark,pass@1,pass@5,pass@10"]
        for r in selected:
            cfg = r.get("config", {})
            res = r.get("results", {})
            model = cfg.get("model_id", "?")
            bench = cfg.get("benchmark", "?")
            vals = [f"{res.get(f'pass@{k}', 0):.4f}" if isinstance(res.get(f"pass@{k}"), float) else ""
                    for k in [1, 5, 10]]
            lines.append(f"{model},{bench},{','.join(vals)}")
        return "\n".join(lines)

    export_btn.click(fn=generate_latex, inputs=[select_indices], outputs=[latex_output])
    csv_btn.click(fn=generate_csv, inputs=[select_indices], outputs=[csv_output])


def _select_results(all_results: list[dict], indices_str: str) -> list[dict]:
    try:
        indices = [int(i.strip()) for i in indices_str.split(",") if i.strip()]
    except ValueError:
        return []
    return [all_results[i] for i in indices if 0 <= i < len(all_results)]


# ========================================================================== #
# Tab 8: Config Editor
# ========================================================================== #

def build_config_tab(state: DashboardState):
    import gradio as gr

    all_configs = list_configs(state)
    flat_list = [f for cat_files in sorted(all_configs.values()) for f in cat_files]

    gr.Markdown("### Config Editor")

    with gr.Row():
        with gr.Column(scale=1):
            config_picker = gr.Dropdown(choices=flat_list, label="Config File", interactive=True,
                                        value=flat_list[0] if flat_list else None)
            validation_status = gr.Markdown("")

            with gr.Accordion("Field Reference", open=False):
                field_docs = gr.Markdown("*Select a config to see field documentation.*")

            with gr.Row():
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

        cls = get_config_class(category)
        docs = ""
        if cls:
            docs = f"**{cls.__name__}**\n\n"
            for name, field in cls.model_fields.items():
                default = field.default if field.default is not None else "*required*"
                desc = field.description or ""
                docs += f"- `{name}`: {desc} *(default: {default})*\n"

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
        backup = path.with_suffix(".yaml.bak")
        if path.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(yaml_text, encoding="utf-8")
        return f"Saved (backup: {backup.name})"

    config_picker.change(fn=load_config, inputs=[config_picker], outputs=[editor, validation_status, field_docs])
    editor.change(fn=on_edit, inputs=[editor, config_picker], outputs=[validation_status])
    save_btn.click(fn=save_config, inputs=[editor, config_picker], outputs=[save_status])
