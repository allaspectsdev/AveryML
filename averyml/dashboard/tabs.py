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
from averyml.dashboard.theme import (
    divider, empty_state, hero_banner, highlight_card, metric_card,
    pipeline_steps, status_badge, tab_header,
)


# ========================================================================== #
# Tab 1: Home
# ========================================================================== #

def build_home_tab(state: DashboardState, runner: JobRunner):
    import gradio as gr

    # Hero banner
    gr.HTML(hero_banner())

    results = load_all_results(state)
    configs = list_configs(state)
    search_df = load_search_results(state)

    # Metric cards
    with gr.Row(equal_height=True):
        gr.HTML(metric_card(str(len(results)), "Eval Runs", icon="&#x1f4ca;"))
        gr.HTML(metric_card(
            str(len(search_df)) if search_df is not None else "0",
            "Grid Cells", icon="&#x1f50d;"
        ))
        gr.HTML(metric_card(
            str(sum(len(v) for v in configs.values())),
            "Configs", icon="&#x2699;&#xfe0f;"
        ))
        status_text, status_class = runner.get_status()
        gr.HTML(metric_card(status_badge(status_text, status_class), "Job Status", icon="&#x26a1;"))

    # Pipeline steps visualization
    gr.HTML(pipeline_steps())

    # Best result highlight + recent results table
    with gr.Row():
        with gr.Column(scale=1):
            if results:
                # Find best pass@1
                best = None
                best_val = 0.0
                for r in results:
                    p1 = r.get("results", {}).get("pass@1", 0)
                    if isinstance(p1, float) and p1 > best_val:
                        best_val = p1
                        best = r
                if best:
                    model = best.get("config", {}).get("model_id", "?").split("/")[-1]
                    bench = best.get("config", {}).get("benchmark", "?")
                    gr.HTML(highlight_card(
                        "Best pass@1",
                        f"{best_val:.1%}",
                        f"{model} on {bench}",
                    ))
            else:
                gr.HTML(empty_state(
                    "No results yet",
                    "Run your first evaluation to see results here.",
                    "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
                    icon="&#x1f680;",
                ))

        with gr.Column(scale=3):
            if results:
                gr.Markdown("#### Recent Results")
                results_table = gr.Dataframe(
                    value=results_to_table(results),
                    headers=["Model", "Benchmark", "Date", "pass@1", "pass@5", "pass@10"],
                    interactive=False, wrap=True,
                )
            else:
                results_table = gr.Dataframe(visible=False)

    refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")

    def refresh():
        new_results = load_all_results(state)
        return results_to_table(new_results)

    refresh_btn.click(fn=refresh, outputs=[results_table])


# ========================================================================== #
# Tab 2: Pipeline
# ========================================================================== #

def build_pipeline_tab(state: DashboardState, runner: JobRunner):
    import gradio as gr

    gr.HTML(tab_header("Pipeline Runner", "Select a step, pick a config, and launch. Logs stream below."))

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
            icon="&#x1f4c8;",
        ))
        return

    gr.HTML(tab_header("Results Explorer", "Compare runs side-by-side. Enter row numbers below and click Compare."))

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
            icon="&#x1f321;&#xfe0f;",
        ))
        return

    gr.HTML(tab_header("Temperature Search", "Explore the (T_train, T_eval) grid &mdash; find the optimal T_eff for your model."))

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
            icon="&#x1f4be;",
        ))

    gr.HTML(tab_header("Data Explorer", "Browse synthesized training data. Inspect samples before training."))

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

    gr.HTML(tab_header("Training Monitor", "Watch training progress. Loss curves and LR schedule from HF Trainer logs."))

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
                "Start a training job &mdash; logs appear here automatically.",
                "averyml train --config configs/training/sft_instruct.yaml",
                icon="&#x1f4c9;",
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


# ========================================================================== #
# Tab 7: Reproduce Paper
# ========================================================================== #

PAPER_PRESETS = {
    "Qwen3-4B-Instruct": {
        "config": "presets/qwen3_4b_instruct.yaml",
        "expected": "+7.5pp pass@1 on LCB v6 (34.8% -> 42.4%)",
        "t_train": "2.0", "t_eval": "1.1", "top_k": "10",
        "iterations": "2500", "gpus": "1-4x A100",
    },
    "Qwen3-4B-Thinking": {
        "config": "presets/qwen3_4b_thinking.yaml",
        "expected": "+3.3pp pass@1 on LCB v6",
        "t_train": "1.5", "t_eval": "0.6", "top_k": "10",
        "iterations": "300", "gpus": "1-4x A100",
    },
    "Qwen3-30B-Instruct": {
        "config": "presets/qwen3_30b_instruct.yaml",
        "expected": "+12.9pp pass@1 on LCB v6 (42.4% -> 55.3%)",
        "t_train": "2.0", "t_eval": "1.1", "top_k": "10",
        "iterations": "2500", "gpus": "4-8x A100",
    },
    "Llama-3.1-8B-Instruct": {
        "config": "presets/llama_8b_instruct.yaml",
        "expected": "+3.5pp pass@1 on LCB v6 (12.7% -> 16.2%)",
        "t_train": "1.5", "t_eval": "0.6", "top_k": "10",
        "iterations": "2500", "gpus": "1x A100",
    },
}


def build_reproduce_tab(state: DashboardState, runner: JobRunner):
    """Tab 7: One-click paper reproduction."""
    import gradio as gr

    gr.HTML(tab_header(
        "Reproduce Paper Results",
        "Select a model, review hyperparameters, click Run. "
        "Uses exact configs from Table 2/3 of the SSD paper."
    ))

    model_names = list(PAPER_PRESETS.keys())

    # Compute initial values for first preset
    init_preset = PAPER_PRESETS[model_names[0]]
    init_info = (
        f"**Expected:** {init_preset.get('expected', '?')}\n\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| T_train | {init_preset.get('t_train', '?')} |\n"
        f"| T_eval | {init_preset.get('t_eval', '?')} |\n"
        f"| top-k | {init_preset.get('top_k', '?')} |\n"
        f"| Iterations | {init_preset.get('iterations', '?')} |\n"
        f"| Hardware | {init_preset.get('gpus', '?')} |"
    )
    init_config_path = state.configs_dir / init_preset.get("config", "")
    init_preview = init_config_path.read_text(encoding="utf-8") if init_config_path.exists() else ""

    with gr.Row():
        with gr.Column(scale=1):
            model_picker = gr.Radio(
                choices=model_names, value=model_names[0],
                label="Model (from paper)",
            )

            info_md = gr.Markdown(value=init_info)

            with gr.Row():
                skip_synth = gr.Checkbox(label="Skip synthesis", value=False)
                skip_train = gr.Checkbox(label="Skip training", value=False)

            run_btn = gr.Button("Reproduce This Result", variant="primary", size="lg")

        with gr.Column(scale=2):
            config_preview = gr.Code(value=init_preview, label="Preset Config", language="yaml", interactive=False, lines=25)
            status_md = gr.Markdown("")
            log_output = gr.Code(label="Logs", language=None, interactive=False, lines=18)
            poll_btn = gr.Button("Refresh Logs", size="sm")

    def show_preset(model_name):
        preset = PAPER_PRESETS.get(model_name, {})
        info = (
            f"**Expected:** {preset.get('expected', '?')}\n\n"
            f"| Parameter | Value |\n|---|---|\n"
            f"| T_train | {preset.get('t_train', '?')} |\n"
            f"| T_eval | {preset.get('t_eval', '?')} |\n"
            f"| top-k | {preset.get('top_k', '?')} |\n"
            f"| Iterations | {preset.get('iterations', '?')} |\n"
            f"| Hardware | {preset.get('gpus', '?')} |"
        )
        config_path = state.configs_dir / preset.get("config", "")
        preview = config_path.read_text(encoding="utf-8") if config_path.exists() else "Preset not found"
        return info, preview

    def launch_reproduce(model_name, skip_s, skip_t):
        preset = PAPER_PRESETS.get(model_name, {})
        config_path = str(state.configs_dir / preset.get("config", ""))
        cmd = ["averyml", "run-pipeline", "--config", config_path]
        if skip_s:
            cmd.append("--skip-synthesis")
        if skip_t:
            cmd.append("--skip-training")
        msg = runner.launch(cmd)
        return f"**{msg}**", ""

    def poll_logs():
        st, cls = runner.get_status()
        return runner.get_logs(), status_badge(st, cls)

    model_picker.change(fn=show_preset, inputs=[model_picker], outputs=[info_md, config_preview])
    run_btn.click(fn=launch_reproduce, inputs=[model_picker, skip_synth, skip_train], outputs=[status_md, log_output])
    poll_btn.click(fn=poll_logs, outputs=[log_output, status_md])


# ========================================================================== #
# Tab 8: Compare
# ========================================================================== #

def build_compare_tab(state: DashboardState):
    """Tab 8: Side-by-side comparison with significance testing."""
    import gradio as gr

    all_results = load_all_results(state)

    if not all_results:
        gr.HTML(empty_state(
            "No results to compare",
            "Run at least two evaluations (base + SSD), then compare them here.",
            "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
            icon="&#x2696;&#xfe0f;",
        ))
        return

    gr.HTML(tab_header(
        "Compare Base vs SSD",
        "Select a base and SSD result to see pass@k deltas with per-difficulty breakdowns and charts."
    ))

    # Build dropdown choices
    choices = []
    for i, r in enumerate(all_results):
        cfg = r.get("config", {})
        res = r.get("results", {})
        model = cfg.get("model_id", "?").split("/")[-1]
        bench = cfg.get("benchmark", "?")
        p1 = f"{res.get('pass@1', 0):.1%}" if isinstance(res.get("pass@1"), float) else "?"
        choices.append(f"[{i}] {model} | {bench} | pass@1={p1}")

    with gr.Row():
        base_picker = gr.Dropdown(choices=choices, label="Base Model Result", interactive=True,
                                   value=choices[0] if choices else None)
        ssd_picker = gr.Dropdown(choices=choices, label="SSD Model Result", interactive=True,
                                  value=choices[1] if len(choices) > 1 else (choices[0] if choices else None))

    compare_btn = gr.Button("Compare", variant="primary")
    refresh_btn = gr.Button("Refresh Results", size="sm")

    comparison_md = gr.Markdown("")
    with gr.Row():
        comp_chart = gr.Plot(label="Pass@k Comparison")
        diff_chart = gr.Plot(label="Difficulty Breakdown")

    def do_compare(base_choice, ssd_choice):
        if not base_choice or not ssd_choice:
            return "Select both results", None, None

        base_idx = int(base_choice.split("]")[0].strip("["))
        ssd_idx = int(ssd_choice.split("]")[0].strip("["))
        base = all_results[base_idx]
        ssd = all_results[ssd_idx]

        base_res = base.get("results", {})
        ssd_res = ssd.get("results", {})
        base_model = base.get("config", {}).get("model_id", "?").split("/")[-1]
        ssd_model = ssd.get("config", {}).get("model_id", "?").split("/")[-1]

        # Build comparison table
        lines = [
            f"**Base:** {base_model} | **SSD:** {ssd_model}\n",
            "| Metric | Base | SSD | Delta |",
            "|---|---|---|---|",
        ]
        for k in [1, 5, 10]:
            key = f"pass@{k}"
            bv = base_res.get(key)
            sv = ssd_res.get(key)
            if isinstance(bv, float) and isinstance(sv, float):
                delta = sv - bv
                sign = "+" if delta > 0 else ""
                lines.append(f"| {key} | {bv:.1%} | {sv:.1%} | **{sign}{delta:.1%}** |")

        # Per-difficulty
        for diff in ["easy", "medium", "hard"]:
            key = f"pass@1_{diff}"
            bv = base_res.get(key)
            sv = ssd_res.get(key)
            if isinstance(bv, float) and isinstance(sv, float):
                delta = sv - bv
                sign = "+" if delta > 0 else ""
                lines.append(f"| pass@1 {diff} | {bv:.1%} | {sv:.1%} | {sign}{delta:.1%} |")

        from averyml.dashboard.charts import build_comparison_chart, build_difficulty_chart
        comp = build_comparison_chart([base, ssd])
        diff = build_difficulty_chart([base, ssd])

        return "\n".join(lines), comp, diff

    def refresh():
        nonlocal all_results
        all_results = load_all_results(state)
        new_choices = []
        for i, r in enumerate(all_results):
            cfg = r.get("config", {})
            res = r.get("results", {})
            model = cfg.get("model_id", "?").split("/")[-1]
            p1 = f"{res.get('pass@1', 0):.1%}" if isinstance(res.get("pass@1"), float) else "?"
            new_choices.append(f"[{i}] {model} | pass@1={p1}")
        return gr.update(choices=new_choices), gr.update(choices=new_choices)

    compare_btn.click(fn=do_compare, inputs=[base_picker, ssd_picker], outputs=[comparison_md, comp_chart, diff_chart])
    refresh_btn.click(fn=refresh, outputs=[base_picker, ssd_picker])


# ========================================================================== #
# Tab 9: Export
# ========================================================================== #

def build_export_tab(state: DashboardState):
    import gradio as gr

    all_results = load_all_results(state)

    if not all_results:
        gr.HTML(empty_state(
            "No results to export",
            "Run evaluations first, then export results to LaTeX or CSV.",
            "averyml evaluate --config configs/evaluation/lcb_v6.yaml",
            icon="&#x1f4e4;",
        ))
        return

    gr.HTML(tab_header("Export Results", "Generate LaTeX tables and CSV for your paper. Copy-paste ready for Overleaf."))

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

    gr.HTML(tab_header("Config Editor", "View, edit, and validate YAML configs with live feedback."))

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
