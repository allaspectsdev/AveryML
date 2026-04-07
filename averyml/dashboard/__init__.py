"""AveryML web dashboard powered by Gradio.

Launch with: averyml dashboard
Requires: pip install averyml[dashboard]
"""

from __future__ import annotations

from typing import Any

from averyml.dashboard.state import DashboardState, JobRunner
from averyml.dashboard.theme import CUSTOM_CSS


def create_app(
    results_dir: str = "./results",
    configs_dir: str = "./configs",
    search_dir: str = "./search_results",
) -> Any:
    """Create the Gradio Blocks app with 10 tabs."""
    import gradio as gr

    from averyml.dashboard.tabs import (
        build_compare_tab,
        build_config_tab,
        build_data_explorer_tab,
        build_export_tab,
        build_home_tab,
        build_pipeline_tab,
        build_reproduce_tab,
        build_results_tab,
        build_search_tab,
        build_training_monitor_tab,
    )

    state = DashboardState(results_dir, configs_dir, search_dir)
    runner = JobRunner()

    with gr.Blocks(
        title="AveryML Dashboard",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css=CUSTOM_CSS,
    ) as app:
        with gr.Tab("Home"):
            build_home_tab(state, runner)
        with gr.Tab("Reproduce Paper"):
            build_reproduce_tab(state, runner)
        with gr.Tab("Pipeline"):
            build_pipeline_tab(state, runner)
        with gr.Tab("Results"):
            build_results_tab(state)
        with gr.Tab("Compare"):
            build_compare_tab(state)
        with gr.Tab("Temperature Search"):
            build_search_tab(state)
        with gr.Tab("Data Explorer"):
            build_data_explorer_tab(state)
        with gr.Tab("Training Monitor"):
            build_training_monitor_tab(state)
        with gr.Tab("Export"):
            build_export_tab(state)
        with gr.Tab("Config Editor"):
            build_config_tab(state)

    return app


def launch(
    results_dir: str = "./results",
    configs_dir: str = "./configs",
    search_dir: str = "./search_results",
    port: int = 7860,
    share: bool = False,
):
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
