"""AveryML CLI: synthesize, train, evaluate, search, analyze, run-pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from averyml.utils.logging import setup_logging

app = typer.Typer(
    name="averyml",
    help="Production-grade Simple Self-Distillation (SSD) pipeline for LLM code generation.",
    no_args_is_help=True,
)


def _load_config(cls, config_path: Path | None, overrides: dict):
    """Load a config from YAML and apply CLI overrides."""
    if config_path is not None:
        cfg = cls.from_yaml(config_path)
        return cfg.merge({k: v for k, v in overrides.items() if v is not None})

    # No config file — try to build from overrides alone
    try:
        return cls.model_validate({k: v for k, v in overrides.items() if v is not None})
    except Exception as e:
        typer.echo(f"Error: {e}\n\nProvide a --config YAML file or all required options.", err=True)
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# synthesize
# --------------------------------------------------------------------------- #
@app.command()
def synthesize(
    config: Annotated[Path, typer.Option("--config", help="YAML config file")] = None,
    model_id: Annotated[Optional[str], typer.Option(help="HF model ID")] = None,
    temperature: Annotated[Optional[float], typer.Option(help="T_train")] = None,
    top_k: Annotated[Optional[int], typer.Option(help="Top-k truncation")] = None,
    top_p: Annotated[Optional[float], typer.Option(help="Top-p truncation")] = None,
    n_samples: Annotated[Optional[int], typer.Option(help="Samples per prompt")] = None,
    backend: Annotated[Optional[str], typer.Option(help="'vllm' or 'hf'")] = None,
    output_path: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Step 1: Sample solutions from a frozen base model."""
    setup_logging(log_level)
    from averyml.config.synthesis import SynthesisConfig
    from averyml.synthesis.sampler import Sampler

    overrides: dict = {}
    if model_id is not None:
        overrides["model_id"] = model_id
    if n_samples is not None:
        overrides["n_samples"] = n_samples
    if backend is not None:
        overrides["backend"] = backend
    if output_path is not None:
        overrides["output_path"] = output_path

    cfg = _load_config(SynthesisConfig, config, overrides)

    # Apply decoding overrides
    decoding_overrides = {}
    if temperature is not None:
        decoding_overrides["temperature"] = temperature
    if top_k is not None:
        decoding_overrides["top_k"] = top_k
    if top_p is not None:
        decoding_overrides["top_p"] = top_p
    if decoding_overrides:
        cfg = cfg.model_copy(update={"decoding": cfg.decoding.merge(decoding_overrides)})

    result_path = Sampler(cfg).run()
    typer.echo(f"Synthesis complete: {result_path}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", help="YAML config file")] = None,
    model_id: Annotated[Optional[str], typer.Option(help="Base model HF ID or path")] = None,
    dataset_path: Annotated[Optional[str], typer.Option(help="Path to synthesis output")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="Peak learning rate")] = None,
    num_iterations: Annotated[Optional[int], typer.Option(help="Training iterations")] = None,
    backend: Annotated[Optional[str], typer.Option(help="'hf_trainer' or 'torchtune'")] = None,
    output_dir: Annotated[Optional[str], typer.Option(help="Checkpoint output dir")] = None,
    wandb_project: Annotated[Optional[str], typer.Option(help="W&B project name")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Step 2: Fine-tune the base model on synthesized data."""
    setup_logging(log_level)
    from averyml.config.training import TrainingConfig
    from averyml.training.trainer import SSDTrainer

    overrides: dict = {}
    if model_id is not None:
        overrides["model_id"] = model_id
    if dataset_path is not None:
        overrides["dataset_path"] = dataset_path
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if num_iterations is not None:
        overrides["num_train_iterations"] = num_iterations
    if backend is not None:
        overrides["backend"] = backend
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if wandb_project is not None:
        overrides["wandb_project"] = wandb_project

    cfg = _load_config(TrainingConfig, config, overrides)
    checkpoint = SSDTrainer(cfg).run()
    typer.echo(f"Training complete: {checkpoint}")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="YAML config file")] = None,
    model_id: Annotated[Optional[str], typer.Option(help="Model path or HF ID")] = None,
    benchmark: Annotated[Optional[str], typer.Option(help="Benchmark name")] = None,
    temperature: Annotated[Optional[float], typer.Option(help="T_eval")] = None,
    top_k: Annotated[Optional[int], typer.Option(help="Top-k truncation")] = None,
    top_p: Annotated[Optional[float], typer.Option(help="Top-p truncation")] = None,
    n_repeat: Annotated[Optional[int], typer.Option(help="Samples per problem")] = None,
    output_path: Annotated[Optional[str], typer.Option(help="Results output dir")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Step 3: Evaluate a model on code generation benchmarks."""
    setup_logging(log_level)
    from averyml.config.evaluation import EvaluationConfig
    from averyml.evaluation.evaluator import Evaluator

    overrides: dict = {}
    if model_id is not None:
        overrides["model_id"] = model_id
    if benchmark is not None:
        overrides["benchmark"] = benchmark
    if n_repeat is not None:
        overrides["n_repeat"] = n_repeat
    if output_path is not None:
        overrides["output_path"] = output_path

    cfg = _load_config(EvaluationConfig, config, overrides)

    decoding_overrides = {}
    if temperature is not None:
        decoding_overrides["temperature"] = temperature
    if top_k is not None:
        decoding_overrides["top_k"] = top_k
    if top_p is not None:
        decoding_overrides["top_p"] = top_p
    if decoding_overrides:
        cfg = cfg.model_copy(update={"decoding": cfg.decoding.merge(decoding_overrides)})

    results = Evaluator(cfg).run()

    typer.echo(f"\n{'=' * 60}")
    for k in cfg.k_values:
        key = f"pass@{k}"
        if key in results and isinstance(results[key], float):
            typer.echo(f"{key}: {results[key]:.2%}")
    typer.echo(f"{'=' * 60}")


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
@app.command()
def search(
    config: Annotated[Path, typer.Option("--config", help="YAML config file")],
    base_model_id: Annotated[Optional[str], typer.Option(help="Override base model")] = None,
    diagonal_only: Annotated[bool, typer.Option(help="Restrict to T_eff diagonal band")] = False,
    output_path: Annotated[Optional[str], typer.Option(help="Search results dir")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Grid search over (T_train, T_eval) temperature configurations."""
    setup_logging(log_level)
    from averyml.config.search import SearchConfig
    from averyml.search.grid_search import GridSearch

    overrides: dict = {}
    if base_model_id is not None:
        overrides["base_model_id"] = base_model_id
    if output_path is not None:
        overrides["output_path"] = output_path

    cfg = _load_config(SearchConfig, config, overrides)
    results_df = GridSearch(cfg, diagonal_only=diagonal_only).run()
    typer.echo(f"\nGrid search complete. {len(results_df)} cells evaluated.")
    typer.echo(results_df.to_string())


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
@app.command()
def dashboard(
    results_dir: Annotated[str, typer.Option(help="Results directory")] = "./results",
    configs_dir: Annotated[str, typer.Option(help="Configs directory")] = "./configs",
    search_dir: Annotated[str, typer.Option(help="Search results directory")] = "./search_results",
    port: Annotated[int, typer.Option(help="Server port")] = 7860,
    share: Annotated[bool, typer.Option(help="Create public Gradio share link")] = False,
):
    """Launch the interactive web dashboard for exploring results and managing experiments."""
    from averyml.dashboard import launch

    launch(results_dir=results_dir, configs_dir=configs_dir, search_dir=search_dir, port=port, share=share)


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
@app.command()
def analyze(
    base_model: Annotated[str, typer.Option(help="Base model HF ID or path")],
    ssd_model: Annotated[str, typer.Option(help="SSD model HF ID or path")],
    analysis_type: Annotated[str, typer.Option(help="'distributions', 'fork_lock', 'compression', or 'all'")] = "all",
    prompts_path: Annotated[Optional[Path], typer.Option(help="JSONL file with prompts")] = None,
    output_path: Annotated[Optional[str], typer.Option(help="Output directory for plots/data")] = None,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Analyze token distributions and fork/lock profiles between base and SSD models."""
    setup_logging(log_level)
    from averyml.analysis.distributions import DistributionAnalyzer
    from averyml.analysis.fork_lock import ForkLockDetector

    output_dir = Path(output_path) if output_path else Path("./analysis_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if analysis_type in ("distributions", "all"):
        analyzer = DistributionAnalyzer(base_model, ssd_model)
        results = analyzer.run(prompts_path=prompts_path, output_dir=output_dir)
        typer.echo(f"Distribution analysis saved to {output_dir}")

    if analysis_type in ("fork_lock", "all"):
        detector = ForkLockDetector(base_model, ssd_model)
        results = detector.run(prompts_path=prompts_path, output_dir=output_dir)
        typer.echo(f"Fork/lock analysis saved to {output_dir}")

    if analysis_type in ("compression", "all"):
        from averyml.analysis.compression import CompressionAnalyzer

        comp = CompressionAnalyzer(base_model, ssd_model)
        results = comp.run(prompts_path=prompts_path, output_dir=output_dir)
        typer.echo(f"Compression analysis saved to {output_dir}")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
results_app = typer.Typer(help="Query and compare stored results.")
app.add_typer(results_app, name="results")


@results_app.command("list")
def results_list(
    results_dir: Annotated[str, typer.Option(help="Results directory")] = "./results",
):
    """List all saved evaluation results."""
    from averyml.evaluation.results import ResultStore

    store = ResultStore(Path(results_dir))
    entries = store.list_results()
    if not entries:
        typer.echo("No results found.")
        return
    for entry in entries:
        typer.echo(f"  {entry['path']}  |  {entry.get('model', '?')}  |  {entry.get('date', '?')}")


@results_app.command("compare")
def results_compare(
    result_paths: Annotated[list[Path], typer.Argument(help="Result JSON files to compare")],
):
    """Compare multiple evaluation result files."""
    from averyml.evaluation.results import ResultStore

    store = ResultStore(Path("."))
    df = store.compare(result_paths)
    typer.echo(df.to_string())


# --------------------------------------------------------------------------- #
# reproduce-paper
# --------------------------------------------------------------------------- #
@app.command("reproduce-paper")
def reproduce_paper(
    model: Annotated[str, typer.Option(help="Model preset: qwen3_4b, qwen3_4b_thinking, qwen3_30b, llama_8b")] = "qwen3_4b",
    skip_synthesis: Annotated[bool, typer.Option(help="Reuse existing synthesis data")] = False,
    skip_training: Annotated[bool, typer.Option(help="Reuse existing checkpoint")] = False,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Reproduce a result from the SSD paper using exact hyperparameters.

    Uses preset configs from configs/presets/ that match Table 2/3 of the paper.
    """
    setup_logging(log_level)
    from averyml.config.experiment import ExperimentConfig
    from averyml.evaluation.evaluator import Evaluator
    from averyml.synthesis.sampler import Sampler
    from averyml.training.trainer import SSDTrainer

    preset_map = {
        "qwen3_4b": "qwen3_4b_instruct",
        "qwen3_4b_instruct": "qwen3_4b_instruct",
        "qwen3_4b_thinking": "qwen3_4b_thinking",
        "qwen3_30b": "qwen3_30b_instruct",
        "qwen3_30b_instruct": "qwen3_30b_instruct",
        "llama_8b": "llama_8b_instruct",
        "llama_8b_instruct": "llama_8b_instruct",
    }

    preset_name = preset_map.get(model)
    if not preset_name:
        typer.echo(f"Unknown model preset: {model}\nAvailable: {', '.join(sorted(preset_map.keys()))}", err=True)
        raise typer.Exit(1)

    config_path = Path("configs/presets") / f"{preset_name}.yaml"
    if not config_path.exists():
        typer.echo(f"Preset config not found: {config_path}", err=True)
        raise typer.Exit(1)

    cfg = ExperimentConfig.from_yaml(config_path)
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Reproducing paper result: {cfg.name}")
    typer.echo(f"Config: {config_path}")
    typer.echo(f"T_train={cfg.synthesis.decoding.temperature}, T_eval={cfg.evaluation.decoding.temperature}")
    typer.echo(f"{'='*60}\n")

    if not skip_synthesis:
        typer.echo("--- Step 1: Data Synthesis ---")
        dataset_path = Sampler(cfg.synthesis).run()
        cfg.training = cfg.training.model_copy(update={"dataset_path": str(dataset_path)})
    else:
        typer.echo("--- Step 1: Skipped ---")

    if not skip_training:
        typer.echo("\n--- Step 2: Fine-tuning ---")
        checkpoint = SSDTrainer(cfg.training).run()
        cfg.evaluation = cfg.evaluation.model_copy(update={"model_id": str(checkpoint)})
    else:
        typer.echo("\n--- Step 2: Skipped ---")

    typer.echo("\n--- Step 3: Evaluation ---")
    results = Evaluator(cfg.evaluation).run()

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Paper reproduction: {cfg.name}")
    for k in cfg.evaluation.k_values:
        key = f"pass@{k}"
        if key in results and isinstance(results[key], float):
            typer.echo(f"  {key}: {results[key]:.2%}")
    typer.echo(f"{'='*60}")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
@app.command()
def compare(
    base_results: Annotated[Path, typer.Argument(help="Base model results JSON")],
    ssd_results: Annotated[Path, typer.Argument(help="SSD model results JSON")],
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Compare base vs SSD results with statistical significance testing.

    Loads two result JSON files and shows pass@k deltas with bootstrap CIs,
    permutation test p-values, and Cohen's d effect sizes.
    """
    setup_logging(log_level)
    import json

    import numpy as np

    from averyml.analysis.significance import bootstrap_delta_ci, cohens_d, permutation_test

    base = json.loads(base_results.read_text())
    ssd = json.loads(ssd_results.read_text())

    base_res = base.get("results", {})
    ssd_res = ssd.get("results", {})
    base_model = base.get("config", {}).get("model_id", "?").split("/")[-1]
    ssd_model = ssd.get("config", {}).get("model_id", "?").split("/")[-1]

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Base: {base_model}")
    typer.echo(f"SSD:  {ssd_model}")
    typer.echo(f"{'='*60}\n")

    typer.echo(f"{'Metric':<20} {'Base':>8} {'SSD':>8} {'Delta':>8} {'Sig?':>6}")
    typer.echo("-" * 54)

    for k in [1, 5, 10]:
        key = f"pass@{k}"
        base_val = base_res.get(key)
        ssd_val = ssd_res.get(key)
        if isinstance(base_val, float) and isinstance(ssd_val, float):
            delta = ssd_val - base_val
            sig = "+" if delta > 0.01 else ("=" if abs(delta) < 0.01 else "-")
            typer.echo(f"{key:<20} {base_val:>7.1%} {ssd_val:>7.1%} {delta:>+7.1%} {sig:>6}")

    # Per-difficulty
    for diff in ["easy", "medium", "hard"]:
        for k in [1, 5]:
            key = f"pass@{k}_{diff}"
            base_val = base_res.get(key)
            ssd_val = ssd_res.get(key)
            if isinstance(base_val, float) and isinstance(ssd_val, float):
                delta = ssd_val - base_val
                typer.echo(f"  {key:<18} {base_val:>7.1%} {ssd_val:>7.1%} {delta:>+7.1%}")

    typer.echo(f"\n{'='*60}")


# --------------------------------------------------------------------------- #
# run-pipeline
# --------------------------------------------------------------------------- #
@app.command("run-pipeline")
def run_pipeline(
    config: Annotated[Path, typer.Option("--config", help="ExperimentConfig YAML")],
    skip_synthesis: Annotated[bool, typer.Option(help="Reuse existing synthesis data")] = False,
    skip_training: Annotated[bool, typer.Option(help="Reuse existing checkpoint")] = False,
    log_level: Annotated[str, typer.Option(help="Log level")] = "INFO",
):
    """Run the full SSD pipeline: synthesis -> training -> evaluation."""
    setup_logging(log_level)
    from averyml.config.experiment import ExperimentConfig
    from averyml.evaluation.evaluator import Evaluator
    from averyml.synthesis.sampler import Sampler
    from averyml.training.trainer import SSDTrainer

    cfg = ExperimentConfig.from_yaml(config)
    typer.echo(f"Running experiment: {cfg.name}")

    # Step 1: Synthesis
    if not skip_synthesis:
        typer.echo("\n--- Step 1: Data Synthesis ---")
        dataset_path = Sampler(cfg.synthesis).run()
        typer.echo(f"Synthesis output: {dataset_path}")
        # Wire synthesis output to training config
        cfg.training = cfg.training.model_copy(update={"dataset_path": str(dataset_path)})
    else:
        typer.echo("\n--- Step 1: Skipped (using existing data) ---")

    # Step 2: Training
    if not skip_training:
        typer.echo("\n--- Step 2: Fine-tuning ---")
        checkpoint = SSDTrainer(cfg.training).run()
        typer.echo(f"Checkpoint: {checkpoint}")
        # Wire checkpoint to evaluation config
        cfg.evaluation = cfg.evaluation.model_copy(update={"model_id": str(checkpoint)})
    else:
        typer.echo("\n--- Step 2: Skipped (using existing checkpoint) ---")

    # Step 3: Evaluation
    typer.echo("\n--- Step 3: Evaluation ---")
    results = Evaluator(cfg.evaluation).run()

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Experiment: {cfg.name}")
    for k in cfg.evaluation.k_values:
        key = f"pass@{k}"
        if key in results and isinstance(results[key], float):
            typer.echo(f"{key}: {results[key]:.2%}")
    typer.echo(f"{'=' * 60}")
