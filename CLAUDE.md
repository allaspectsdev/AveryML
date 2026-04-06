# AveryML

Production-grade implementation of Simple Self-Distillation (SSD) for improving LLM code generation. Based on the paper "Embarrassingly Simple Self-Distillation Improves Code Generation" (Apple, 2604.01193).

## Project Structure

- `averyml/config/` - Pydantic config models with YAML I/O
- `averyml/synthesis/` - Step 1: Sample solutions from frozen base model
- `averyml/training/` - Step 2: SFT on raw unverified outputs
- `averyml/evaluation/` - Step 3: Benchmark on LiveCodeBench v5/v6
- `averyml/search/` - Grid search over (T_train, T_eval) space
- `averyml/analysis/` - Token distribution, fork/lock, compression analysis
- `averyml/dashboard.py` - Gradio web dashboard (5 tabs)
- `averyml/utils/` - Logging, I/O, registry, tracking
- `averyml/cli.py` - Typer CLI with 8 commands + results subcommands
- `configs/` - YAML config files for all pipeline stages
- `tests/` - pytest test suite

## Commands

```bash
pip install -e ".[dev]"              # Install with dev deps
pip install -e ".[dashboard]"        # Install with dashboard deps
pytest                                # Run tests

averyml synthesize --config ...       # Step 1: Sample from frozen model
averyml train --config ...            # Step 2: Fine-tune on samples
averyml evaluate --config ...         # Step 3: Benchmark on LCB
averyml search --config ...           # Temperature grid search
averyml analyze --base-model ... --ssd-model ...  # Distribution analysis
averyml dashboard                     # Launch Gradio web dashboard
averyml run-pipeline --config ...     # Full pipeline end-to-end
averyml results list                  # List saved results
averyml results compare PATH...       # Compare result files
```

## Key Design Decisions

- N=1 sample per prompt is the default (paper shows this suffices)
- NO correctness filtering in synthesis (core SSD insight)
- DecodingConfig is shared between synthesis (T_train) and evaluation (T_eval)
- Backend abstraction via registries: vLLM/HF for inference, HF Trainer/torchtune for training
- Config-first: every step takes a Pydantic config, serializable to YAML
- Dashboard is GPU-free: launches pipeline steps as subprocesses

## Registry Pattern

Backends and prompt sources register via decorators in their modules. The `__init__.py` in each backends/prompts package imports the modules to trigger registration. If adding a new backend, add the import to the corresponding `__init__.py`.

## Testing

Run `pytest` from project root. Tests cover configs, filters, metrics, and grid search. GPU-dependent tests (backends, evaluator) require appropriate hardware.
