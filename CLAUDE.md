# AveryML v0.2.0

Production-grade Simple Self-Distillation (SSD) pipeline for improving LLM code generation. Based on "Embarrassingly Simple Self-Distillation Improves Code Generation" (Apple, 2604.01193).

## Project Structure

- `averyml/config/` - Pydantic configs with YAML I/O (synthesis, training with LoRA, evaluation, search, experiment)
- `averyml/synthesis/` - Step 1: Sample from frozen model (checkpointing, progress bars)
- `averyml/training/` - Step 2: SFT with sequence packing, LoRA, flash attn fallback, checkpoint resume
- `averyml/evaluation/` - Step 3: Benchmark on LiveCodeBench v5/v6, HumanEval, MBPP
- `averyml/search/` - Grid search with synthesis caching (40-80% compute savings)
- `averyml/analysis/` - Full Eq.4 decomposition, fork/lock detection, significance testing
- `averyml/dashboard.py` - 8-tab Gradio dashboard (results, heatmaps, training monitor, LaTeX export)
- `averyml/utils/` - Logging, I/O, registry, tracking
- `averyml/cli.py` - Typer CLI with 8 commands + results subcommands

## Commands

```bash
averyml synthesize --config ...       # Step 1: Sample from frozen model
averyml train --config ...            # Step 2: Fine-tune (supports LoRA, packing)
averyml evaluate --config ...         # Step 3: Benchmark (LCB, HumanEval, MBPP)
averyml search --config ...           # Temperature grid search (with caching)
averyml analyze --base-model ... --ssd-model ...  # Distribution analysis
averyml dashboard                     # Launch 8-tab Gradio web dashboard
averyml run-pipeline --config ...     # Full pipeline end-to-end
averyml results list                  # List saved results
averyml results compare PATH...       # Compare result files
```

## Key Design Decisions

- N=1 sample per prompt (paper shows this suffices)
- NO correctness filtering in synthesis (core SSD insight)
- DecodingConfig shared between synthesis (T_train) and evaluation (T_eval)
- Backend abstraction via registries: vLLM/HF for inference, HF Trainer for training
- Grid search caches synthesis outputs by T_train (saves 40-80% compute)
- Sequence packing eliminates padding waste (3-5x throughput)
- Full Eq.4 decomposition: support compression + within-support reshaping + alignment
- Significance testing with bootstrap CIs and permutation tests

## V2 Features

- Synthesis checkpointing (resume on crash)
- Grid search synthesis caching
- Sequence packing for training
- LoRA/PEFT support
- HumanEval + MBPP benchmarks
- Multi-benchmark evaluation
- Full Eq.4 loss decomposition (all 3 terms)
- Statistical significance testing
- Training monitor dashboard tab
- LaTeX/CSV export dashboard tab

## Registry Pattern

Backends, benchmarks, and prompt sources register via decorators. The `__init__.py` in each subpackage imports modules to trigger registration. When adding a new component, add the import to the corresponding `__init__.py`.

## Testing

112 tests covering configs, registries, filters, metrics, grid search, results, dashboard, significance testing, and more. GPU-dependent tests require appropriate hardware.
