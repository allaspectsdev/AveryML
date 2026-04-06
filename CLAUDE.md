# AveryML

Production-grade implementation of Simple Self-Distillation (SSD) for improving LLM code generation. Based on the paper "Embarrassingly Simple Self-Distillation Improves Code Generation" (Apple, 2604.01193).

## Project Structure

- `averyml/config/` - Pydantic config models with YAML I/O
- `averyml/synthesis/` - Step 1: Sample solutions from frozen base model
- `averyml/training/` - Step 2: SFT on raw unverified outputs
- `averyml/evaluation/` - Step 3: Benchmark on LiveCodeBench v5/v6
- `averyml/search/` - Grid search over (T_train, T_eval) space
- `averyml/analysis/` - Token distribution and fork/lock analysis
- `averyml/utils/` - Logging, I/O, registry, tracking
- `configs/` - YAML config files for all pipeline stages
- `tests/` - pytest test suite

## Commands

```bash
pip install -e ".[dev]"          # Install with dev deps
pytest                            # Run tests
averyml --help                    # CLI help
averyml synthesize --config ...   # Step 1
averyml train --config ...        # Step 2
averyml evaluate --config ...     # Step 3
averyml search --config ...       # Temperature grid search
averyml run-pipeline --config ... # Full pipeline
```

## Key Design Decisions

- N=1 sample per prompt is the default (paper shows this suffices)
- NO correctness filtering in synthesis (core SSD insight)
- DecodingConfig is shared between synthesis (T_train) and evaluation (T_eval)
- Backend abstraction: vLLM/HF for inference, HF Trainer/torchtune for training
- Config-first: every step takes a Pydantic config, serializable to YAML

## Testing

Run `pytest` from project root. Tests cover configs, filters, metrics, and grid search. GPU-dependent tests (backends, evaluator) require appropriate hardware.
