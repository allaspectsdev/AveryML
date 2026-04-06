"""Configuration for the fine-tuning pipeline (Step 2 of SSD)."""

from __future__ import annotations

from pydantic import Field

from averyml.config.base import BaseConfig


class TrainingConfig(BaseConfig):
    """Configuration for supervised fine-tuning on synthesized data.

    Uses standard cross-entropy loss with AdamW + cosine LR schedule.
    Hyperparameters from the paper: peak LR 5e-6, batch size 32,
    seq length 65536, 2500 iterations (instruct) / 300 (thinking).
    """

    model_id: str = Field(description="Base model to fine-tune (HF ID or local path)")
    dataset_path: str = Field(description="Path to synthesis output (JSONL or HF dataset)")
    backend: str = Field(default="hf_trainer", description="Training backend: 'hf_trainer' or 'torchtune'")
    output_dir: str = Field(default="./checkpoints", description="Checkpoint output directory")

    # Optimizer / schedule
    learning_rate: float = Field(default=5e-6, gt=0, description="Peak learning rate")
    optimizer: str = Field(default="adamw", description="Optimizer: 'adamw'")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    lr_scheduler_type: str = Field(default="cosine", description="LR scheduler: 'cosine' or 'linear'")

    # Batch / sequence
    global_batch_size: int = Field(default=32, ge=1, description="Global batch size across all GPUs")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    max_seq_length: int = Field(default=65536, ge=1, description="Maximum sequence length")

    # Iterations
    num_train_iterations: int = Field(
        default=2500, ge=1, description="Training iterations (2500 instruct, 300 thinking)"
    )
    warmup_iterations: int = Field(
        default=250, ge=0, description="Warmup iterations (250 instruct, 50 thinking)"
    )

    # Precision / hardware
    bf16: bool = Field(default=True, description="Use bfloat16 mixed precision")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")

    # Checkpointing
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")

    # Tracking
    wandb_project: str | None = Field(default=None, description="W&B project (None to disable)")
    wandb_run_name: str | None = Field(default=None, description="W&B run name")

    seed: int = Field(default=42, description="Random seed")
