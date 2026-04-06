"""HuggingFace Trainer-based SFT backend."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from averyml.config.training import TrainingConfig
from averyml.training.backends.base import TrainingBackend
from averyml.utils.registry import training_backend_registry

logger = logging.getLogger(__name__)


@training_backend_registry.register("hf_trainer")
class HFTrainerBackend(TrainingBackend):
    """Standard HuggingFace Trainer-based supervised fine-tuning.

    Uses cross-entropy loss, AdamW optimizer, cosine LR schedule.
    Supports gradient checkpointing and bfloat16 mixed precision.
    """

    def train(self, config: TrainingConfig, dataset, tokenizer=None) -> Path:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )

        logger.info(f"Loading model: {config.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            attn_implementation="flash_attention_2" if config.bf16 else None,
        )

        # Reuse the tokenizer from the trainer if provided, to avoid mismatches
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Compute per-device batch size
        num_gpus = max(torch.cuda.device_count(), 1)
        per_device_batch = config.global_batch_size // (num_gpus * config.gradient_accumulation_steps)
        if per_device_batch < 1:
            logger.warning(
                f"global_batch_size={config.global_batch_size} < "
                f"num_gpus({num_gpus}) * grad_accum({config.gradient_accumulation_steps}). "
                f"Clamping per_device_batch to 1; effective batch size will differ from requested."
            )
            per_device_batch = 1

        output_dir = Path(config.output_dir)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=config.warmup_iterations,
            max_steps=config.num_train_iterations,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            gradient_checkpointing=config.gradient_checkpointing,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            seed=config.seed,
            report_to="wandb" if config.wandb_project else "none",
            run_name=config.wandb_run_name,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info(f"Starting training: {config.num_train_iterations} steps, lr={config.learning_rate}")
        trainer.train()

        # Save final checkpoint
        final_path = output_dir / "final_checkpoint"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Final checkpoint saved to {final_path}")

        return final_path
