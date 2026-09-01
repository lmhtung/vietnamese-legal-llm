#!/usr/bin/env python3
"""Continued pre-training Qwen3.5-4B-Base on Vietnamese legal text.

This script intentionally performs raw-text causal language modeling (CPT):

    input_ids = tokenize(chunk_text) + [eos_token_id]
    labels    = input_ids (padding positions are replaced by -100)

It does NOT apply a chat template and does NOT create system/user/assistant
messages. The expected local dataset is a Hugging Face DatasetDict with
``train`` and ``validation`` splits and a ``text`` column, for example:

    datasets/vietnamese_legal_documents_processed/cpt_10k

The script uses 16-bit LoRA, evaluates token-weighted NLL/perplexity/top-k accuracy, selects the best
checkpoint by validation token NLL, supports resume, saves JSON reports, and can
optionally merge the best adapter into the base model.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Set this before importing torch. It helps the CUDA allocator when sequence
# lengths vary substantially between batches.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

try:
    from .config import parse_finetune_args
    from .metric import add_loss_perplexities, build_trainer_token_metrics
except ImportError:
    from config import parse_finetune_args
    from metric import add_loss_perplexities, build_trainer_token_metrics


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


@dataclass
class DatasetStats:
    train_rows: int
    validation_rows: int
    train_documents: int
    validation_documents: int
    train_tokens_with_eos: int
    validation_tokens_with_eos: int
    train_min_tokens: int
    train_max_tokens: int
    validation_min_tokens: int
    validation_max_tokens: int


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def newest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return None
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_PATTERN.match(child.name)
        if match:
            candidates.append((int(match.group(1)), child))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    value = args.resume_from_checkpoint
    if value is None:
        return None
    if value.lower() == "auto":
        checkpoint = newest_checkpoint(args.output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint-* directory found in {args.output_dir}"
            )
        return str(checkpoint)
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    return str(checkpoint)


def load_cpt_dataset(args: argparse.Namespace) -> DatasetDict:
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset = load_from_disk(str(args.dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {args.dataset_path}, got {type(dataset).__name__}"
        )
    missing_splits = {"train", "validation"} - set(dataset.keys())
    if missing_splits:
        raise KeyError(f"Missing dataset splits: {sorted(missing_splits)}")

    for split_name in ("train", "validation"):
        split = dataset[split_name]
        if args.text_field not in split.column_names:
            raise KeyError(
                f"Split {split_name!r} has no {args.text_field!r} column. "
                f"Available: {split.column_names}"
            )

    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    if args.max_validation_samples is not None:
        n = min(args.max_validation_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(n))

    for split_name in ("train", "validation"):
        if len(dataset[split_name]) == 0:
            raise ValueError(f"Split {split_name!r} is empty")

    # Verify document-level separation when document_id exists.
    train = dataset["train"]
    validation = dataset["validation"]
    if "document_id" in train.column_names and "document_id" in validation.column_names:
        train_ids = set(train.unique("document_id"))
        validation_ids = set(validation.unique("document_id"))
        overlap = train_ids.intersection(validation_ids)
        if overlap:
            examples = list(overlap)[:10]
            raise ValueError(
                "Document leakage between train and validation. "
                f"Example document_id values: {examples}"
            )

    return dataset


def get_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    """Return a plain text tokenizer if Unsloth returns a processor wrapper."""
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: Any,
    args: argparse.Namespace,
) -> DatasetDict:
    """Tokenize raw text, append exactly one EOS, and reject overlong chunks."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot mark chunk boundaries")

    text_field = args.text_field
    max_length = args.max_seq_length

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        raw_texts = batch[text_field]
        texts: list[str] = []
        for index, value in enumerate(raw_texts):
            if value is None or not str(value).strip():
                raise ValueError(f"Empty text encountered at batch offset {index}")
            texts.append(str(value))

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        input_ids_batch: list[list[int]] = []
        attention_masks: list[list[int]] = []
        lengths: list[int] = []
        for index, token_ids in enumerate(encoded["input_ids"]):
            # EOS is the explicit boundary between independent legal chunks.
            if not token_ids or token_ids[-1] != eos_token_id:
                token_ids = list(token_ids) + [eos_token_id]
            else:
                token_ids = list(token_ids)

            length = len(token_ids)
            if length > max_length:
                document_id = batch.get("document_id", [None] * len(texts))[index]
                chunk_id = batch.get("chunk_id", [None] * len(texts))[index]
                raise ValueError(
                    "Chunk exceeds max sequence length after Qwen3.5 tokenization: "
                    f"document_id={document_id}, chunk_id={chunk_id}, "
                    f"tokens_with_eos={length}, max_seq_length={max_length}. "
                    "Re-chunk this sample; the script will not truncate it silently."
                )

            input_ids_batch.append(token_ids)
            attention_masks.append([1] * length)
            lengths.append(length)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks,
            "length_with_eos": lengths,
        }

    tokenized_splits: dict[str, Dataset] = {}
    for split_name in ("train", "validation"):
        split = dataset[split_name]
        tokenized_splits[split_name] = split.map(
            tokenize_batch,
            batched=True,
            batch_size=args.tokenize_batch_size,
            num_proc=args.dataset_num_proc,
            remove_columns=split.column_names,
            desc=f"Tokenizing {split_name} with Qwen3.5 tokenizer",
            load_from_cache_file=True,
            keep_in_memory=False,
            writer_batch_size=64,
        )

    return DatasetDict(tokenized_splits)


def collect_dataset_stats(
    original: DatasetDict,
    tokenized: DatasetDict,
) -> DatasetStats:
    train_lengths = tokenized["train"]["length_with_eos"]
    validation_lengths = tokenized["validation"]["length_with_eos"]

    def document_count(split: Dataset) -> int:
        if "document_id" in split.column_names:
            return len(split.unique("document_id"))
        return len(split)

    return DatasetStats(
        train_rows=len(tokenized["train"]),
        validation_rows=len(tokenized["validation"]),
        train_documents=document_count(original["train"]),
        validation_documents=document_count(original["validation"]),
        train_tokens_with_eos=int(sum(train_lengths)),
        validation_tokens_with_eos=int(sum(validation_lengths)),
        train_min_tokens=int(min(train_lengths)),
        train_max_tokens=int(max(train_lengths)),
        validation_min_tokens=int(min(validation_lengths)),
        validation_max_tokens=int(max(validation_lengths)),
    )


def count_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / total,
    }


class CudaCleanupCallback(TrainerCallback):
    """Release cached blocks after scheduled evaluations."""

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


class EnsureFinalEvalSaveCallback(TrainerCallback):
    """Evaluate and save at the exact final step even if it is off-cycle."""

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if state.max_steps > 0 and state.global_step >= state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True
        return control


class CausalLMCollator:
    """Pad a batch and mask padding labels while preserving the real EOS label.

    Qwen commonly uses EOS as PAD. DataCollatorForLanguageModeling masks labels
    by token ID, which can also mask the genuine EOS appended to every chunk.
    This collator masks only positions where attention_mask == 0.
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels.masked_fill_(batch["attention_mask"].eq(0), -100)
        batch["labels"] = labels
        return batch


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def evaluate_and_save(
    trainer: SFTTrainer,
    prefix: str,
    output_path: Path,
) -> dict[str, Any]:
    clean_cuda()
    metrics = trainer.evaluate(metric_key_prefix=prefix)
    metrics = add_loss_perplexities(metrics)
    trainer.log_metrics(prefix, metrics)
    write_json(output_path, metrics)
    return metrics


def main() -> None:
    args = parse_finetune_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training script")

    visible_gpus = torch.cuda.device_count()
    if visible_gpus > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        print(
            f"WARNING: {visible_gpus} GPUs are visible, but this script is configured "
            "for one training process. Prefer CUDA_VISIBLE_DEVICES=<gpu_id>.",
            file=sys.stderr,
        )

    set_seed(args.seed)
    random.seed(args.seed)
    use_bf16 = bool(is_bfloat16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("=" * 80)
    print("Qwen3.5-4B-Base Vietnamese legal CPT")
    print(f"Config      : {args.config_path}")
    print(f"Model       : {args.model_name}")
    print(f"Dataset     : {args.dataset_path}")
    print(f"Output      : {args.output_dir}")
    print(f"Precision   : {'bfloat16' if use_bf16 else 'float16'} LoRA")
    print(f"Max length  : {args.max_seq_length:,} tokens (including EOS)")
    print("Data format : raw text CPT; no chat template")
    print("=" * 80)

    original_dataset = load_cpt_dataset(args)

    print("Loading base model and tokenizer...")
    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = get_text_tokenizer(tokenizer_or_processor)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing and validating every chunk (no truncation)...")
    tokenized_dataset = tokenize_dataset(original_dataset, tokenizer, args)
    dataset_stats = collect_dataset_stats(original_dataset, tokenized_dataset)
    write_json(args.output_dir / "dataset_summary.json", asdict(dataset_stats))

    print(
        f"Train      : {dataset_stats.train_rows:,} chunks, "
        f"{dataset_stats.train_documents:,} documents, "
        f"{dataset_stats.train_tokens_with_eos:,} tokens"
    )
    print(
        f"Validation : {dataset_stats.validation_rows:,} chunks, "
        f"{dataset_stats.validation_documents:,} documents, "
        f"{dataset_stats.validation_tokens_with_eos:,} tokens"
    )
    print(
        "Token range: "
        f"train={dataset_stats.train_min_tokens:,}..{dataset_stats.train_max_tokens:,}, "
        f"validation={dataset_stats.validation_min_tokens:,}.."
        f"{dataset_stats.validation_max_tokens:,}"
    )

    if args.target_modules == "all-linear":
        target_modules: str | list[str] = "all-linear"
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
        max_seq_length=args.max_seq_length,
    )
    model.config.use_cache = False
    parameter_stats = count_parameters(model)
    print(
        f"Trainable parameters: {parameter_stats['trainable_parameters']:,} / "
        f"{parameter_stats['total_parameters']:,} "
        f"({parameter_stats['trainable_percent']:.4f}%)"
    )

    effective_batch_size = (
        args.train_batch_size * args.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(len(tokenized_dataset["train"]) / effective_batch_size)
    if args.max_steps > 0:
        total_training_steps = args.max_steps
    else:
        total_training_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    warmup_steps = math.ceil(total_training_steps * args.warmup_ratio)
    eval_steps = min(args.eval_steps, total_training_steps)
    logging_steps = min(args.logging_steps, total_training_steps)

    run_config = vars(args).copy()
    run_config.update(
        {
            "dataset_path": str(args.dataset_path),
            "output_dir": str(args.output_dir),
            "merged_output_dir": (
                str(args.merged_output_dir) if args.merged_output_dir else None
            ),
            "dtype": str(dtype),
            "effective_batch_size_single_gpu": effective_batch_size,
            "steps_per_epoch_single_gpu": steps_per_epoch,
            "total_training_steps": total_training_steps,
            "warmup_steps": warmup_steps,
            "effective_eval_steps": eval_steps,
            "parameter_stats": parameter_stats,
        }
    )
    write_json(args.output_dir / "run_config.json", run_config)

    print(f"Optimizer steps/epoch : {steps_per_epoch:,}")
    print(f"Total optimizer steps : {total_training_steps:,}")
    print(f"Effective batch       : {effective_batch_size} chunks")
    print(f"Warmup steps          : {warmup_steps:,}")
    print(f"Eval/save every       : {eval_steps:,} optimizer steps")

    data_collator = CausalLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    # length_with_eos is retained only long enough to build dataset_summary.json.
    # Do not send this bookkeeping column into model.forward().
    trainer_train_dataset = tokenized_dataset["train"].remove_columns(
        "length_with_eos"
    )
    trainer_validation_dataset = tokenized_dataset["validation"].remove_columns(
        "length_with_eos"
    )

    preprocess_logits_for_metrics, compute_metrics = build_trainer_token_metrics(
        args.metric_logits_chunk_size
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        bf16=use_bf16,
        fp16=not use_bf16,
        bf16_full_eval=use_bf16,
        fp16_full_eval=not use_bf16,
        tf32=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_nll",
        greater_is_better=False,
        prediction_loss_only=False,
        batch_eval_metrics=True,
        include_num_input_tokens_seen=True,
        include_tokens_per_second=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=True,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=trainer_train_dataset,
        eval_dataset=trainer_validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CudaCleanupCallback(), EnsureFinalEvalSaveCallback()],
    )

    all_results: dict[str, Any] = {
        "dataset": asdict(dataset_stats),
        "parameters": parameter_stats,
    }

    if not args.skip_baseline_eval:
        print("\nEvaluating the base-equivalent model before CPT...")
        baseline_metrics = evaluate_and_save(
            trainer,
            prefix="baseline",
            output_path=args.output_dir / "baseline_results.json",
        )
        all_results["baseline"] = baseline_metrics

    resume_checkpoint = resolve_resume_checkpoint(args)
    print("\nStarting training...")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
    start_time = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.perf_counter() - start_time

    train_metrics = dict(train_result.metrics)
    train_metrics["wall_clock_seconds"] = elapsed
    train_metrics["wall_clock_hours"] = elapsed / 3600.0
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    write_json(args.output_dir / "train_results.json", train_metrics)
    all_results["train"] = train_metrics

    # load_best_model_at_end=True means trainer.model now contains the adapter
    # from the checkpoint with the lowest token-weighted eval_token_nll, not necessarily the last one.
    print("\nEvaluating the best checkpoint...")
    best_metrics = evaluate_and_save(
        trainer,
        prefix="best",
        output_path=args.output_dir / "best_results.json",
    )
    all_results["best"] = best_metrics
    all_results["best_checkpoint"] = trainer.state.best_model_checkpoint
    all_results["best_metric"] = trainer.state.best_metric

    best_adapter_dir = args.output_dir / "best_adapter"
    print(f"Saving best LoRA adapter to: {best_adapter_dir}")
    trainer.save_model(str(best_adapter_dir))
    tokenizer.save_pretrained(str(best_adapter_dir))

    if args.merge_after_train:
        merged_dir = args.merged_output_dir or (
            args.output_dir / "merged_16bit_best"
        )
        print(f"Merging best adapter into base weights: {merged_dir}")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        all_results["merged_model_dir"] = str(merged_dir)

    write_json(args.output_dir / "all_results.json", all_results)

    print("\n" + "=" * 80)
    print("Training completed")
    print(f"Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"Best token NLL  : {trainer.state.best_metric}")
    print(f"Best adapter    : {best_adapter_dir}")
    print(f"Reports         : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
