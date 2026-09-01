#!/usr/bin/env python3
"""Compare base and finetuned Qwen3.5 on identical held-out legal documents."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_from_disk
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

try:
    from .metric import (
        aggregate_evaluation_records,
        continuation_metrics,
        normalize_document,
        score_causal_document,
        split_long_continuation,
    )
except ImportError:
    from metric import (
        aggregate_evaluation_records,
        continuation_metrics,
        normalize_document,
        score_causal_document,
        split_long_continuation,
    )


QWEN35_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = (
    QWEN35_DIR / "datasets/vietnamese_legal_documents_vhtd_p95_20k"
)
DEFAULT_ADAPTER_PATH = (
    QWEN35_DIR / "outputs/qwen35_4b_legal_cpt_p95_20k_r16/best_adapter"
)
DEFAULT_OUTPUT_DIR = QWEN35_DIR / "evaluation/base_vs_finetuned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate base and finetuned Qwen3.5 with shared metrics.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--finetuned-model", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument(
        "--finetuned-format",
        choices=("auto", "adapter", "merged"),
        default="auto",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="document_id")
    parser.add_argument("--num-documents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--ppl-context-tokens", type=int, default=8192)
    parser.add_argument("--ppl-stride-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-document-tokens",
        type=int,
        default=0,
        help="0 scores the complete P95 document.",
    )
    parser.add_argument(
        "--generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-total-tokens", type=int, default=20_000)
    parser.add_argument("--max-generation-tokens", type=int, default=1024)
    parser.add_argument("--fallback-prefix-tokens", type=int, default=1024)
    parser.add_argument("--minimum-reference-tokens", type=int, default=128)
    parser.add_argument("--rouge-l-chunk-words", type=int, default=512)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_documents",
        "ppl_context_tokens",
        "ppl_stride_tokens",
        "max_total_tokens",
        "max_generation_tokens",
        "fallback_prefix_tokens",
        "minimum_reference_tokens",
        "rouge_l_chunk_words",
        "num_beams",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.ppl_context_tokens < 2:
        raise ValueError("--ppl-context-tokens must be >= 2")
    if args.ppl_stride_tokens > args.ppl_context_tokens:
        raise ValueError("--ppl-stride-tokens cannot exceed context")
    if args.max_document_tokens < 0:
        raise ValueError("--max-document-tokens must be >= 0")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be > 0 with sampling")
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid top-p or top-k")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be > 0")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
            default=str,
        )
    temporary.replace(path)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )


def load_dataset_split(args: argparse.Namespace):
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    dataset = load_from_disk(str(args.dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset).__name__}")
    if args.split not in dataset:
        raise KeyError(f"Missing split {args.split!r}; available: {list(dataset)}")
    split = dataset[args.split]
    if args.text_field not in split.column_names:
        raise KeyError(f"Missing text field {args.text_field!r}")
    return split


def append_one_eos(ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    ids = list(ids)
    if not ids or ids[-1] != eos_token_id:
        ids.append(eos_token_id)
    return ids


def select_documents(dataset, tokenizer, args: argparse.Namespace):
    """Select one deterministic sample used by both models."""
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected = []
    skipped: dict[str, int] = {}

    for dataset_index in tqdm(indices, desc="Selecting held-out documents"):
        if len(selected) >= args.num_documents:
            break
        row = dataset[dataset_index]
        raw_text = row.get(args.text_field)
        if not isinstance(raw_text, str) or not raw_text.strip():
            skipped["empty"] = skipped.get("empty", 0) + 1
            continue
        text = normalize_document(raw_text)
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if args.max_document_tokens:
            full_ids = full_ids[: max(1, args.max_document_tokens - 1)]
        full_ids = append_one_eos(full_ids, tokenizer.eos_token_id)
        if len(full_ids) < 2:
            skipped["short"] = skipped.get("short", 0) + 1
            continue

        continuation = None
        if args.generation:
            continuation = split_long_continuation(
                text,
                tokenizer,
                args.max_total_tokens,
                args.max_generation_tokens,
                args.fallback_prefix_tokens,
                args.minimum_reference_tokens,
            )
            if continuation is None:
                key = "no_continuation"
                skipped[key] = skipped.get(key, 0) + 1
                continue
        selected.append(
            {
                "dataset_index": dataset_index,
                "document_id": str(
                    row.get(args.id_field, dataset_index)
                    if args.id_field
                    else dataset_index
                ),
                "title": row.get("title"),
                "full_document_ids": full_ids,
                "continuation": continuation,
            }
        )

    if not selected:
        raise RuntimeError("No valid documents found")
    if len(selected) < args.num_documents:
        print(f"WARNING: selected {len(selected)}/{args.num_documents}: {skipped}")
    return selected


def adapter_checkpoint(model_path: str, requested_format: str) -> bool:
    if requested_format == "adapter":
        return True
    if requested_format == "merged":
        return False
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            "With a Hub checkpoint, explicitly pass --finetuned-format."
        )
    return (path / "adapter_config.json").is_file()


def load_model(
    model_path: str,
    base_model: str,
    as_adapter: bool,
    dtype: torch.dtype,
    args: argparse.Namespace,
):
    kwargs = {
        "dtype": dtype,
        "device_map": {"": args.gpu},
        "trust_remote_code": True,
        "attn_implementation": args.attention_backend,
    }
    if as_adapter:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    return model


def build_generation_kwargs(prefix, reference_tokens, tokenizer, args):
    values = {
        "input_ids": prefix,
        "attention_mask": torch.ones_like(prefix),
        "max_new_tokens": reference_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        values.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return values


def evaluate_model(label, model, tokenizer, documents, args):
    output_dir = args.output_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    records = []
    started = time.perf_counter()

    with (output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        progress = tqdm(documents, desc=f"Evaluating {label}")
        for sample_number, item in enumerate(progress, start=1):
            score = score_causal_document(
                model,
                item["full_document_ids"],
                device,
                args.ppl_context_tokens,
                args.ppl_stride_tokens,
            )
            token_count = int(score["scored_tokens"])
            if token_count == 0:
                continue
            nll = float(score["nll_sum"]) / token_count
            record = {
                "sample_number": sample_number,
                "dataset_index": item["dataset_index"],
                "document_id": item["document_id"],
                "title": item["title"],
                "full_document_tokens": len(item["full_document_ids"]),
                "full_document_nll_sum": score["nll_sum"],
                "full_document_scored_tokens": token_count,
                "full_document_top1_correct": score["top1_correct"],
                "full_document_top5_correct": score["top5_correct"],
                "full_document_nll": nll,
                "full_document_perplexity": math.exp(min(nll, 80.0)),
            }

            if args.generation:
                prefix_ids, reference_ids, method = item["continuation"]
                prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **build_generation_kwargs(
                            prefix, len(reference_ids), tokenizer, args
                        )
                    )
                generation_seconds = time.perf_counter() - generation_started
                generated_ids = generated[0, len(prefix_ids) :].tolist()
                reference = normalize_document(
                    tokenizer.decode(reference_ids, skip_special_tokens=True)
                )
                prediction = normalize_document(
                    tokenizer.decode(generated_ids, skip_special_tokens=True)
                )
                record.update(
                    {
                        "split_method": method,
                        "prefix_tokens": len(prefix_ids),
                        "reference_tokens": len(reference_ids),
                        "generated_tokens": len(generated_ids),
                        "generation_seconds": generation_seconds,
                        "ended_with_eos": bool(
                            generated_ids
                            and generated_ids[-1] == tokenizer.eos_token_id
                        ),
                        "length_ratio": len(generated_ids) / len(reference_ids),
                        "reference": reference,
                        "prediction": prediction,
                    }
                )
                record.update(
                    continuation_metrics(
                        reference,
                        prediction,
                        args.rouge_l_chunk_words,
                    )
                )
                del generated, prefix

            records.append(record)
            output.write(
                json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n"
            )
            output.flush()
            current = aggregate_evaluation_records(records)
            progress.set_postfix(
                ppl=f"{current['full_document_perplexity']:.3f}",
                top1=(
                    f"{current['full_document_top1_accuracy_percent']:.2f}"
                ),
            )

    metrics = aggregate_evaluation_records(records)
    metrics["wall_clock_seconds"] = time.perf_counter() - started
    metrics["model_label"] = label
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def compare_metrics(base, finetuned):
    ignored = {
        "num_documents",
        "num_generation_documents",
        "full_document_scored_tokens",
        "wall_clock_seconds",
    }
    lower_is_better = {
        "full_document_nll",
        "full_document_perplexity",
        "repeated_4gram_ratio_percent",
    }
    comparison = {}
    for name in sorted(set(base).intersection(finetuned) - ignored):
        left, right = base[name], finetuned[name]
        if not isinstance(left, (int, float)) or not isinstance(
            right, (int, float)
        ):
            continue
        delta = float(right) - float(left)
        comparison[name] = {
            "base": left,
            "finetuned": right,
            "delta_finetuned_minus_base": delta,
            "improvement": -delta if name in lower_is_better else delta,
            "higher_is_better": name not in lower_is_better,
        }
    return comparison


def write_report(path, comparison, args, num_documents):
    preferred = [
        "full_document_nll",
        "full_document_perplexity",
        "full_document_top1_accuracy_percent",
        "full_document_top5_accuracy_percent",
        "rouge1_f1_percent",
        "rouge2_f1_percent",
        "rougeL_chunked_f1_percent",
        "bleu",
        "chrf_plus_plus",
        "article_heading_f1_percent",
        "distinct2_percent",
        "repeated_4gram_ratio_percent",
        "eos_rate_percent",
    ]
    lines = [
        "# Qwen3.5 Vietnamese Legal CPT Evaluation",
        "",
        f"- Base model: {args.base_model}",
        f"- Finetuned model: {args.finetuned_model}",
        f"- Dataset: {args.dataset_path} ({args.split})",
        f"- Same held-out documents per model: {num_documents}",
        "",
        "| Metric | Base | Finetuned | Delta | Improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in preferred:
        if name not in comparison:
            continue
        row = comparison[name]
        values = [
            row["base"],
            row["finetuned"],
            row["delta_finetuned_minus_base"],
            row["improvement"],
        ]
        rendered = [
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in values
        ]
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "NLL, perplexity and repeated 4-gram ratio are lower-is-better.",
            "Teacher-forced metrics are token-weighted; every target token is "
            "scored exactly once through overlapping windows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Qwen3.5-4B evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dtype = resolve_dtype(args.dtype)

    print(f"Dataset: {args.dataset_path} ({args.split})")
    dataset = load_dataset_split(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    documents = select_documents(dataset, tokenizer, args)
    write_json(
        args.output_dir / "selected_documents.json",
        [
            {
                "dataset_index": row["dataset_index"],
                "document_id": row["document_id"],
                "title": row["title"],
                "tokens": len(row["full_document_ids"]),
            }
            for row in documents
        ],
    )
    config = vars(args).copy()
    config.update(
        {
            "dataset_path": str(args.dataset_path),
            "output_dir": str(args.output_dir),
            "resolved_dtype": str(dtype),
        }
    )
    write_json(args.output_dir / "config.json", config)

    print("\nEvaluating base model...")
    model = load_model(args.base_model, args.base_model, False, dtype, args)
    base_metrics = evaluate_model("base", model, tokenizer, documents, args)
    release_model(model)

    is_adapter = adapter_checkpoint(
        args.finetuned_model, args.finetuned_format
    )
    print(f"\nEvaluating finetuned {'adapter' if is_adapter else 'model'}...")
    model = load_model(
        args.finetuned_model,
        args.base_model,
        is_adapter,
        dtype,
        args,
    )
    finetuned_metrics = evaluate_model(
        "finetuned", model, tokenizer, documents, args
    )
    release_model(model)

    comparison = compare_metrics(base_metrics, finetuned_metrics)
    write_json(
        args.output_dir / "comparison.json",
        {
            "base": base_metrics,
            "finetuned": finetuned_metrics,
            "comparison": comparison,
        },
    )
    write_report(
        args.output_dir / "report.md",
        comparison,
        args,
        base_metrics["num_documents"],
    )
    print(f"\nBase PPL      : {base_metrics['full_document_perplexity']:.6f}")
    print(
        "Finetuned PPL : "
        f"{finetuned_metrics['full_document_perplexity']:.6f}"
    )
    print(f"Report        : {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
