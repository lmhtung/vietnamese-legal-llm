#!/usr/bin/env python3
"""Prepare one-document-one-chunk CPT data from the P95 token bucket.

This script consumes the output from analyze_data.py, keeps documents whose
token length is at or below the P95 threshold, normalizes their Markdown, and
saves a DatasetDict with train/validation/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from analyze_data import choose_content_column, clean_html, unwrap_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets/vietnamese_legal_documents_vhtd"
DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis/vietnamese_legal_documents_vhtd"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "datasets/vietnamese_legal_documents_vhtd_p95_20k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build P95 one-doc-per-chunk CPT dataset for Qwen3.5.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--content-dir-name", default="content")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--stats-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--content-column", default=None)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-seq-length", type=int, default=20_000)
    parser.add_argument("--reserved-special-tokens", type=int, default=16)
    parser.add_argument(
        "--p95-threshold",
        type=int,
        default=None,
        help="Override P95 token threshold. Default reads summary.json.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Fraction of selected documents used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Fraction of selected documents used for test.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep documents that become empty after Markdown normalization.",
    )
    parser.add_argument(
        "--limit-selected",
        type=int,
        default=None,
        help="Debug only: keep at most N selected documents before splitting.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_analysis_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stats_csv = args.stats_csv or (args.analysis_dir / "document_token_stats.csv")
    summary_json = args.summary_json or (args.analysis_dir / "summary.json")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Cannot find token stats CSV: {stats_csv}")
    if not summary_json.exists() and args.p95_threshold is None:
        raise FileNotFoundError(
            f"Cannot find summary JSON for P95 threshold: {summary_json}"
        )
    return stats_csv, summary_json


def load_content_dataset(args: argparse.Namespace) -> Dataset:
    content_path = args.data_dir / args.content_dir_name
    if not content_path.exists():
        raise FileNotFoundError(f"Cannot find content dataset: {content_path}")
    dataset = unwrap_dataset(load_from_disk(str(content_path)))
    if args.id_column not in dataset.column_names:
        raise KeyError(
            f"Content dataset has no id column {args.id_column!r}. "
            f"Available columns: {dataset.column_names}"
        )
    return dataset


def load_selected_stats(
    stats_csv: Path,
    p95_threshold: int,
    content_capacity: int,
    keep_empty: bool,
    limit_selected: int | None,
) -> tuple[list[int], dict[str, list[Any]], list[str], dict[str, Any]]:
    selected_indices: list[int] = []
    selected_ids: set[str] = set()
    selected_columns: dict[str, list[Any]] = {}
    metadata_fields: list[str] = []
    counters = {
        "stats_rows": 0,
        "selected_rows": 0,
        "over_p95_threshold": 0,
        "over_context_capacity": 0,
        "empty_after_clean": 0,
    }
    numeric_fields = {
        "num_chars",
        "num_lines",
        "tokens_without_eos",
        "tokens_with_eos",
        "article_count",
        "chapter_count",
        "section_count",
    }
    analysis_fields = {"document_id", *numeric_fields, "empty_after_clean"}

    with stats_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        required = {"document_id", "tokens_with_eos", "tokens_without_eos"}
        missing = required - set(fieldnames)
        if missing:
            raise KeyError(f"Stats CSV missing columns: {sorted(missing)}")

        kept_fields = [field for field in fieldnames if field != "empty_after_clean"]
        selected_columns = {field: [] for field in kept_fields}
        selected_columns["source_index"] = []
        metadata_fields = [field for field in fieldnames if field not in analysis_fields]

        for row_index, row in enumerate(reader):
            counters["stats_rows"] += 1
            document_id = str(row["document_id"])
            tokens_with_eos = int(float(row["tokens_with_eos"]))
            is_empty = str(row.get("empty_after_clean", "")).lower() == "true"
            if is_empty:
                counters["empty_after_clean"] += 1
                if not keep_empty:
                    continue
            if tokens_with_eos > p95_threshold:
                counters["over_p95_threshold"] += 1
                continue
            if tokens_with_eos > content_capacity:
                counters["over_context_capacity"] += 1
                continue
            if document_id in selected_ids:
                raise ValueError(f"Duplicate selected document_id: {document_id}")

            selected_ids.add(document_id)
            selected_indices.append(row_index)
            selected_columns["source_index"].append(row_index)
            for field in kept_fields:
                value: Any = row.get(field, "")
                if field == "document_id":
                    value = document_id
                elif field in numeric_fields:
                    value = int(float(value or 0))
                selected_columns[field].append(value)
            counters["selected_rows"] += 1
            if limit_selected is not None and len(selected_indices) >= limit_selected:
                break

    if not selected_indices:
        raise ValueError(
            "No documents selected. Check P95/context thresholds and the stats CSV."
        )
    return selected_indices, selected_columns, metadata_fields, counters


def split_dataset(
    dataset: Dataset,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    if validation_ratio < 0 or test_ratio < 0:
        raise ValueError("validation_ratio and test_ratio must be non-negative")
    heldout_ratio = validation_ratio + test_ratio
    if heldout_ratio <= 0:
        return DatasetDict({"train": dataset})
    if heldout_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be < 1")

    first_split = dataset.train_test_split(
        test_size=heldout_ratio,
        seed=seed,
        shuffle=True,
        load_from_cache_file=False,
    )
    train_dataset = first_split["train"]
    heldout_dataset = first_split["test"]

    if validation_ratio == 0:
        return DatasetDict({"train": train_dataset, "test": heldout_dataset})
    if test_ratio == 0:
        return DatasetDict({"train": train_dataset, "validation": heldout_dataset})

    test_fraction_of_heldout = test_ratio / heldout_ratio
    second_split = heldout_dataset.train_test_split(
        test_size=test_fraction_of_heldout,
        seed=seed + 1,
        shuffle=True,
        load_from_cache_file=False,
    )
    return DatasetDict(
        {
            "train": train_dataset,
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )


def summarize_split(dataset: Dataset) -> dict[str, int | float]:
    lengths = [int(value) for value in dataset["tokens_with_eos"]]
    documents = set(str(value) for value in dataset["document_id"])
    return {
        "rows": len(dataset),
        "documents": len(documents),
        "tokens": int(sum(lengths)),
        "min_tokens": int(min(lengths)) if lengths else 0,
        "mean_tokens": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_tokens": int(max(lengths)) if lengths else 0,
    }


def main() -> None:
    args = parse_args()
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive")
    if args.validation_ratio < 0 or args.test_ratio < 0:
        raise ValueError("--validation-ratio and --test-ratio must be non-negative")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("--validation-ratio + --test-ratio must be < 1")
    if args.reserved_special_tokens < 1:
        raise ValueError("--reserved-special-tokens must be >= 1")
    if args.num_proc < 1 or args.batch_size < 1:
        raise ValueError("--num-proc and --batch-size must be >= 1")
    if args.limit_selected is not None and args.limit_selected < 3:
        raise ValueError("--limit-selected must be >= 3 for train/validation/test")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new path to avoid overwriting an existing dataset."
        )

    stats_csv, summary_json = resolve_analysis_paths(args)
    summary = read_json(summary_json) if summary_json.exists() else {}
    if summary.get("content_format", "markdown") != "markdown":
        raise ValueError(
            "P95 stats were not generated in Markdown mode. "
            "Rerun analyze_data.py with --content-format markdown."
        )
    raw_p95 = (
        args.p95_threshold
        if args.p95_threshold is not None
        else summary["tokens_with_eos"]["p95"]
    )
    p95_threshold = int(math.ceil(float(raw_p95)))
    content_capacity = args.max_seq_length - args.reserved_special_tokens
    if content_capacity <= 0:
        raise ValueError("--max-seq-length must be larger than reserved tokens")
    effective_threshold = min(p95_threshold, content_capacity)

    print("=" * 80)
    print("Preparing P95 one-document CPT dataset")
    print(f"Raw data        : {args.data_dir}")
    print(f"Stats CSV       : {stats_csv}")
    print(f"P95 threshold   : {p95_threshold:,} tokens with EOS")
    print(f"Max seq length  : {args.max_seq_length:,}")
    print(f"Content capacity: {content_capacity:,}")
    print(f"Effective cap   : {effective_threshold:,}")
    print("=" * 80)

    content_dataset = load_content_dataset(args)
    expected_fingerprint = summary.get("dataset_fingerprint")
    actual_fingerprint = getattr(content_dataset, "_fingerprint", None)
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Analysis and source dataset fingerprints differ: "
            f"analysis={expected_fingerprint}, source={actual_fingerprint}. "
            "Rerun analyze_data.py for this dataset."
        )
    content_column = choose_content_column(content_dataset, args.content_column)
    selected_indices, selected_columns, passthrough_fields, counters = load_selected_stats(
        stats_csv=stats_csv,
        p95_threshold=p95_threshold,
        content_capacity=content_capacity,
        keep_empty=args.keep_empty,
        limit_selected=args.limit_selected,
    )
    selected_content = content_dataset.select(selected_indices)
    selected_stats = Dataset.from_dict(selected_columns)
    selected_dataset = concatenate_datasets(
        [selected_content, selected_stats],
        axis=1,
    )
    print(f"Selected documents: {len(selected_dataset):,} / {len(content_dataset):,}")

    def clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {
            "document_id": [],
            "chunk_id": [],
            "chunk_type": [],
            "source_index": [],
            "text": [],
            "tokens_without_eos": [],
            "tokens_with_eos": [],
            "num_chars": [],
            "num_lines": [],
            "article_count": [],
            "chapter_count": [],
            "section_count": [],
        }
        for field in passthrough_fields:
            output[field] = []

        batch_length = len(batch[args.id_column])
        for index in range(batch_length):
            document_id = str(batch[args.id_column][index])
            analyzed_id = str(batch["document_id"][index])
            if document_id != analyzed_id:
                raise ValueError(
                    "Analysis CSV and content dataset are out of order: "
                    f"content_id={document_id}, analyzed_id={analyzed_id}"
                )

            text = clean_html(batch[content_column][index], content_format="markdown")
            if not args.keep_empty and not text:
                continue

            expected_chars = int(batch["num_chars"][index])
            if len(text) != expected_chars:
                raise ValueError(
                    "Markdown normalization no longer matches analyze_data.py: "
                    f"document_id={document_id}, expected_chars={expected_chars}, "
                    f"actual_chars={len(text)}. Rerun analyze_data.py first."
                )

            tokens_with_eos = int(batch["tokens_with_eos"][index])
            if tokens_with_eos > effective_threshold:
                raise ValueError(
                    "Selected document exceeds the effective P95/context threshold: "
                    f"document_id={document_id}, tokens={tokens_with_eos}, "
                    f"threshold={effective_threshold}"
                )

            output["document_id"].append(document_id)
            output["chunk_id"].append(f"{document_id}::0000")
            output["chunk_type"].append("full_document_p95")
            output["source_index"].append(int(batch["source_index"][index]))
            output["text"].append(text)
            output["tokens_without_eos"].append(int(batch["tokens_without_eos"][index]))
            output["tokens_with_eos"].append(tokens_with_eos)
            output["num_chars"].append(expected_chars)
            output["num_lines"].append(int(batch["num_lines"][index]))
            output["article_count"].append(int(batch["article_count"][index]))
            output["chapter_count"].append(int(batch["chapter_count"][index]))
            output["section_count"].append(int(batch["section_count"][index]))
            for field in passthrough_fields:
                output[field].append(batch[field][index] or "")
        return output

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.processing_",
            dir=args.output_dir.parent,
        )
    )
    cleaned_cache_file = cache_dir / "p95_cleaned.arrow"
    print("Cleaning selected documents...")
    cleaned_dataset = selected_dataset.map(
        clean_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=selected_dataset.column_names,
        desc="Clean P95 documents",
        cache_file_name=str(cleaned_cache_file),
        load_from_cache_file=False,
    )

    dataset = split_dataset(
        cleaned_dataset,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_summaries = {
        split_name: summarize_split(split_dataset_item)
        for split_name, split_dataset_item in dataset.items()
    }
    id_sets = {
        split_name: set(split["document_id"])
        for split_name, split in dataset.items()
    }
    leakage = {
        "train_validation": len(id_sets["train"].intersection(id_sets["validation"])),
        "train_test": len(id_sets["train"].intersection(id_sets["test"])),
        "validation_test": len(id_sets["validation"].intersection(id_sets["test"])),
    }
    if any(leakage.values()):
        raise ValueError(f"Document leakage detected: {leakage}")

    manifest = {
        "source_data_dir": str(args.data_dir),
        "analysis_dir": str(args.analysis_dir),
        "stats_csv": str(stats_csv),
        "summary_json": str(summary_json),
        "content_column": content_column,
        "content_format": "markdown",
        "source_dataset_fingerprint": actual_fingerprint,
        "analysis_cache_key": summary.get("analysis_cache_key"),
        "analysis_model_name": summary.get("model_name"),
        "max_seq_length": args.max_seq_length,
        "reserved_special_tokens": args.reserved_special_tokens,
        "p95_threshold": p95_threshold,
        "effective_threshold": effective_threshold,
        "selection_counters": counters,
        "split_ratios": {
            "train": 1.0 - args.validation_ratio - args.test_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "split_summaries": split_summaries,
        "document_leakage": leakage,
        "format": {
            "text": "Markdown-normalized full document. finetune.py appends EOS.",
            "stored_splits": ["train", "validation", "test"],
            "validation_alias": "The validation split is the requested valid split.",
            "chunking": "Each selected document is exactly one CPT sample/chunk.",
        },
    }

    dataset.save_to_disk(str(args.output_dir), max_shard_size=args.max_shard_size)
    shutil.rmtree(cache_dir, ignore_errors=True)
    write_json(args.output_dir / "manifest.json", manifest)

    print("\n" + "=" * 80)
    print("P95 DATASET READY")
    print("=" * 80)
    for split_name, split_summary in split_summaries.items():
        print(
            f"{split_name:<10}: {split_summary['rows']:,} docs | "
            f"{split_summary['tokens']:,} tokens | "
            f"max={split_summary['max_tokens']:,}"
        )
    print(f"Leakage check : {leakage}")
    print(f"Output        : {args.output_dir}")
    print(f"Manifest      : {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
