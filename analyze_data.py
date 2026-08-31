#!/usr/bin/env python3
"""Analyze Vietnamese legal Markdown documents before CPT chunking.

The script reads the local Hugging Face dataset saved at:

    datasets/vietnamese_legal_documents_vhtd/
      content/   -> id, content (Markdown/plain text)
      metadata/  -> id, title, legal_type, legal_sectors, ...

It normalizes each Markdown document, counts Qwen tokenizer tokens, writes
per-document statistics, and recommends chunking settings for CPT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback keeps the script usable.
    BeautifulSoup = None


ARTICLE_PATTERN = re.compile(
    r"(?im)^\s*Điều\s+\d+[A-Za-zĐđ]*(?:\.\d+)*\s*[.:)]?"
)
CHAPTER_PATTERN = re.compile(r"(?im)^\s*CHƯƠNG\s+(?:[IVXLCDM]+|\d+)\b")
SECTION_PATTERN = re.compile(r"(?im)^\s*(?:MỤC|TIỂU\s+MỤC)\s+(?:[IVXLCDM]+|\d+)\b")

BLOCK_TAGS = [
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
]

HTML_TAG_PATTERN = re.compile(
    r"(?is)<!doctype\s+html\b|</?(?:html|head|body|p|div|section|article|header|footer|"
    r"ul|ol|li|h[1-6]|table|thead|tbody|tfoot|tr|td|th|br|span|script|style)\b[^>]*>"
)

ANALYSIS_CACHE_VERSION = "markdown-v2"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets/vietnamese_legal_documents_vhtd"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "qwen35/analysis/vietnamese_legal_documents_vhtd"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Read, tokenize, and analyze Vietnamese legal documents.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--content-dir-name", default="content")
    parser.add_argument("--metadata-dir-name", default="metadata")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download tokenizer files from Hugging Face.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--content-column",
        default=None,
        help="Auto-detect by default. Usually content, content_html, or text.",
    )
    parser.add_argument(
        "--content-format",
        choices=("markdown", "html", "auto"),
        default="markdown",
        help=(
            "Input representation. This dataset already stores Markdown, so markdown "
            "preserves legal comparison signs such as < 1km and > 1.15."
        ),
    )
    parser.add_argument("--id-column", default="id")
    parser.add_argument(
        "--metadata-fields",
        nargs="*",
        default=[
            "document_number",
            "title",
            "url",
            "legal_type",
            "legal_sectors",
            "issuing_authority",
            "issuance_date",
            "signers",
        ],
        help="Metadata fields copied into the per-document CSV when present.",
    )
    parser.add_argument(
        "--group-by-fields",
        nargs="*",
        default=["legal_type", "legal_sectors", "issuing_authority"],
        help="Low-cardinality metadata fields summarized in summary.json.",
    )
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N documents for a quick smoke run.",
    )
    parser.add_argument(
        "--max-seq-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 10000, 12000, 16000, 32768],
        help="Context lengths to compare for chunk-planning estimates.",
    )
    parser.add_argument(
        "--reserved-special-tokens",
        type=int,
        default=16,
        help="Reserved tokens for EOS/BOS/templates/safety when estimating chunks.",
    )
    parser.add_argument(
        "--target-utilization",
        type=float,
        default=0.95,
        help="Target chunk size is floor((max_seq_length - reserved) * utilization).",
    )
    parser.add_argument(
        "--forced-split-overlap",
        type=int,
        default=128,
        help="Estimated token overlap for documents/chunks longer than one window.",
    )
    parser.add_argument(
        "--top-k-longest",
        type=int,
        default=50,
        help="Number of longest documents to write to longest_documents.jsonl.",
    )
    parser.add_argument(
        "--analysis-cache-dir",
        type=Path,
        default=None,
        help="Directory for Hugging Face map cache; default is <output-dir>/.cache.",
    )
    parser.add_argument(
        "--no-reuse-analysis-cache",
        action="store_true",
        help="Recompute document statistics instead of reusing the analysis cache.",
    )
    return parser.parse_args()


def unwrap_dataset(dataset: Dataset | DatasetDict, preferred_split: str = "data") -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")
    if preferred_split in dataset:
        return dataset[preferred_split]
    if "train" in dataset:
        return dataset["train"]
    if len(dataset) == 1:
        return next(iter(dataset.values()))
    raise ValueError(f"Cannot choose split from DatasetDict: {list(dataset.keys())}")


def load_local_dataset(path: Path, name: str) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {name} dataset: {path}")
    return unwrap_dataset(load_from_disk(str(path)))


def choose_content_column(dataset: Dataset, requested: str | None) -> str:
    if requested is not None:
        if requested not in dataset.column_names:
            raise KeyError(
                f"Requested content column {requested!r} is missing. "
                f"Available columns: {dataset.column_names}"
            )
        return requested
    for candidate in ("content_html", "content", "text"):
        if candidate in dataset.column_names:
            return candidate
    raise KeyError(
        "Cannot detect content column. Expected one of content_html/content/text; "
        f"available columns: {dataset.column_names}"
    )


def normalize_text(text: str) -> str:
    """Normalize text without deleting Markdown syntax or legal operators."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\x00", "")
    )
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_html(text: str) -> bool:
    """Detect real HTML tags conservatively, not arbitrary angle-bracket text."""
    return bool(HTML_TAG_PATTERN.search(text))


def clean_html(raw_html: str | None, content_format: str = "markdown") -> str:
    """Normalize Markdown/plain text, parsing HTML only when explicitly needed.

    The function name is retained because prepare_p95_dataset.py imports it. Its
    default now matches this dataset stored Markdown representation.
    """
    if not raw_html:
        return ""
    if content_format not in {"markdown", "html", "auto"}:
        raise ValueError(f"Unsupported content format: {content_format}")

    should_parse_html = content_format == "html" or (
        content_format == "auto" and looks_like_html(raw_html)
    )
    if not should_parse_html:
        return normalize_text(raw_html)

    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html, "html.parser")
        if soup.head is not None:
            soup.head.decompose()
        body = soup.body if soup.body is not None else soup
        for tag in body.find_all(["script", "style", "noscript", "template"]):
            tag.decompose()
        for br in body.find_all("br"):
            br.replace_with("\n")
        for cell in body.find_all(["td", "th"]):
            cell.append("\t")
        for tag in body.find_all(BLOCK_TAGS):
            tag.insert_after("\n")
        text = body.get_text(separator="", strip=False)
    else:
        text = re.sub(r"(?is)<(script|style|noscript|template).*?>.*?</\1>", " ", raw_html)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(?:p|div|section|article|li|tr|h[1-6])>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)

    return normalize_text(text)


def count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]


def count_tokens_with_optional_eos(
    tokenizer: Any,
    token_ids: list[int],
    add_eos: bool,
) -> int:
    if add_eos and tokenizer.eos_token_id is not None:
        if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
            return len(token_ids) + 1
    return len(token_ids)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the effective tokenizer, including local tokenizer files."""
    hasher = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        hasher.update(backend.to_str().encode("utf-8"))
    else:
        fallback = {
            "class": tokenizer.__class__.__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        }
        hasher.update(json.dumps(fallback, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def build_analysis_cache_key(
    dataset: Dataset,
    tokenizer: Any,
    content_column: str,
    content_format: str,
    id_column: str,
    add_eos: bool,
    limit: int | None,
) -> str:
    payload = {
        "cache_version": ANALYSIS_CACHE_VERSION,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unknown"),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "content_column": content_column,
        "content_format": content_format,
        "id_column": id_column,
        "add_eos": add_eos,
        "limit": limit,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def dataset_to_rows(dataset: Dataset, batch_size: int = 10_000) -> list[dict[str, Any]]:
    """Materialize a compact dataset in batches instead of row-by-row Arrow reads."""
    rows: list[dict[str, Any]] = []
    columns = dataset.column_names
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        batch_length = len(batch[columns[0]]) if columns else 0
        rows.extend(
            {column: batch[column][index] for column in columns}
            for index in range(batch_length)
        )
    return rows


def safe_percentile(values: list[int | float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def numeric_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "sum": int(sum(values)),
        "min": int(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": safe_percentile(values, 50),
        "p75": safe_percentile(values, 75),
        "p90": safe_percentile(values, 90),
        "p95": safe_percentile(values, 95),
        "p99": safe_percentile(values, 99),
        "max": int(max(values)),
    }


def estimate_chunks_for_length(
    token_count: int,
    target_chunk_tokens: int,
    overlap_tokens: int,
) -> int:
    if token_count <= target_chunk_tokens:
        return 1
    step = target_chunk_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("target_chunk_tokens must be larger than overlap_tokens")
    return 1 + math.ceil((token_count - target_chunk_tokens) / step)


def chunk_plan_summary(
    token_lengths: list[int],
    max_seq_lengths: list[int],
    reserved_special_tokens: int,
    target_utilization: float,
    forced_split_overlap: int,
) -> list[dict[str, int | float]]:
    plans: list[dict[str, int | float]] = []
    for max_seq_length in sorted(set(max_seq_lengths)):
        content_capacity = max_seq_length - reserved_special_tokens
        if content_capacity <= 0:
            continue
        target_chunk_tokens = int(content_capacity * target_utilization)
        target_chunk_tokens = max(1, min(target_chunk_tokens, content_capacity))
        chunk_counts = [
            estimate_chunks_for_length(length, target_chunk_tokens, forced_split_overlap)
            for length in token_lengths
        ]
        over_context = sum(length > content_capacity for length in token_lengths)
        over_target = sum(length > target_chunk_tokens for length in token_lengths)
        plans.append(
            {
                "max_seq_length": max_seq_length,
                "content_capacity_tokens": content_capacity,
                "target_chunk_tokens": target_chunk_tokens,
                "documents_over_target": over_target,
                "documents_over_target_percent": 100.0 * over_target / len(token_lengths),
                "documents_over_context": over_context,
                "documents_over_context_percent": 100.0 * over_context / len(token_lengths),
                "estimated_chunks": int(sum(chunk_counts)),
                "estimated_mean_chunks_per_doc": float(statistics.fmean(chunk_counts)),
                "estimated_max_chunks_for_one_doc": int(max(chunk_counts)),
            }
        )
    return plans


def choose_recommended_plan(plans: list[dict[str, int | float]]) -> dict[str, int | float]:
    if not plans:
        return {}
    reasonable = [
        plan
        for plan in plans
        if float(plan["documents_over_context_percent"]) <= 5.0
    ]
    if reasonable:
        return min(reasonable, key=lambda item: int(item["max_seq_length"]))
    return min(plans, key=lambda item: float(item["documents_over_context_percent"]))


def attach_metadata(
    rows: list[dict[str, Any]],
    metadata_dataset: Dataset | None,
    id_column: str,
    fields: list[str],
    batch_size: int = 10_000,
) -> list[str]:
    """Attach metadata efficiently, using the shared row order when possible."""
    if metadata_dataset is None or id_column not in metadata_dataset.column_names:
        return []

    available_fields = [field for field in fields if field in metadata_dataset.column_names]
    if not available_fields:
        return []

    metadata_view = metadata_dataset.select_columns([id_column, *available_fields])
    aligned = len(rows) <= len(metadata_view)
    if aligned:
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = metadata_view[start:end]
            expected_ids = [str(row["document_id"]) for row in rows[start : start + batch_size]]
            if expected_ids != [str(value) for value in batch[id_column]]:
                aligned = False
                break

    if aligned:
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = metadata_view[start:end]
            for offset, row in enumerate(rows[start : start + batch_size]):
                for field in available_fields:
                    row[field] = batch[field][offset]
        return available_fields

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for start in range(0, len(metadata_view), batch_size):
        batch = metadata_view[start : start + batch_size]
        for offset, value in enumerate(batch[id_column]):
            metadata_by_id[str(value)] = {
                field: batch[field][offset] for field in available_fields
            }
    for row in rows:
        row.update(metadata_by_id.get(str(row["document_id"]), {}))
    return available_fields


def summarize_by_field(rows: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        key = str(value).strip() if value else "unknown"
        groups[key].append(int(row["tokens_with_eos"]))
    summaries = []
    for key, values in groups.items():
        summaries.append(
            {
                "value": key,
                "documents": len(values),
                "mean_tokens": round(statistics.fmean(values), 2),
                "p95_tokens": round(safe_percentile(values, 95), 2),
                "max_tokens": max(values),
            }
        )
    summaries.sort(key=lambda item: item["documents"], reverse=True)
    return summaries


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(
    path: Path,
    summary: dict[str, Any],
    chunk_plans: list[dict[str, Any]],
    recommended_plan: dict[str, Any],
    longest_rows: list[dict[str, Any]],
) -> None:
    token_stats = summary["tokens_with_eos"]
    char_stats = summary["characters"]
    article_stats = summary["articles"]
    lines = [
        "# Vietnamese Legal Documents Data Analysis",
        "",
        "## Dataset",
        "",
        f"- Content rows: {summary['content_rows']:,}",
        f"- Metadata rows: {summary['metadata_rows']:,}",
        f"- Content column: `{summary['content_column']}`",
        f"- Tokenizer: `{summary['model_name']}`",
        f"- EOS counted: {summary['add_eos']}",
        "",
        "## Token Statistics",
        "",
        f"- Total tokens: {token_stats['sum']:,}",
        f"- Mean tokens/doc: {token_stats['mean']:,.2f}",
        f"- Median tokens/doc: {token_stats['median']:,.2f}",
        f"- P90/P95/P99: {token_stats['p90']:,.2f} / {token_stats['p95']:,.2f} / {token_stats['p99']:,.2f}",
        f"- Max tokens/doc: {token_stats['max']:,}",
        "",
        "## Text Structure",
        "",
        f"- Mean chars/doc: {char_stats['mean']:,.2f}",
        f"- Mean articles/doc: {article_stats['mean']:,.2f}",
        f"- P95 articles/doc: {article_stats['p95']:,.2f}",
        f"- Max articles/doc: {article_stats['max']:,}",
        "",
        "## Chunk Plan Comparison",
        "",
        "| max_seq_length | target_chunk_tokens | docs > context | docs > target | estimated chunks | mean chunks/doc | max chunks/doc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for plan in chunk_plans:
        lines.append(
            "| "
            f"{plan['max_seq_length']:,} | "
            f"{plan['target_chunk_tokens']:,} | "
            f"{plan['documents_over_context']:,} ({plan['documents_over_context_percent']:.2f}%) | "
            f"{plan['documents_over_target']:,} ({plan['documents_over_target_percent']:.2f}%) | "
            f"{plan['estimated_chunks']:,} | "
            f"{plan['estimated_mean_chunks_per_doc']:.2f} | "
            f"{plan['estimated_max_chunks_for_one_doc']:,} |"
        )

    if recommended_plan:
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                f"- `max_seq_length`: {recommended_plan['max_seq_length']:,}",
                f"- `target_chunk_tokens`: {recommended_plan['target_chunk_tokens']:,}",
                f"- `reserved_special_tokens`: {summary['reserved_special_tokens']:,}",
                f"- Estimated chunks: {recommended_plan['estimated_chunks']:,}",
                "",
                "Practical policy: split train/validation by document ID first, then chunk within each document. "
                "Pack complete `Điều` units together up to `target_chunk_tokens`; only split an individual `Điều` "
                "when it exceeds the model context.",
            ]
        )

    lines.extend(
        [
            "",
            "## Longest Documents",
            "",
            "| rank | document_id | tokens | chars | articles | title |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for index, row in enumerate(longest_rows[:10], start=1):
        title = str(row.get("title") or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {index} | `{row['document_id']}` | {row['tokens_with_eos']:,} | "
            f"{row['num_chars']:,} | {row['article_count']:,} | {title} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_utilization <= 1.0:
        raise ValueError("--target-utilization must be in (0, 1]")
    if args.forced_split_overlap < 0:
        raise ValueError("--forced-split-overlap must be >= 0")
    if args.num_proc < 1:
        raise ValueError("--num-proc must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.top_k_longest < 1:
        raise ValueError("--top-k-longest must be >= 1")

    content_path = args.data_dir / args.content_dir_name
    metadata_path = args.data_dir / args.metadata_dir_name
    print(f"Loading content dataset : {content_path}")
    content_dataset = load_local_dataset(content_path, "content")
    metadata_dataset = None
    if metadata_path.exists():
        print(f"Loading metadata dataset: {metadata_path}")
        metadata_dataset = load_local_dataset(metadata_path, "metadata")

    if args.id_column not in content_dataset.column_names:
        raise KeyError(
            f"Content dataset has no id column {args.id_column!r}. "
            f"Available columns: {content_dataset.column_names}"
        )
    content_column = choose_content_column(content_dataset, args.content_column)

    if args.limit is not None:
        limit = min(args.limit, len(content_dataset))
        content_dataset = content_dataset.select(range(limit))

    print(f"Loading tokenizer       : {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    add_eos = tokenizer.eos_token_id is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_cache_dir = args.analysis_cache_dir or (args.output_dir / ".cache")
    analysis_cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_cache_key = build_analysis_cache_key(
        dataset=content_dataset,
        tokenizer=tokenizer,
        content_column=content_column,
        content_format=args.content_format,
        id_column=args.id_column,
        add_eos=add_eos,
        limit=args.limit,
    )
    analysis_cache_file = analysis_cache_dir / (
        f"document_stats_{analysis_cache_key}.arrow"
    )

    print(f"Content format        : {args.content_format}")
    print(f"Dataset fingerprint   : {getattr(content_dataset, '_fingerprint', 'unknown')}")
    print("Tokenizing documents...")
    print(f"Analysis cache        : {analysis_cache_file}")

    def analyze_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = {
            "document_id": [],
            "num_chars": [],
            "num_lines": [],
            "tokens_without_eos": [],
            "tokens_with_eos": [],
            "article_count": [],
            "chapter_count": [],
            "section_count": [],
            "empty_after_clean": [],
        }
        for document_id, raw_text in zip(batch[args.id_column], batch[content_column]):
            clean_text = clean_html(raw_text, content_format=args.content_format)
            token_ids = tokenize_text(tokenizer, clean_text)
            tokens_without_eos = len(token_ids)
            tokens_with_eos = count_tokens_with_optional_eos(
                tokenizer,
                token_ids,
                add_eos=add_eos,
            )
            output["document_id"].append(str(document_id))
            output["num_chars"].append(len(clean_text))
            output["num_lines"].append(count_lines(clean_text))
            output["tokens_without_eos"].append(tokens_without_eos)
            output["tokens_with_eos"].append(tokens_with_eos)
            output["article_count"].append(len(ARTICLE_PATTERN.findall(clean_text)))
            output["chapter_count"].append(len(CHAPTER_PATTERN.findall(clean_text)))
            output["section_count"].append(len(SECTION_PATTERN.findall(clean_text)))
            output["empty_after_clean"].append(not bool(clean_text.strip()))
        return output

    stats_dataset = content_dataset.map(
        analyze_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=content_dataset.column_names,
        desc="Analyze documents",
        keep_in_memory=False,
        cache_file_name=str(analysis_cache_file),
        load_from_cache_file=not args.no_reuse_analysis_cache,
    )

    rows = dataset_to_rows(stats_dataset)
    available_metadata_fields = attach_metadata(
        rows=rows,
        metadata_dataset=metadata_dataset,
        id_column=args.id_column,
        fields=args.metadata_fields,
    )

    token_lengths = [int(row["tokens_with_eos"]) for row in rows]
    char_lengths = [int(row["num_chars"]) for row in rows]
    line_counts = [int(row["num_lines"]) for row in rows]
    article_counts = [int(row["article_count"]) for row in rows]
    empty_count = sum(bool(row["empty_after_clean"]) for row in rows)

    chunk_plans = chunk_plan_summary(
        token_lengths=token_lengths,
        max_seq_lengths=args.max_seq_lengths,
        reserved_special_tokens=args.reserved_special_tokens,
        target_utilization=args.target_utilization,
        forced_split_overlap=args.forced_split_overlap,
    )
    recommended_plan = choose_recommended_plan(chunk_plans)
    longest_rows = heapq.nlargest(
        args.top_k_longest,
        rows,
        key=lambda row: int(row["tokens_with_eos"]),
    )

    type_summaries = {}
    for field in args.group_by_fields:
        if field in available_metadata_fields:
            type_summaries[field] = summarize_by_field(rows, field)[:30]

    summary = {
        "data_dir": str(args.data_dir),
        "content_rows": len(content_dataset),
        "metadata_rows": 0 if metadata_dataset is None else len(metadata_dataset),
        "content_column": content_column,
        "content_format": args.content_format,
        "dataset_fingerprint": getattr(content_dataset, '_fingerprint', 'unknown'),
        "model_name": args.model_name,
        "analysis_cache_version": ANALYSIS_CACHE_VERSION,
        "analysis_cache_key": analysis_cache_key,
        "metadata_fields": available_metadata_fields,
        "group_by_fields": [
            field for field in args.group_by_fields if field in available_metadata_fields
        ],
        "add_eos": add_eos,
        "reserved_special_tokens": args.reserved_special_tokens,
        "target_utilization": args.target_utilization,
        "forced_split_overlap": args.forced_split_overlap,
        "analysis_cache_file": str(analysis_cache_file),
        "empty_after_clean": empty_count,
        "characters": numeric_summary(char_lengths),
        "lines": numeric_summary(line_counts),
        "tokens_with_eos": numeric_summary(token_lengths),
        "articles": numeric_summary(article_counts),
        "chunk_plans": chunk_plans,
        "recommended_plan": recommended_plan,
        "metadata_group_summaries": type_summaries,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "document_id",
        *available_metadata_fields,
        "num_chars",
        "num_lines",
        "tokens_without_eos",
        "tokens_with_eos",
        "article_count",
        "chapter_count",
        "section_count",
        "empty_after_clean",
    ]
    write_csv(args.output_dir / "document_token_stats.csv", rows, csv_fields)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "longest_documents.jsonl", longest_rows)
    write_report(
        args.output_dir / "report.md",
        summary=summary,
        chunk_plans=chunk_plans,
        recommended_plan=recommended_plan,
        longest_rows=longest_rows,
    )

    print("\n" + "=" * 80)
    print("DATA ANALYSIS DONE")
    print("=" * 80)
    print(f"Documents          : {len(rows):,}")
    print(f"Total tokens + EOS : {sum(token_lengths):,}")
    print(f"Mean tokens/doc    : {statistics.fmean(token_lengths):,.2f}")
    print(f"P95 tokens/doc     : {safe_percentile(token_lengths, 95):,.2f}")
    print(f"P99 tokens/doc     : {safe_percentile(token_lengths, 99):,.2f}")
    print(f"Max tokens/doc     : {max(token_lengths):,}")
    if recommended_plan:
        print("\nRecommended chunking:")
        print(f"  max_seq_length      : {recommended_plan['max_seq_length']:,}")
        print(f"  target_chunk_tokens : {recommended_plan['target_chunk_tokens']:,}")
        print(f"  estimated chunks    : {recommended_plan['estimated_chunks']:,}")
    print("\nOutputs:")
    print(f"  CSV     : {args.output_dir / 'document_token_stats.csv'}")
    print(f"  Summary : {args.output_dir / 'summary.json'}")
    print(f"  Report  : {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
