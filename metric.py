#!/usr/bin/env python3
"""Shared metrics for Qwen3.5 legal CPT training and offline evaluation."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ARTICLE_2_PATTERN = re.compile(r"(?i)(?<!\w)Điều\s+2\s*(?:[.:]|\b)")
ARTICLE_HEADING_PATTERN = re.compile(
    r"(?i)(?<!\w)Điều\s+(\d+[a-zđ]?(?:\.\d+)*)\b"
)


def normalize_document(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraphs."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_metric_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


class VietnameseUnicodeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", normalize_metric_text(text), flags=re.UNICODE)


WORD_TOKENIZER = VietnameseUnicodeTokenizer()


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def make_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    )


def counter_f1(reference: Counter[Any], prediction: Counter[Any]) -> float:
    reference_count = sum(reference.values())
    prediction_count = sum(prediction.values())
    if reference_count == 0:
        return 1.0 if prediction_count == 0 else 0.0
    if prediction_count == 0:
        return 0.0
    overlap = sum((reference & prediction).values())
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    return counter_f1(
        make_ngrams(WORD_TOKENIZER.tokenize(reference), n),
        make_ngrams(WORD_TOKENIZER.tokenize(prediction), n),
    )


def lcs_f1(reference_tokens: list[str], prediction_tokens: list[str]) -> float:
    """ROUGE-L F1 with two-row dynamic programming."""
    if not reference_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    overlap = previous[-1]
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def chunked_rouge_l_f1(
    reference: str,
    prediction: str,
    chunk_words: int = 512,
) -> float:
    """Compute aligned chunk ROUGE-L to bound quadratic LCS work."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    reference_tokens = WORD_TOKENIZER.tokenize(reference)
    prediction_tokens = WORD_TOKENIZER.tokenize(prediction)
    reference_chunks = [
        reference_tokens[index : index + chunk_words]
        for index in range(0, len(reference_tokens), chunk_words)
    ]
    prediction_chunks = [
        prediction_tokens[index : index + chunk_words]
        for index in range(0, len(prediction_tokens), chunk_words)
    ]
    chunk_count = max(len(reference_chunks), len(prediction_chunks))
    if chunk_count == 0:
        return 1.0

    weighted_sum = 0.0
    total_weight = 0
    for index in range(chunk_count):
        reference_chunk = (
            reference_chunks[index] if index < len(reference_chunks) else []
        )
        prediction_chunk = (
            prediction_chunks[index] if index < len(prediction_chunks) else []
        )
        weight = max(1, len(reference_chunk), len(prediction_chunk))
        score = lcs_f1(reference_chunk, prediction_chunk)
        weighted_sum += score * weight
        total_weight += weight
    return weighted_sum / total_weight


def distinct_n(text: str, n: int = 2) -> float:
    ngrams = make_ngrams(WORD_TOKENIZER.tokenize(text), n)
    total = sum(ngrams.values())
    return len(ngrams) / total if total else 0.0


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    ngrams = make_ngrams(WORD_TOKENIZER.tokenize(text), n)
    total = sum(ngrams.values())
    if total == 0:
        return 0.0
    return sum(count - 1 for count in ngrams.values() if count > 1) / total


def article_heading_f1(reference: str, prediction: str) -> float:
    reference_articles = Counter(
        match.lower() for match in ARTICLE_HEADING_PATTERN.findall(reference)
    )
    prediction_articles = Counter(
        match.lower() for match in ARTICLE_HEADING_PATTERN.findall(prediction)
    )
    return counter_f1(reference_articles, prediction_articles)


def add_loss_perplexities(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add perplexity next to each loss or token-NLL metric."""
    result = dict(metrics)
    for key, value in list(metrics.items()):
        if not isinstance(value, (int, float)):
            continue
        if key.endswith("_loss"):
            output_key = f"{key[:-len('_loss')]}_perplexity"
        elif key.endswith("_token_nll"):
            output_key = f"{key[:-len('_token_nll')]}_token_perplexity"
        else:
            continue
        result.setdefault(
            output_key,
            math.exp(float(value))
            if math.isfinite(float(value)) and value < 80
            else float("inf"),
        )
    return result


@dataclass
class TokenMetricPreprocessor:
    """Compress causal-LM logits to four aggregate values on the GPU."""

    logits_chunk_size: int = 256

    @torch.inference_mode()
    def __call__(self, logits: Any, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected logits tensor, got {type(logits).__name__}")
        if self.logits_chunk_size <= 0:
            raise ValueError("logits_chunk_size must be > 0")

        shifted_logits = logits[..., :-1, :]
        shifted_labels = labels[..., 1:]
        totals = torch.zeros(4, dtype=torch.float64, device=logits.device)
        for start in range(0, shifted_labels.shape[-1], self.logits_chunk_size):
            end = min(start + self.logits_chunk_size, shifted_labels.shape[-1])
            chunk_logits = shifted_logits[..., start:end, :].float()
            chunk_labels = shifted_labels[..., start:end]
            valid = chunk_labels.ne(-100)
            if not bool(valid.any()):
                continue
            valid_logits = chunk_logits[valid]
            valid_labels = chunk_labels[valid]
            totals[0] += F.cross_entropy(
                valid_logits, valid_labels, reduction="sum"
            ).double()
            totals[1] += valid_labels.numel()
            totals[2] += valid_logits.argmax(dim=-1).eq(valid_labels).sum().double()
            top5 = valid_logits.topk(
                k=min(5, valid_logits.shape[-1]), dim=-1
            ).indices
            totals[3] += (
                top5.eq(valid_labels.unsqueeze(-1)).any(dim=-1).sum().double()
            )
        # The row dimension makes distributed gathering unambiguous.
        return totals.unsqueeze(0)


class TokenMetricAccumulator:
    """Streaming compute_metrics callback for batch_eval_metrics=True."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.nll_sum = 0.0
        self.scored_tokens = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def __call__(
        self,
        eval_prediction: Any,
        compute_result: bool = False,
    ) -> dict[str, Any]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        sums = (
            torch.as_tensor(predictions)
            .reshape(-1, 4)
            .double()
            .sum(dim=0)
            .tolist()
        )
        self.nll_sum += float(sums[0])
        self.scored_tokens += int(round(sums[1]))
        self.top1_correct += int(round(sums[2]))
        self.top5_correct += int(round(sums[3]))

        if not compute_result:
            return {}
        if self.scored_tokens == 0:
            result = {
                "token_nll": float("nan"),
                "token_perplexity": float("nan"),
                "top1_token_accuracy_percent": 0.0,
                "top5_token_accuracy_percent": 0.0,
                "scored_tokens": 0,
            }
        else:
            token_nll = self.nll_sum / self.scored_tokens
            result = {
                "token_nll": token_nll,
                "token_perplexity": math.exp(min(token_nll, 80.0)),
                "top1_token_accuracy_percent": (
                    100.0 * self.top1_correct / self.scored_tokens
                ),
                "top5_token_accuracy_percent": (
                    100.0 * self.top5_correct / self.scored_tokens
                ),
                "scored_tokens": self.scored_tokens,
            }
        self.reset()
        return result


def build_trainer_token_metrics(
    logits_chunk_size: int = 256,
) -> tuple[TokenMetricPreprocessor, TokenMetricAccumulator]:
    return TokenMetricPreprocessor(logits_chunk_size), TokenMetricAccumulator()


@torch.inference_mode()
def score_causal_document(
    model: Any,
    document_ids: list[int],
    device: torch.device,
    max_context_tokens: int,
    stride_tokens: int,
) -> dict[str, float | int]:
    """Score every predictable token exactly once using overlapping windows."""
    if max_context_tokens < 2:
        raise ValueError("max_context_tokens must be >= 2")
    if stride_tokens <= 0 or stride_tokens > max_context_tokens:
        raise ValueError("stride_tokens must be in (0, max_context_tokens]")
    if len(document_ids) < 2:
        return {
            "nll_sum": 0.0,
            "scored_tokens": 0,
            "top1_correct": 0,
            "top5_correct": 0,
        }

    sequence_length = len(document_ids)
    end = min(max_context_tokens, sequence_length)
    scored_until = 1
    totals = {
        "nll_sum": 0.0,
        "scored_tokens": 0,
        "top1_correct": 0,
        "top5_correct": 0,
    }
    while True:
        begin = max(0, end - max_context_tokens)
        window = torch.tensor(
            [document_ids[begin:end]], dtype=torch.long, device=device
        )
        output = model(
            input_ids=window,
            attention_mask=torch.ones_like(window),
            use_cache=False,
        )
        first_target_global = max(scored_until, begin + 1)
        if first_target_global < end:
            first_logit_local = first_target_global - begin - 1
            last_logit_local = end - begin - 1
            logits = output.logits[
                0, first_logit_local:last_logit_local, :
            ].float()
            labels = window[0, first_target_global - begin : end - begin]
            totals["nll_sum"] += F.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            totals["scored_tokens"] += labels.numel()
            totals["top1_correct"] += logits.argmax(dim=-1).eq(labels).sum().item()
            top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            totals["top5_correct"] += (
                top5.eq(labels.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            del logits, labels, top5
        del output, window
        if end >= sequence_length:
            break
        scored_until = end
        end = min(end + stride_tokens, sequence_length)
    return totals


def split_long_continuation(
    text: str,
    tokenizer: Any,
    max_total_tokens: int,
    max_generation_tokens: int,
    fallback_prefix_tokens: int,
    minimum_reference_tokens: int,
) -> tuple[list[int], list[int], str] | None:
    """Split before Article 2, with a token-prefix fallback."""
    article_2_match = ARTICLE_2_PATTERN.search(text)
    if article_2_match is not None:
        prefix_ids = tokenizer(
            text[: article_2_match.start()].rstrip(), add_special_tokens=False
        )["input_ids"]
        reference_ids = tokenizer(
            text[article_2_match.start() :].lstrip(), add_special_tokens=False
        )["input_ids"]
        split_method = "before_article_2"
    else:
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= fallback_prefix_tokens + minimum_reference_tokens:
            return None
        prefix_ids = full_ids[:fallback_prefix_tokens]
        reference_ids = full_ids[fallback_prefix_tokens:]
        split_method = "token_fallback"

    reference_length = min(
        len(reference_ids),
        max_generation_tokens,
        max_total_tokens - len(prefix_ids),
    )
    if reference_length < minimum_reference_tokens:
        return None
    return prefix_ids, reference_ids[:reference_length], split_method


def continuation_metrics(
    reference: str,
    prediction: str,
    rouge_l_chunk_words: int = 512,
) -> dict[str, float]:
    return {
        "rouge1_f1": rouge_n_f1(reference, prediction, n=1),
        "rouge2_f1": rouge_n_f1(reference, prediction, n=2),
        "rougeL_chunked_f1": chunked_rouge_l_f1(
            reference, prediction, chunk_words=rouge_l_chunk_words
        ),
        "article_heading_f1": article_heading_f1(reference, prediction),
        "distinct2": distinct_n(prediction, n=2),
        "repeated_4gram_ratio": repeated_ngram_ratio(prediction, n=4),
    }


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus BLEU-4 with modified precision and exponential smoothing."""
    clipped = [0] * 4
    totals = [0] * 4
    prediction_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references):
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        prediction_length += len(prediction_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, 5):
            predicted = make_ngrams(prediction_tokens, order)
            expected = make_ngrams(reference_tokens, order)
            clipped[order - 1] += sum((predicted & expected).values())
            totals[order - 1] += sum(predicted.values())
    if prediction_length == 0:
        return 0.0
    smooth = 1.0
    log_precisions = []
    for matches, total in zip(clipped, totals):
        if total == 0:
            return 0.0
        if matches == 0:
            smooth *= 2.0
            precision = 1.0 / (smooth * total)
        else:
            precision = matches / total
        log_precisions.append(math.log(precision))
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return 100.0 * brevity_penalty * math.exp(sum(log_precisions) / 4.0)


def corpus_chrf_plus_plus(
    predictions: list[str], references: list[str], beta: float = 2.0
) -> float:
    """Corpus chrF++ using character orders 1..6 and word orders 1..2."""
    precision_sum = 0.0
    recall_sum = 0.0
    order_count = 0
    for level, maximum_order in (("char", 6), ("word", 2)):
        for order in range(1, maximum_order + 1):
            matches = predicted_total = reference_total = 0
            for prediction, reference in zip(predictions, references):
                prediction_units = list(prediction) if level == "char" else prediction.split()
                reference_units = list(reference) if level == "char" else reference.split()
                predicted = make_ngrams(prediction_units, order)
                expected = make_ngrams(reference_units, order)
                matches += sum((predicted & expected).values())
                predicted_total += sum(predicted.values())
                reference_total += sum(expected.values())
            precision_sum += matches / predicted_total if predicted_total else 0.0
            recall_sum += matches / reference_total if reference_total else 0.0
            order_count += 1
    precision = precision_sum / order_count
    recall = recall_sum / order_count
    denominator = beta * beta * precision + recall
    return (
        100.0 * (1.0 + beta * beta) * precision * recall / denominator
        if denominator
        else 0.0
    )


def aggregate_evaluation_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate token-weighted LM metrics and optional generation metrics."""
    if not records:
        return {"num_documents": 0}

    total_nll = sum(float(row["full_document_nll_sum"]) for row in records)
    scored_tokens = sum(int(row["full_document_scored_tokens"]) for row in records)
    top1_correct = sum(int(row["full_document_top1_correct"]) for row in records)
    top5_correct = sum(int(row["full_document_top5_correct"]) for row in records)
    token_nll = total_nll / scored_tokens if scored_tokens else float("nan")
    metrics: dict[str, Any] = {
        "num_documents": len(records),
        "full_document_scored_tokens": scored_tokens,
        "full_document_nll": token_nll,
        "full_document_perplexity": (
            math.exp(min(token_nll, 80.0)) if scored_tokens else float("nan")
        ),
        "full_document_top1_accuracy_percent": (
            100.0 * top1_correct / scored_tokens if scored_tokens else 0.0
        ),
        "full_document_top5_accuracy_percent": (
            100.0 * top5_correct / scored_tokens if scored_tokens else 0.0
        ),
    }

    generated = [
        row for row in records if "prediction" in row and "reference" in row
    ]
    metrics["num_generation_documents"] = len(generated)
    if not generated:
        return metrics

    predictions = [normalize_metric_text(row["prediction"]) for row in generated]
    references = [normalize_metric_text(row["reference"]) for row in generated]
    fields = {
        "rouge1_f1_percent": "rouge1_f1",
        "rouge2_f1_percent": "rouge2_f1",
        "rougeL_chunked_f1_percent": "rougeL_chunked_f1",
        "article_heading_f1_percent": "article_heading_f1",
        "distinct2_percent": "distinct2",
        "repeated_4gram_ratio_percent": "repeated_4gram_ratio",
    }
    metrics.update(
        {
            output: 100.0 * safe_mean(row[source] for row in generated)
            for output, source in fields.items()
        }
    )
    metrics.update(
        {
            "bleu": corpus_bleu(predictions, references),
            "chrf_plus_plus": corpus_chrf_plus_plus(predictions, references),
            "average_prefix_tokens": safe_mean(
                row["prefix_tokens"] for row in generated
            ),
            "average_reference_tokens": safe_mean(
                row["reference_tokens"] for row in generated
            ),
            "average_generated_tokens": safe_mean(
                row["generated_tokens"] for row in generated
            ),
            "average_length_ratio": safe_mean(
                row["length_ratio"] for row in generated
            ),
            "eos_rate_percent": 100.0
            * safe_mean(float(row["ended_with_eos"]) for row in generated),
            "average_generation_seconds": safe_mean(
                row["generation_seconds"] for row in generated
            ),
            "total_generation_seconds": sum(
                row["generation_seconds"] for row in generated
            ),
        }
    )
    return metrics
