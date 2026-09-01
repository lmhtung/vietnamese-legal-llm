"""Typed YAML configuration for Qwen3.5 legal continued pre-training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

try:
    import yaml
except ImportError as error:  # pragma: no cover - environment dependent
    raise ImportError(
        "PyYAML is required to read config.yaml. Install it with `pip install pyyaml`."
    ) from error


QWEN35_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = QWEN35_DIR / "config.yaml"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-4B-Base"
    max_seq_length: int = 20_000
    load_in_4bit: bool = False
    load_in_16bit: bool = True
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    path: Path = Path("datasets/vietnamese_legal_documents_vhtd_p95_20k")
    text_field: str = "text"
    num_proc: int = field(default_factory=lambda: max(1, min(8, os.cpu_count() or 1)))
    tokenize_batch_size: int = 32
    max_train_samples: int | None = None
    max_validation_samples: int | None = None


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = "all-linear"
    use_rslora: bool = False


@dataclass
class OptimizationConfig:
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"


@dataclass
class EvaluationConfig:
    metric_logits_chunk_size: int = 256
    eval_steps: int = 500
    skip_baseline_eval: bool = False


@dataclass
class CheckpointConfig:
    output_dir: Path = Path("outputs/qwen35_4b_legal_cpt_p95_20k_r16")
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None


@dataclass
class RuntimeConfig:
    seed: int = 3407
    logging_steps: int = 5
    disable_tqdm: bool = False
    pad_to_multiple_of: int = 8


@dataclass
class ExportConfig:
    merge_after_train: bool = False
    merged_output_dir: Path | None = None


@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def resolve_paths(self, base_dir: Path) -> None:
        self.data.path = _resolve_path(self.data.path, base_dir)
        self.checkpoint.output_dir = _resolve_path(
            self.checkpoint.output_dir, base_dir
        )
        if self.export.merged_output_dir is not None:
            self.export.merged_output_dir = _resolve_path(
                self.export.merged_output_dir, base_dir
            )

    def validate(self) -> None:
        positive_ints = {
            "model.max_seq_length": self.model.max_seq_length,
            "data.num_proc": self.data.num_proc,
            "data.tokenize_batch_size": self.data.tokenize_batch_size,
            "lora.r": self.lora.r,
            "lora.alpha": self.lora.alpha,
            "optimization.train_batch_size": self.optimization.train_batch_size,
            "optimization.eval_batch_size": self.optimization.eval_batch_size,
            "optimization.gradient_accumulation_steps": (
                self.optimization.gradient_accumulation_steps
            ),
            "evaluation.metric_logits_chunk_size": (
                self.evaluation.metric_logits_chunk_size
            ),
            "evaluation.eval_steps": self.evaluation.eval_steps,
            "checkpoint.save_total_limit": self.checkpoint.save_total_limit,
            "runtime.logging_steps": self.runtime.logging_steps,
            "runtime.pad_to_multiple_of": self.runtime.pad_to_multiple_of,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if (
            self.optimization.num_train_epochs <= 0
            and self.optimization.max_steps <= 0
        ):
            raise ValueError(
                "optimization.num_train_epochs must be > 0 when max_steps is not positive"
            )
        if not 0.0 <= self.optimization.warmup_ratio < 1.0:
            raise ValueError("optimization.warmup_ratio must be in [0, 1)")
        if self.lora.dropout < 0.0:
            raise ValueError("lora.dropout must be >= 0")
        if self.optimization.learning_rate <= 0.0:
            raise ValueError("optimization.learning_rate must be > 0")
        if self.optimization.weight_decay < 0.0:
            raise ValueError("optimization.weight_decay must be >= 0")
        if self.optimization.max_grad_norm <= 0.0:
            raise ValueError("optimization.max_grad_norm must be > 0")
        if not self.model.name.strip():
            raise ValueError("model.name must not be empty")
        if not self.data.text_field.strip():
            raise ValueError("data.text_field must not be empty")
        if self.lora.target_modules not in {"all-linear", "standard"}:
            raise ValueError("lora.target_modules must be 'all-linear' or 'standard'")
        if self.optimization.lr_scheduler_type not in {
            "linear", "cosine", "constant", "constant_with_warmup",
        }:
            raise ValueError(
                "optimization.lr_scheduler_type must be linear, cosine, constant, "
                "or constant_with_warmup"
            )
        if self.model.load_in_4bit == self.model.load_in_16bit:
            raise ValueError(
                "Exactly one of model.load_in_4bit and model.load_in_16bit must be true"
            )
        for name, value in {
            "data.max_train_samples": self.data.max_train_samples,
            "data.max_validation_samples": self.data.max_validation_samples,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be null or > 0")

    def to_namespace(self, config_path: Path) -> argparse.Namespace:
        """Flatten YAML sections to the names consumed by finetune.py."""
        return argparse.Namespace(
            config_path=config_path,
            model_name=self.model.name,
            max_seq_length=self.model.max_seq_length,
            load_in_4bit=self.model.load_in_4bit,
            load_in_16bit=self.model.load_in_16bit,
            trust_remote_code=self.model.trust_remote_code,
            dataset_path=self.data.path,
            text_field=self.data.text_field,
            dataset_num_proc=self.data.num_proc,
            tokenize_batch_size=self.data.tokenize_batch_size,
            max_train_samples=self.data.max_train_samples,
            max_validation_samples=self.data.max_validation_samples,
            lora_r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            target_modules=self.lora.target_modules,
            use_rslora=self.lora.use_rslora,
            train_batch_size=self.optimization.train_batch_size,
            eval_batch_size=self.optimization.eval_batch_size,
            gradient_accumulation_steps=self.optimization.gradient_accumulation_steps,
            num_train_epochs=self.optimization.num_train_epochs,
            max_steps=self.optimization.max_steps,
            learning_rate=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
            warmup_ratio=self.optimization.warmup_ratio,
            lr_scheduler_type=self.optimization.lr_scheduler_type,
            max_grad_norm=self.optimization.max_grad_norm,
            optim=self.optimization.optim,
            metric_logits_chunk_size=self.evaluation.metric_logits_chunk_size,
            eval_steps=self.evaluation.eval_steps,
            skip_baseline_eval=self.evaluation.skip_baseline_eval,
            output_dir=self.checkpoint.output_dir,
            save_total_limit=self.checkpoint.save_total_limit,
            resume_from_checkpoint=self.checkpoint.resume_from_checkpoint,
            seed=self.runtime.seed,
            logging_steps=self.runtime.logging_steps,
            disable_tqdm=self.runtime.disable_tqdm,
            pad_to_multiple_of=self.runtime.pad_to_multiple_of,
            merge_after_train=self.export.merge_after_train,
            merged_output_dir=self.export.merged_output_dir,
        )


ConfigSection = TypeVar("ConfigSection")
SECTION_TYPES: dict[str, type[Any]] = {
    "model": ModelConfig,
    "data": DataConfig,
    "lora": LoraConfig,
    "optimization": OptimizationConfig,
    "evaluation": EvaluationConfig,
    "checkpoint": CheckpointConfig,
    "runtime": RuntimeConfig,
    "export": ExportConfig,
}


def _resolve_path(path: Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _load_section(
    section_type: type[ConfigSection], section_name: str, values: Any
) -> ConfigSection:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError(f"Config section {section_name!r} must be a mapping")
    valid_keys = {item.name for item in fields(section_type)}
    unknown = set(values) - valid_keys
    if unknown:
        raise KeyError(
            f"Unknown keys in config section {section_name!r}: {sorted(unknown)}"
        )
    return section_type(**values)


def load_finetune_config(path: Path) -> FinetuneConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError("The YAML root must be a mapping")

    unknown_sections = set(payload) - set(SECTION_TYPES)
    if unknown_sections:
        raise KeyError(f"Unknown config sections: {sorted(unknown_sections)}")
    sections = {
        name: _load_section(section_type, name, payload.get(name))
        for name, section_type in SECTION_TYPES.items()
    }
    config = FinetuneConfig(**sections)
    config.resolve_paths(path.parent)
    config.validate()
    return config


def _add_bool_override(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )


def _build_override_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Qwen3.5 text CPT. Explicit CLI values override YAML.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    scalar_arguments: tuple[tuple[str, type[Any]], ...] = (
        ("model-name", str), ("max-seq-length", int),
        ("dataset-path", Path), ("text-field", str),
        ("dataset-num-proc", int), ("tokenize-batch-size", int),
        ("max-train-samples", int), ("max-validation-samples", int),
        ("lora-r", int), ("lora-alpha", int), ("lora-dropout", float),
        ("target-modules", str), ("train-batch-size", int),
        ("eval-batch-size", int), ("gradient-accumulation-steps", int),
        ("num-train-epochs", float), ("max-steps", int),
        ("learning-rate", float), ("weight-decay", float),
        ("warmup-ratio", float), ("lr-scheduler-type", str),
        ("max-grad-norm", float), ("optim", str),
        ("metric-logits-chunk-size", int), ("eval-steps", int),
        ("output-dir", Path), ("save-total-limit", int),
        ("resume-from-checkpoint", str), ("seed", int),
        ("logging-steps", int), ("pad-to-multiple-of", int),
        ("merged-output-dir", Path),
    )
    for name, value_type in scalar_arguments:
        parser.add_argument(
            f"--{name}", dest=name.replace("-", "_"), type=value_type,
            default=argparse.SUPPRESS,
        )
    for name in (
        "load_in_4bit", "load_in_16bit", "trust_remote_code", "use_rslora",
        "skip_baseline_eval", "disable_tqdm", "merge_after_train",
    ):
        _add_bool_override(parser, name)
    return parser


def parse_finetune_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Load YAML, apply explicit CLI overrides, resolve paths, and validate."""
    cli = _build_override_parser().parse_args(argv)
    config_path = cli.config.expanduser().resolve()
    args = load_finetune_config(config_path).to_namespace(config_path)
    overrides = vars(cli).copy()
    overrides.pop("config", None)
    for name, value in overrides.items():
        if name in {"dataset_path", "output_dir", "merged_output_dir"}:
            value = value.expanduser().resolve()
        setattr(args, name, value)
    namespace_to_config(args).validate()
    return args


def namespace_to_config(args: argparse.Namespace) -> FinetuneConfig:
    """Convert the flat namespace back to the typed schema for validation."""
    return FinetuneConfig(
        model=ModelConfig(
            name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            load_in_16bit=args.load_in_16bit,
            trust_remote_code=args.trust_remote_code,
        ),
        data=DataConfig(
            path=args.dataset_path,
            text_field=args.text_field,
            num_proc=args.dataset_num_proc,
            tokenize_batch_size=args.tokenize_batch_size,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
        ),
        lora=LoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.target_modules,
            use_rslora=args.use_rslora,
        ),
        optimization=OptimizationConfig(
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_grad_norm=args.max_grad_norm,
            optim=args.optim,
        ),
        evaluation=EvaluationConfig(
            metric_logits_chunk_size=args.metric_logits_chunk_size,
            eval_steps=args.eval_steps,
            skip_baseline_eval=args.skip_baseline_eval,
        ),
        checkpoint=CheckpointConfig(
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        runtime=RuntimeConfig(
            seed=args.seed,
            logging_steps=args.logging_steps,
            disable_tqdm=args.disable_tqdm,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
        export=ExportConfig(
            merge_after_train=args.merge_after_train,
            merged_output_dir=args.merged_output_dir,
        ),
    )
