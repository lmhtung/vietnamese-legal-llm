#!/usr/bin/env python3
"""Tải Vietnamese Legal Documents từ Hugging Face về local."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from datasets import load_dataset


DEFAULT_REPO_ID = "vohuutridung/vietnamese-legal-documents"
DEFAULT_OUTPUT_DIR = "datasets/vietnamese_legal_documents_vhtd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải hai config metadata và content rồi lưu bằng save_to_disk()."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset ID (mặc định: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu dữ liệu (mặc định: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Thư mục Hugging Face cache; mặc định là <output-dir>/hf_cache.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Kích thước tối đa mỗi Arrow shard khi save_to_disk().",
    )
    return parser.parse_args()


def download_and_save_config(
    repo_id: str,
    config_name: str,
    output_path: Path,
    cache_dir: Path,
    max_shard_size: str,
) -> None:
    if output_path.exists():
        print(f"[SKIP] Đường dẫn đã tồn tại: {output_path}")
        print("       Chọn --output-dir khác nếu muốn tải một bản mới.")
        return

    print("\n" + "=" * 80)
    print(f"Đang tải config: {config_name}")
    print(f"Repository      : {repo_id}")

    dataset = load_dataset(
        repo_id,
        config_name,
        split="data",
        cache_dir=str(cache_dir),
    )

    print(f"Số dòng         : {len(dataset):,}")
    print(f"Các cột         : {dataset.column_names}")
    print(f"Đang lưu tại    : {output_path}")

    dataset.save_to_disk(
        str(output_path),
        max_shard_size=max_shard_size,
    )

    print(f"[OK] Đã lưu config '{config_name}'")

    del dataset
    gc.collect()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output_dir / "hf_cache"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository : {args.repo_id}")
    print(f"Output dir : {output_dir}")
    print(f"Cache dir  : {cache_dir}")

    # Metadata nhỏ hơn nên tải trước để kiểm tra kết nối và schema nhanh.
    download_and_save_config(
        repo_id=args.repo_id,
        config_name="metadata",
        output_path=output_dir / "metadata",
        cache_dir=cache_dir,
        max_shard_size=args.max_shard_size,
    )

    # Content chứa toàn bộ nội dung văn bản và có kích thước lớn hơn.
    download_and_save_config(
        repo_id=args.repo_id,
        config_name="content",
        output_path=output_dir / "content",
        cache_dir=cache_dir,
        max_shard_size=args.max_shard_size,
    )

    print("\n" + "=" * 80)
    print("HOÀN TẤT")
    print(f"Metadata: {output_dir / 'metadata'}")
    print(f"Content : {output_dir / 'content'}")
    print(f"Cache   : {cache_dir}")


if __name__ == "__main__":
    main()