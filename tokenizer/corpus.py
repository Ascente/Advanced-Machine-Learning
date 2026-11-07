"""Local Parquet readers: no network access, no train/validation mixing."""

from pathlib import Path

import pyarrow.parquet as pq


def split_files(data_dir: Path, split: str):
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    files = sorted((data_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} Parquet files in {data_dir / 'data'}")
    return files


def split_inventory(data_dir: Path, split: str):
    return [{"path": str(path.relative_to(data_dir)),
             "num_rows": pq.ParquetFile(path).metadata.num_rows}
            for path in split_files(data_dir, split)]


def iter_documents(data_dir: Path, split: str):
    """Stream the entire split in sorted file/row order; never sample or shuffle."""
    files = split_files(data_dir, split)
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise TypeError(f"Non-string document in {path}")
                yield text

