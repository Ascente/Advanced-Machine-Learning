"""Local Parquet readers: no network access, no train/validation mixing."""

from pathlib import Path
import random

import pyarrow.parquet as pq


DEFAULT_DATA_DIRS = {
    "tinystories": Path("/user-data/chaojh26/TinyStories"),
    "fineweb": Path("/user-data/chaojh26/FineWeb-Edu/sample-10BT"),
}


def split_files(data_dir: Path, split: str, dataset: str = "tinystories"):
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if dataset == "tinystories":
        files = sorted((data_dir / "data").glob(f"{split}-*.parquet"))
    elif dataset == "fineweb":
        files = sorted(data_dir.glob("*.parquet"))
    else:
        raise ValueError("dataset must be tinystories or fineweb")
    if not files:
        raise FileNotFoundError(f"No {dataset} {split} Parquet files in {data_dir}")
    return files


def split_inventory(data_dir: Path, split: str, dataset: str = "tinystories"):
    return [{"path": str(path.relative_to(data_dir)),
             "num_rows": pq.ParquetFile(path).metadata.num_rows}
            for path in split_files(data_dir, split, dataset)]


def iter_document_batches(data_dir: Path, split: str, dataset: str = "tinystories", seed: int = 42):
    """FineWeb randomly reserves 1% of document indices using the given seed."""
    files = split_files(data_dir, split, dataset)
    if dataset == "fineweb":
        total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        validation_indices = set(random.Random(seed).sample(range(total), total // 100))
    offset = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if dataset == "fineweb":
                texts = [text for index, text in enumerate(texts, offset)
                         if (index in validation_indices) == (split == "validation")]
                offset += len(batch)
            if not all(isinstance(text, str) for text in texts):
                raise TypeError(f"Non-string document in {path}")
            if texts:
                yield texts


def iter_documents(data_dir: Path, split: str, dataset: str = "tinystories", seed: int = 42):
    for texts in iter_document_batches(data_dir, split, dataset, seed):
        yield from texts

