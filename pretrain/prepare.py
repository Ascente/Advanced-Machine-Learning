"""Encode local TinyStories or FineWeb documents into packed token streams."""

import argparse
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("RAYON_NUM_THREADS", "8")

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm.auto import tqdm


class TokenWriter:
    def __init__(self, directory: Path, split: str, context_length: int, bos_token_id: int, eos_token_id: int):
        self.tokens_file = (directory / f"{split}.bin").open("wb")
        self.bytes_file = (directory / f"{split}.bytes.bin").open("wb")
        self.context_length = context_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.documents = self.tokens = self.raw_bytes = 0
        self.block_bytes = self.scored_bytes = self.blocks = 0

    def _finish_block(self):
        self.bytes_file.write(np.asarray([self.block_bytes], dtype="<u8").tobytes())
        self.scored_bytes += self.block_bytes
        self.block_bytes = 0
        self.blocks += 1

    def append(self, text, encoding):
        ids, offsets = encoding.ids, encoding.offsets
        self.tokens_file.write(np.asarray([self.bos_token_id] + ids + [self.eos_token_id], dtype="<u2").tobytes())
        self.documents += 1
        self.raw_bytes += len(text.encode("utf-8"))

        # Attribute each original character's bytes once, even when several byte
        # tokens share its offsets. The final piece includes any NFC-normalized tail.
        cursor = 0
        # BOS/EOS can fall exactly on a packed-block boundary, but add no bytes.
        if self.tokens > 0 and self.tokens % self.context_length == 0:
            self._finish_block()
        index = 0
        while index < len(ids):
            global_index = self.tokens + 1 + index
            boundary = ((global_index - 1) // self.context_length + 1) * self.context_length
            end = min(len(ids), boundary - self.tokens)
            char_end = len(text) if end == len(ids) else max(cursor, offsets[end - 1][1])
            self.block_bytes += len(text[cursor:char_end].encode("utf-8"))
            cursor = char_end
            if self.tokens + end == boundary:
                self._finish_block()
            index = end
        eos_index = self.tokens + 1 + len(ids)
        if eos_index > 0 and eos_index % self.context_length == 0:
            self._finish_block()
        self.tokens += len(ids) + 2

    def finish(self):
        if (self.tokens - 1) % self.context_length:
            self._finish_block()
        self.tokens_file.close()
        self.bytes_file.close()
        if self.tokens < 2 or self.blocks != math.ceil((self.tokens - 1) / self.context_length):
            raise ValueError("Each split needs at least two tokens")
        if self.scored_bytes != self.raw_bytes:
            raise ValueError("Packed byte counts do not match the original UTF-8 byte total")
        return {
            "documents": self.documents, "tokens": self.tokens, "blocks": self.blocks,
            "raw_utf8_bytes": self.raw_bytes, "scored_utf8_bytes": self.scored_bytes,
            "scored_tokens": self.tokens - self.documents,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "fineweb"], default="tinystories")
    parser.add_argument("--data-dir", type=Path, default=Path("/user-data/chaojh26/TinyStories"))
    parser.add_argument("--output-dir", type=Path, default=Path("/user-data/chaojh26/TinyStories/packed-1024"))
    parser.add_argument("--tokenizer", default="flexitok/bpe_fw_edu_16000")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.context_length < 1 or args.batch_size < 1:
        parser.error("context-length and batch-size must be positive")
    if args.dataset == "tinystories":
        file_splits = []
        for split in ("train", "validation"):
            paths = sorted((args.data_dir / "data").glob(f"{split}-*.parquet"))
            if not paths:
                raise FileNotFoundError(f"No TinyStories {split} files in {args.data_dir / 'data'}")
            file_splits.extend((path, split) for path in paths)
    else:
        file_splits = [(path, None) for path in sorted(args.data_dir.glob("*.parquet"))]
        if not file_splits:
            raise FileNotFoundError(f"No Parquet files in {args.data_dir}")
    files = [path for path, _ in file_splits]
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if args.dataset == "fineweb" and total < 100:
        raise ValueError("At least 100 documents are needed for the 99:1 split")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(source)) if source.is_file() else Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    # Literal special-token strings in web text are ordinary text. Add boundaries ourselves.
    tokenizer.encode_special_tokens = True
    bos, eos, pad = (tokenizer.token_to_id(token) for token in ("<s>", "</s>", "<pad>"))
    if None in (bos, eos, pad) or len({bos, eos, pad}) != 3 or tokenizer.get_vocab_size() > 65536:
        raise ValueError("Expected distinct BOS/EOS/PAD and a vocabulary fitting uint16")
    tokenizer.save(str(args.output_dir / "tokenizer.json"))
    validation_indices = (set(random.Random(args.seed).sample(range(total), total // 100))
                          if args.dataset == "fineweb" else set())
    writers = {split: TokenWriter(args.output_dir, split, args.context_length, bos, eos)
               for split in ("train", "validation")}
    offset = 0
    with tqdm(total=total, desc=f"Encode {args.dataset}", unit="doc") as progress:
        for path, official_split in file_splits:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size, columns=["text"]):
                texts = batch.column(0).to_pylist()
                if not all(isinstance(text, str) for text in texts):
                    raise TypeError(f"Non-string document in {path}")
                encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
                for index, (text, encoding) in enumerate(zip(texts, encodings), offset):
                    split = official_split or ("validation" if index in validation_indices else "train")
                    writers[split].append(text, encoding)
                offset += len(texts)
                progress.update(len(texts))

    metadata = {
        "dataset": "TinyStories" if args.dataset == "tinystories" else "FineWeb-Edu sample-10BT",
        "data_dir": str(args.data_dir.resolve()),
        "files": [str(p.relative_to(args.data_dir)) for p in files], "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.get_vocab_size(), "bos_token_id": bos, "eos_token_id": eos, "pad_token_id": pad,
        "context_length": args.context_length, "seed": args.seed if args.dataset == "fineweb" else None,
        "split_policy": ("official" if args.dataset == "tinystories" else
                         "random.Random(seed).sample(range(total_documents), total_documents // 100)"),
        "encoding_policy": "Concatenate BOS + text + EOS per document; causal attention restricted to each document; ignore BOS/PAD targets",
        "byte_policy": "All original UTF-8 text bytes attributed by tokenizer offsets; BOS/EOS/PAD have zero bytes",
        "splits": {split: writer.finish() for split, writer in writers.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
