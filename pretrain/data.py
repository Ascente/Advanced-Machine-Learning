"""Memory-mapped token streams, packed into fixed-length next-token examples."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    def __init__(self, directory: str | Path, split: str):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.context_length = self.metadata["context_length"]
        self.bos_token_id = self.metadata["bos_token_id"]
        self.pad_token_id = self.metadata["pad_token_id"]
        self.tokens = np.memmap(directory / f"{split}.bin", dtype="<u2", mode="r")
        self.byte_counts = np.memmap(directory / f"{split}.bytes.bin", dtype="<u8", mode="r")
        self.num_blocks = math.ceil((len(self.tokens) - 1) / self.context_length)
        if self.num_blocks != len(self.byte_counts) or self.num_blocks == 0:
            raise ValueError(f"Invalid or empty packed {split} data")

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        start = int(index) * self.context_length
        chunk = np.full(self.context_length + 1, self.pad_token_id, dtype=np.int64)
        available = self.tokens[start:start + self.context_length + 1]
        chunk[:len(available)] = available
        inputs = torch.from_numpy(chunk[:-1].copy())
        targets = torch.from_numpy(chunk[1:].copy())
        # BOS belongs to the new document. A chunk may begin inside a document.
        document_ids = (inputs == self.bos_token_id).long().cumsum(dim=-1)
        # Predict text and EOS from this document, never the next document's BOS.
        targets[targets == self.bos_token_id] = self.pad_token_id
        return inputs, targets, document_ids, int(self.byte_counts[index])
