"""Measure held-out compression and verify exact text reconstruction."""

import argparse
from itertools import islice
import json
from pathlib import Path
import time

if __package__:
    from .bpe import BPETokenizer
    from .corpus import iter_documents
else:
    from bpe import BPETokenizer
    from corpus import iter_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent / "tinystories-bpe-16000")
    parser.add_argument("--data-dir", type=Path, default=Path("/user-data/chaojh26/TinyStories"))
    parser.add_argument("--max-documents", type=int, default=1000)
    args = parser.parse_args()
    if args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    model = BPETokenizer.load(args.model_dir)
    num_bytes = num_tokens = num_characters = documents = 0
    examples = []
    started = time.perf_counter()
    for text in islice(iter_documents(args.data_dir, "validation"), args.max_documents):
        ids = model.encode(text)
        if model.decode(ids) != text:
            raise AssertionError(f"Roundtrip failed for validation document {documents}")
        if any(token < 0 or token >= model.vocab_size for token in ids):
            raise AssertionError("Token out of vocabulary")
        documents += 1
        num_bytes += len(text.encode("utf-8"))
        num_characters += len(text)
        num_tokens += len(ids)
        if len(examples) < 3:
            examples.append({"text": text, "ids": ids, "is_match": True})
    if not num_tokens:
        raise ValueError("No validation tokens")
    seconds = time.perf_counter() - started
    metrics = {
        "split": "validation", "documents": documents,
        "vocab_size": model.vocab_size, "utf8_bytes": num_bytes,
        "unicode_characters": num_characters, "tokens": num_tokens,
        "bytes_per_token": num_bytes / num_tokens,
        "characters_per_token": num_characters / num_tokens,
        "roundtrip_passed": documents, "elapsed_seconds": seconds,
    }
    (args.model_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.model_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
