"""Evaluate a complete dataset split using custom BPE or a reference tokenizer."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from tokenizers import Tokenizer

if __package__:
    from .bpe import BPETokenizer
    from .corpus import DEFAULT_DATA_DIRS, iter_document_batches, split_inventory
else:
    from bpe import BPETokenizer
    from corpus import DEFAULT_DATA_DIRS, iter_document_batches, split_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="tinystories-bpe-16000",
                        help="Custom model directory or flexitok/bpe_fw_edu_16000")
    parser.add_argument("--dataset", type=str.lower, choices=["tinystories", "fineweb"], default="tinystories")
    parser.add_argument("--data-dir", type=Path, help="Defaults to the selected dataset's local directory")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the FineWeb 99:1 split")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir or DEFAULT_DATA_DIRS[args.dataset]
    inventory = split_inventory(data_dir, args.split, args.dataset)
    source_documents = sum(entry["num_rows"] for entry in inventory)
    if args.dataset == "fineweb":
        validation_documents = source_documents // 100
        expected_documents = validation_documents if args.split == "validation" else source_documents - validation_documents
        split_policy = {
            "method": "random_document_indices",
            "seed": args.seed,
            "validation_ratio": 0.01,
            "train_documents": source_documents - validation_documents,
            "validation_documents": validation_documents,
        }
    else:
        expected_documents = source_documents
        split_policy = {"method": "official"}
    if args.model_dir == "flexitok/bpe_fw_edu_16000":
        model = Tokenizer.from_pretrained(args.model_dir)
        model.no_padding()
        model.no_truncation()
        vocab_size = model.get_vocab_size()
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            model_dir = Path(__file__).resolve().parent / model_dir
        model = BPETokenizer.load(model_dir)
        vocab_size = model.vocab_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_bytes = num_tokens = num_characters = documents = 0
    matched_documents = normalized_matched_documents = 0
    examples = []
    started = time.perf_counter()
    print(f"Evaluating {args.dataset}/{args.split}: {expected_documents:,} documents, "
          f"tokenizer={args.model_dir}", flush=True)
    next_progress = 100_000
    for texts in iter_document_batches(data_dir, args.split, args.dataset, args.seed):
        if args.model_dir == "flexitok/bpe_fw_edu_16000":
            sequences = [encoding.ids for encoding in model.encode_batch(texts, add_special_tokens=False)]
            decoded_texts = model.decode_batch(sequences, skip_special_tokens=False)
            normalized_texts = [model.normalizer.normalize_str(text) for text in texts] if model.normalizer else texts
        else:
            sequences = [model.encode(text) for text in texts]
            decoded_texts = [model.decode(ids) for ids in sequences]
            normalized_texts = texts
        for text, ids, decoded, normalized in zip(texts, sequences, decoded_texts, normalized_texts):
            is_match = decoded == text
            matched_documents += is_match
            normalized_matched_documents += decoded == normalized
            documents += 1
            num_bytes += len(text.encode("utf-8"))
            num_characters += len(text)
            num_tokens += len(ids)
            if len(examples) < 3:
                examples.append({"text": text, "ids": ids, "decoded": decoded, "is_match": is_match})
        if documents >= next_progress:
            print(f"Documents: {documents:,}/{expected_documents:,}, tokens: {num_tokens:,}", flush=True)
            next_progress = (documents // 100_000 + 1) * 100_000
    if documents != expected_documents:
        raise RuntimeError(f"Expected {expected_documents} documents, processed {documents}")
    seconds = time.perf_counter() - started
    metrics = {
        "dataset": args.dataset, "data_dir": str(data_dir.resolve()),
        "data_files": inventory, "split": args.split, "split_policy": split_policy,
        "tokenizer": {"source": args.model_dir, "vocab_size": vocab_size},
        "expected_documents": expected_documents,
        "documents": documents, "utf8_bytes": num_bytes,
        "unicode_characters": num_characters, "tokens": num_tokens,
        "bytes_per_token": num_bytes / num_tokens if num_tokens else None,
        "characters_per_token": num_characters / num_tokens if num_tokens else None,
        "matched_documents": matched_documents,
        "mismatched_documents": documents - matched_documents,
        "match_rate": matched_documents / documents if documents else None,
        "normalized_matched_documents": normalized_matched_documents,
        "elapsed_seconds": seconds,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
