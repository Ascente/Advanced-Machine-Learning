"""Train custom BPE on the complete local TinyStories training split."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

if __package__:
    from .bpe import BPETokenizer
    from .corpus import iter_documents, split_inventory
else:
    from bpe import BPETokenizer
    from corpus import iter_documents, split_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/user-data/chaojh26/TinyStories"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "tinystories-bpe-16000")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    inventory = split_inventory(args.data_dir, "train")
    expected_documents = sum(entry["num_rows"] for entry in inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "tokenizer.json").exists():
        parser.error("Output already contains a tokenizer; choose a new --output-dir")
    with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as log:
        def progress(event):
            event = {**event, "elapsed_seconds": round(time.perf_counter() - started, 3)}
            line = json.dumps(event)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        progress({"stage": "input", "scope": "full_train_split",
                  "expected_documents": expected_documents, "files": inventory})
        model = BPETokenizer()
        stats = model.train(iter_documents(args.data_dir, "train"),
                            args.vocab_size, args.min_frequency, progress)
        if stats["documents"] != expected_documents:
            raise RuntimeError("Training did not consume every document in the train split")
        stats.update({"dataset": "roneneldan/TinyStories", "split": "train",
                      "corpus_scope": "full_train_split", "data_files": inventory,
                      "expected_documents": expected_documents,
                      "data_dir": str(args.data_dir.resolve()),
                      "created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": round(time.perf_counter() - started, 3)})
        model.training_stats = stats
        model.save(args.output_dir)
        progress({"stage": "finished", **stats})
        print(f"Saved tokenizer to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
