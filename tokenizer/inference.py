"""Load a trained BPE tokenizer and demonstrate encoding and decoding."""

import argparse
import json
from pathlib import Path

if __package__:
    from .bpe import BPETokenizer
else:
    from bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent / "tinystories-bpe-16000")
    parser.add_argument("--text", default="Once upon a time, a little cat found a red ball.")
    args = parser.parse_args()

    model = BPETokenizer.load(args.model_dir)
    ids = model.encode(args.text)
    decoded = model.decode(ids)
    print(json.dumps({
        "text": args.text,
        "ids": ids,
        "decoded": decoded,
        "is_match": decoded == args.text,
    }, ensure_ascii=False, indent=2))
    if decoded != args.text:
        raise RuntimeError("Decoded text differs from input")


if __name__ == "__main__":
    main()
