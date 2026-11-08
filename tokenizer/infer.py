"""Load a trained BPE tokenizer and demonstrate encoding and decoding."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

if __package__:
    from .bpe import BPETokenizer
else:
    from bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="tinystories-bpe-16000",
                        help="Custom model directory or flexitok/bpe_fw_edu_16000")
    parser.add_argument("--text", default="Once upon a time, a little cat found a red ball.")
    args = parser.parse_args()

    if args.model_dir == "flexitok/bpe_fw_edu_16000":
        model = Tokenizer.from_pretrained(args.model_dir)
        model.no_padding()
        model.no_truncation()
        ids = model.encode(args.text, add_special_tokens=False).ids
        decoded = model.decode(ids, skip_special_tokens=False)
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            model_dir = Path(__file__).resolve().parent / model_dir
        model = BPETokenizer.load(model_dir)
        ids = model.encode(args.text)
        decoded = model.decode(ids)
    print(json.dumps({
        "text": args.text,
        "ids": ids,
        "decoded": decoded,
        "is_match": decoded == args.text,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
