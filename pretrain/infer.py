"""Generate text from a pretraining checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .evaluate import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not ids:
        parser.error("prompt must encode to at least one token")
    ids = [tokenizer.token_to_id("<s>")] + ids
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        output = model.generate(torch.tensor([ids], device=device), max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k,
                                eos_token_id=tokenizer.token_to_id("</s>"), pad_token_id=tokenizer.token_to_id("<pad>"))
    result = {"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"], "prompt": args.prompt,
              "text": tokenizer.decode(output[0].tolist(), skip_special_tokens=True)}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
