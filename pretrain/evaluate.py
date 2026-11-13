"""Token-weighted loss, perplexity and original-byte BPB on packed data."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from .data import PackedDataset
from .model import ModelConfig, TransformerLM


def metrics_from_totals(loss_sum, tokens, raw_bytes):
    if tokens <= 0 or raw_bytes <= 0:
        raise ValueError("Evaluation needs scored tokens and original text bytes")
    loss = loss_sum / tokens
    return {"loss": loss, "ppl": math.exp(loss), "bpb": loss_sum / (math.log(2) * raw_bytes),
            "tokens": int(tokens), "utf8_bytes": int(raw_bytes)}


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=8, precision="bf16", max_batches=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    limit = len(dataset) if max_batches is None else min(len(dataset), max_batches * batch_size)
    loader = DataLoader(Subset(dataset, range(rank, limit, world_size)), batch_size=batch_size,
                        pin_memory=device.type == "cuda")
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    try:
        for inputs, targets, document_ids, raw_bytes in tqdm(loader, desc="Validation", leave=False, disable=rank != 0):
            inputs, targets, document_ids = inputs.to(device), targets.to(device), document_ids.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(inputs, document_ids=document_ids)
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                           ignore_index=dataset.pad_token_id, reduction="sum")
            totals[0] += loss_sum.double()
            totals[1] += (targets != dataset.pad_token_id).sum()
            totals[2] += raw_bytes.sum().to(device)
        if dist.is_initialized():
            dist.all_reduce(totals)
    finally:
        model.train(was_training)
    result = metrics_from_totals(*totals.tolist())
    result.update({"blocks": limit, "full_split": limit == len(dataset)})
    return result


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    return model, tokenizer, checkpoint


@torch.inference_mode()
def generate_samples(model, tokenizer, device, max_new_tokens=100, precision="bf16"):
    prompts = ["Once upon a time", "The solar system consists of", "Education is important because"]
    samples = []
    devices = [device] if device.type == "cuda" else []
    # Generation must not change the random state used for the training run.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        for prompt in prompts:
            ids = [tokenizer.token_to_id("<s>")] + tokenizer.encode(prompt, add_special_tokens=False).ids
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model.generate(torch.tensor([ids], device=device), max_new_tokens=max_new_tokens,
                                        temperature=0.8, top_k=50, eos_token_id=tokenizer.token_to_id("</s>"),
                                        pad_token_id=tokenizer.token_to_id("<pad>"))
            samples.append({"prompt": prompt, "text": tokenizer.decode(output[0].tolist(), skip_special_tokens=True)})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    device = torch.device(args.device)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    data_dir = args.data_dir or Path(checkpoint["train_config"]["data_dir"])
    dataset = PackedDataset(data_dir, args.split)
    if dataset.metadata != checkpoint["data_metadata"]:
        raise ValueError("Packed dataset metadata differs from the checkpoint")
    metrics = evaluate(model, dataset, device, args.batch_size, args.precision)
    metrics.update({"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"],
                    "split": args.split, "tokenizer": dataset.metadata["tokenizer"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    samples = generate_samples(model, tokenizer, device, precision=args.precision)
    (args.output_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
