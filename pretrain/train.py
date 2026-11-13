"""Language model pretraining with full validation, DDP, tqdm and W&B."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from .data import PackedDataset
from .evaluate import evaluate, generate_samples, metrics_from_totals
from .model import ModelConfig, TransformerLM


def learning_rate(step, total_steps, peak, warmup_ratio, min_ratio):
    """step is zero-based; the final optimizer update uses peak * min_ratio."""
    warmup = min(total_steps, max(1, round(total_steps * warmup_ratio)))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup + 1) / max(1, total_steps - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def epoch_indices(size, seed, epoch, rank, world_size, batch_size, accumulation, completed_steps=0):
    generator = torch.Generator().manual_seed(seed + epoch)
    global_batch = world_size * batch_size * accumulation
    usable = size // global_batch * global_batch
    order = torch.randperm(size, generator=generator)[:usable]
    local = order.view(-1, world_size, batch_size)[:, rank, :].reshape(-1)
    return local[completed_steps * accumulation * batch_size:]


def save_checkpoint(path, model, optimizer, state, device, is_best=False):
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
    states = [None] * dist.get_world_size() if dist.is_initialized() else [rng]
    if dist.is_initialized():
        dist.all_gather_object(states, rng)
    if not dist.is_initialized() or dist.get_rank() == 0:
        payload = {**state, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_states": states}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        if is_best:
            best = path.parent / "best.pt"
            shutil.copyfile(path, best.with_suffix(".pt.tmp"))
            best.with_suffix(".pt.tmp").replace(best)
    if dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/user-data/chaojh26/TinyStories/packed-1024"))
    parser.add_argument("--output-dir", type=Path, default=Path("/user-data/chaojh26/pretrain/tinystories-234m"))
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16, help="Sequences per GPU per microbatch")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Optional cap on total optimizer updates")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-tokens", type=int, default=100)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="aml-hw1")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--run-name", default="tinystories-234m")
    args = parser.parse_args()
    for name in ("batch_size", "gradient_accumulation", "epochs", "eval_every", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if not 0 < args.warmup_ratio < 1 or not 0 < args.min_lr_ratio <= 1:
        parser.error("warmup-ratio must be in (0, 1) and min-lr-ratio in (0, 1]")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0 or args.sample_tokens < 0:
        parser.error("Invalid learning rate, weight decay, clipping or sample length")
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    # Authenticate before creating NCCL groups so other ranks cannot time out
    # while rank 0 waits for an interactive W&B login prompt.
    if rank == 0 and args.wandb_mode == "online":
        import wandb
        if not wandb.login(prompt=False, verify=True):
            raise RuntimeError(
                "W&B credentials are not configured. Run 'wandb login --verify' "
                "in the same environment before torchrun, or use --wandb-mode offline/disabled."
            )
    device = torch.device(args.device)
    if world_size > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    elif device.type == "cuda":
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_data, val_data = PackedDataset(args.data_dir, "train"), PackedDataset(args.data_dir, "validation")
    metadata = train_data.metadata
    config = ModelConfig(vocab_size=metadata["vocab_size"], context_length=metadata["context_length"],
                         d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff)
    steps_per_epoch = len(train_data) // (args.batch_size * world_size * args.gradient_accumulation)
    total_steps = min(args.max_steps or math.inf, args.epochs * steps_per_epoch)
    if total_steps < 1:
        raise ValueError("Not enough packed blocks for one effective batch; reduce batch/accumulation")
    total_steps = int(total_steps)
    model = TransformerLM(config).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
                                  fused=device.type == "cuda")
    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters = sum(p.numel() for p in model.parameters())
    effective_batch = args.batch_size * world_size * args.gradient_accumulation * config.context_length
    state = {"model_config": asdict(config), "train_config": train_config, "data_metadata": metadata,
             "tokenizer_json": (args.data_dir / "tokenizer.json").read_text(),
             "step": 0, "tokens_seen": 0, "best_val_loss": math.inf, "elapsed_seconds": 0.0,
             "world_size": world_size, "total_steps": total_steps, "parameters": parameters,
             "effective_batch_tokens": effective_batch, "wandb_run_id": None}
    if args.resume:
        if args.resume.resolve().parent != args.output_dir:
            raise ValueError("Resume from last.pt or best.pt in the same output directory")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for name in ("model_config", "data_metadata", "world_size", "total_steps"):
            if saved[name] != state[name]:
                raise ValueError(f"Resume mismatch: {name}")
        for name in ("batch_size", "gradient_accumulation", "seed", "epochs", "learning_rate", "warmup_ratio",
                     "min_lr_ratio", "weight_decay", "grad_clip", "precision"):
            if saved["train_config"][name] != train_config[name]:
                raise ValueError(f"Resume mismatch: {name}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state.update({key: saved[key] for key in ("step", "tokens_seen", "best_val_loss", "elapsed_seconds", "wandb_run_id")})
        rng = saved["rng_states"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
        del saved
    if world_size > 1:
        wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      broadcast_buffers=False, gradient_as_bucket_view=True)
    else:
        wrapped = model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    metrics_path = args.output_dir / "metrics.jsonl"
    if rank == 0:
        if (args.output_dir / "config.json").exists() and not args.resume:
            raise FileExistsError("Output directory already contains a run; use --resume or a new output directory")
        if args.resume and metrics_path.exists():
            rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= state["step"]))
        if args.wandb_mode != "disabled":
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                             mode=args.wandb_mode, dir=str(args.output_dir), id=state["wandb_run_id"],
                             resume="allow" if state["wandb_run_id"] else None,
                             config={**train_config, "model": asdict(config), "parameters": parameters,
                                     "effective_batch_tokens": effective_batch})
            state["wandb_run_id"] = run.id
            run.define_metric("tokens_seen")
            run.define_metric("train/*", step_metric="tokens_seen")
            run.define_metric("val/*", step_metric="tokens_seen")
            run.define_metric("val_full/*", step_metric="tokens_seen")
        run_config = {key: state[key] for key in ("model_config", "train_config", "data_metadata", "world_size",
                                                 "total_steps", "parameters", "effective_batch_tokens")}
        (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        tqdm.write(f"Parameters: {parameters:,}; effective batch: {effective_batch:,} tokens; total steps: {total_steps:,}")
    if dist.is_initialized():
        dist.barrier()
    tokenizer = Tokenizer.from_str(state["tokenizer_json"])
    started = time.perf_counter()
    previous_elapsed = state["elapsed_seconds"]

    def record(split, result, **extra):
        if rank != 0:
            return
        row = {"split": split, "step": state["step"], "tokens_seen": state["tokens_seen"],
               "estimated_flops": 6 * parameters * state["tokens_seen"],
               "elapsed_seconds": previous_elapsed + time.perf_counter() - started, **result, **extra}
        with metrics_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        if run:
            prefix = "train" if split == "train" else "val_full" if extra.get("final") else "val"
            run.log({"optimizer_step": state["step"], "tokens_seen": state["tokens_seen"],
                     "estimated_flops": row["estimated_flops"],
                     **{f"{prefix}/{key}": value for key, value in result.items() if isinstance(value, (float, int))}})

    def validation():
        result = evaluate(model, val_data, device, args.batch_size, args.precision)
        record("validation", result)
        if rank == 0:
            tqdm.write(f"step {state['step']}: val loss={result['loss']:.4f}, PPL={result['ppl']:.2f}, BPB={result['bpb']:.4f}")
            if args.sample_tokens:
                samples = generate_samples(model, tokenizer, device, args.sample_tokens, args.precision)
                with (args.output_dir / "samples.jsonl").open("a") as file:
                    file.write(json.dumps({"step": state["step"], "samples": samples}, ensure_ascii=False) + "\n")
                if run:
                    import wandb
                    run.log({"samples": wandb.Table(columns=["step", "prompt", "text"],
                                                    data=[[state["step"], s["prompt"], s["text"]] for s in samples]),
                             "tokens_seen": state["tokens_seen"]})
        improved = result["loss"] < state["best_val_loss"]
        if improved:
            state["best_val_loss"] = result["loss"]
        return improved

    if state["step"] == 0:
        improved = validation()
        state["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, state, device, improved)
    progress = tqdm(total=total_steps, initial=state["step"], desc="Train", unit="step", disable=rank != 0)
    window = torch.zeros(3, device=device, dtype=torch.float64)
    window_started = time.perf_counter()
    last_lr = grad_norm = 0.0
    while state["step"] < total_steps:
        epoch, completed = divmod(state["step"], steps_per_epoch)
        indices = epoch_indices(len(train_data), args.seed, epoch, rank, world_size,
                                args.batch_size, args.gradient_accumulation, completed)
        loader = iter(DataLoader(Subset(train_data, indices.tolist()), batch_size=args.batch_size,
                                pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed + epoch)))
        for _ in range(min(steps_per_epoch - completed, total_steps - state["step"])):
            batches = [next(loader) for _ in range(args.gradient_accumulation)]
            global_tokens = torch.tensor(sum(int((y != metadata["pad_token_id"]).sum()) for _, y, _, _ in batches),
                                         dtype=torch.float64, device=device)
            if dist.is_initialized():
                dist.all_reduce(global_tokens)
            last_lr = learning_rate(state["step"], total_steps, args.learning_rate, args.warmup_ratio, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.zero_grad(set_to_none=True)
            step_totals = torch.zeros(3, device=device, dtype=torch.float64)
            for micro, (inputs, targets, document_ids, raw_bytes) in enumerate(batches):
                sync = wrapped.no_sync() if world_size > 1 and micro < args.gradient_accumulation - 1 else nullcontext()
                with sync:
                    inputs, targets, document_ids = inputs.to(device), targets.to(device), document_ids.to(device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                        logits = wrapped(inputs, document_ids=document_ids)
                        loss_sum = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1),
                                                   ignore_index=metadata["pad_token_id"], reduction="sum")
                        # DDP averages gradients across ranks; normalize by the exact global valid-token count.
                        loss = loss_sum * (world_size / global_tokens.item())
                    loss.backward()
                step_totals[0] += loss_sum.detach().double()
                step_totals[1] += (targets != metadata["pad_token_id"]).sum()
                step_totals[2] += raw_bytes.sum().to(device)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True))
            optimizer.step()
            state["step"] += 1
            state["tokens_seen"] += int(global_tokens.item())
            window += step_totals
            progress.update(1)
            final_step = state["step"] == total_steps
            should_eval = state["step"] % args.eval_every == 0 or final_step
            should_save = should_eval or state["step"] % args.save_every == 0
            if state["step"] % args.log_every == 0 or should_save:
                if dist.is_initialized():
                    dist.all_reduce(window)
                result = metrics_from_totals(*window.tolist())
                result.update({"learning_rate": last_lr, "grad_norm": grad_norm,
                               "tokens_per_second": result["tokens"] / (time.perf_counter() - window_started)})
                record("train", result)
                progress.set_postfix(loss=f"{result['loss']:.4f}", lr=f"{last_lr:.2e}", tokens=state["tokens_seen"])
                window.zero_()
                window_started = time.perf_counter()
            improved = validation() if should_eval else False
            if should_save:
                state["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                save_checkpoint(args.output_dir / "last.pt", model, optimizer, state, device, improved)
    progress.close()

    # Select and report the best checkpoint using the complete validation split.
    best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    best_step = best["step"]
    del best
    final = evaluate(model, val_data, device, args.batch_size, args.precision)
    record("validation", final, final=True, checkpoint_step=best_step)
    if rank == 0:
        final.update({"checkpoint": "best.pt", "checkpoint_step": best_step, "training_step": state["step"],
                      "tokens_seen": state["tokens_seen"], "tokenizer": metadata["tokenizer"]})
        (args.output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
        if args.sample_tokens:
            samples = generate_samples(model, tokenizer, device, args.sample_tokens, args.precision)
            (args.output_dir / "final_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
        from .plot import plot_metrics
        plot_metrics(metrics_path, args.output_dir)
        if run:
            run.summary.update({f"final/{key}": value for key, value in final.items()})
            run.finish()
        print(json.dumps(final, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
