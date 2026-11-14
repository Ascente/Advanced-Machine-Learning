#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=2,3
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_APP_URL=https://forge.coreweave.com/wandb

runs_dir=/user-data/chaojh26/pretrain/scaling/tinystories-1

tail -n +2 pretrain/scaling_models.csv | while IFS=, read -r name n_layers d_model n_heads d_ff; do
    output_dir="$runs_dir/$name"
    if [ -f "$output_dir/final_metrics.json" ]; then
        echo "Skipping completed model: $name"
        continue
    fi
    mkdir -p "$output_dir"
    echo "Training $name: layers=$n_layers, width=$d_model, heads=$n_heads, FFN=$d_ff"
    echo "Log: $output_dir/train.log"

    torchrun --standalone --nproc_per_node=2 -m pretrain.train \
        --data-dir /user-data/chaojh26/TinyStories/packed-1024 \
        --output-dir "$output_dir" \
        --d-model "$d_model" \
        --n-layers "$n_layers" \
        --n-heads "$n_heads" \
        --d-ff "$d_ff" \
        --batch-size 16 \
        --gradient-accumulation 4 \
        --epochs 1 \
        --learning-rate 4e-4 \
        --min-lr-ratio 0.1 \
        --warmup-ratio 0.02 \
        --weight-decay 0.1 \
        --grad-clip 1.0 \
        --eval-every 100 \
        --save-every 1000 \
        --log-every 10 \
        --sample-tokens 0 \
        --precision bf16 \
        --seed 42 \
        --wandb-mode online \
        --wandb-project aml-hw1 \
        --run-name "scaling-$name" \
        >> "$output_dir/train.log" 2>&1
done
