#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES=2,3
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_APP_URL=https://forge.coreweave.com/wandb

if [ ! -f /user-data/chaojh26/TinyStories/packed-1024/metadata.json ]; then
    python -m pretrain.prepare \
        --dataset tinystories \
        --data-dir /user-data/chaojh26/TinyStories \
        --output-dir /user-data/chaojh26/TinyStories/packed-1024 \
        --tokenizer flexitok/bpe_fw_edu_16000 \
        --context-length 1024 \
        --batch-size 1024
fi

torchrun --standalone --nproc_per_node=2 -m pretrain.train \
    --data-dir /user-data/chaojh26/TinyStories/packed-1024 \
    --output-dir /user-data/chaojh26/pretrain/runs/20261113-tinystories-1 \
    --d-model 1024 \
    --n-layers 16 \
    --n-heads 16 \
    --d-ff 4096 \
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
    --sample-tokens 100 \
    --precision bf16 \
    --seed 42 \
    --wandb-mode online \
    --wandb-project aml-hw1 \
    --run-name 20261113-tinystories-1
