#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python train.py \
    --data-dir /user-data/chaojh26/TinyStories \
    --output-dir tinystories-bpe-16000 \
    --vocab-size 16000 \
    --min-frequency 2

python infer.py \
    --model-dir tinystories-bpe-16000 \
    --text "Once upon a time, a little cat found a red ball."
