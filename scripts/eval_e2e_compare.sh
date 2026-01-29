#!/bin/bash

MODEL_PATH="/root/autodl-tmp/models/qwen3-8b"
COMMON_ARGS="--model_path $MODEL_PATH --genlen 256 --num_warmups 1 --num_runs 1 --eval_batch_size 1 --attn_implementation flash_attention_2 --eviction_mode proportional --retain_rate 0.1"

echo "========================================================"
echo "Running E2E Benchmark for FullKV"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method fullkv \
    $COMMON_ARGS

echo "========================================================"
echo "Running E2E Benchmark for H2O"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method h2o \
    $COMMON_ARGS \
    --window_size 8

echo "========================================================"
echo "Running E2E Benchmark for SnapKV"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method snapkv \
    $COMMON_ARGS \
    --window_size 8 \
    --kernel_size 7 \
    --pooling maxpool

echo "========================================================"
echo "Running E2E Benchmark for FastKV"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method fastkv \
    $COMMON_ARGS \
    --tsp_idx 17 \
    --tsp_rate 0.2

echo "Done."
