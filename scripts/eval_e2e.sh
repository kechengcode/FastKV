model_path="/root/autodl-tmp/models/Qwen3-8B"
method="h2o"
#method="snapkv"
# Use eager attention implementation for compatibility
attn_implementation="eager"

CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method $method \
    --model_path $model_path \
    --attn_implementation $attn_implementation \
    --genlen 256 \
    --tsp_idx 15 \
    --tsp_rate 0.2 \
    --retain_rate 0.1 \
    --eviction_mode proportional \
    --num_warmups 1 \
    --num_runs 1