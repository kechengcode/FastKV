# Test fullkv (Base)
# CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
#     --method fullkv \
#     --model_path "/root/autodl-tmp/models/Qwen3-8B" \
#     --attn_implementation flash_attention_2 \
#     --save_dir "outputs/results_longbench_qwen3" \
#     --eviction_mode proportional \
#     --dataset samsum \
#     --retain_rate 0.1

# # Test FastKV
# CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
#     --method fastkv \
#     --model_path "/root/autodl-tmp/models/Qwen3-8B" \
#     --attn_implementation flash_attention_2 \
#     --save_dir "outputs/results_longbench_qwen3" \
#     --eviction_mode proportional \
#     --tsp_rate 0.2 \
#     --tsp_idx 15 \
#     --dataset samsum \
#     --retain_rate 0.1

# Test SnapKV
CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
    --method snapkv \
    --model_path "/root/autodl-tmp/models/Qwen3-8B" \
    --attn_implementation flash_attention_2 \
    --save_dir "outputs/results_longbench_qwen3" \
    --eviction_mode proportional \
    --dataset samsum \
    --window_size 8 \
    --kernel_size 7 \
    --pooling maxpool \
    --retain_rate 0.1

# Test H2O
CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
    --method h2o \
    --model_path "/root/autodl-tmp/models/Qwen3-8B" \
    --attn_implementation flash_attention_2 \
    --save_dir "outputs/results_longbench_qwen3" \
    --eviction_mode proportional \
    --dataset samsum \
    --window_size 8 \
    --retain_rate 0.1

# Finally Eval All
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_longbench \
    --results_dir "outputs/results_longbench_qwen3"
