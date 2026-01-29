#!/bin/bash

# 设置通用参数
MODEL_PATH="/root/autodl-tmp/models/qwen3-8b"
SAVE_DIR="outputs/results_longbench_qwen3_all"
GPU_ID=0

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 将所有输出重定向到日志文件，同时在终端显示
LOG_FILE="$SAVE_DIR/run.log"
exec > >(tee -i "$LOG_FILE") 2>&1
echo "Log file saved to: $LOG_FILE"

# 定义执行函数
run_eval() {
    local method=$1
    shift # 移除第一个参数(method)，剩下的都是该方法的特有参数
    
    echo "========================================================"
    echo "Starting evaluation for method: $method"
    echo "========================================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval.run_longbench \
        --method "$method" \
        --model_path "$MODEL_PATH" \
        --save_dir "$SAVE_DIR" \
        --eviction_mode proportional \
        --retain_rate 0.1 \
        --dataset all \
        "$@"
}

# 1. Test fullkv (Base)
run_eval fullkv \
    --attn_implementation "flash_attention_2"

# 2. Test FastKV
run_eval fastkv \
    --attn_implementation eager \
    --tsp_rate 0.2 \
    --tsp_idx 17

# 3. Test SnapKV
run_eval snapkv \
    --attn_implementation eager \
    --window_size 8 \
    --kernel_size 7 \
    --pooling maxpool

# 4. Test H2O
run_eval h2o \
    --attn_implementation eager \
    --window_size 8

echo "========================================================"
echo "All evaluations finished. Calculating scores..."
echo "========================================================"

# Finally Eval All - 汇总分数
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval.eval_longbench \
    --results_dir "$SAVE_DIR"
