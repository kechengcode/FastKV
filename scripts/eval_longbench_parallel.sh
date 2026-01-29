#!/bin/bash

# 设置通用参数
MODEL_PATH="/root/autodl-tmp/models/qwen3-8b"
SAVE_DIR="outputs/results_longbench_qwen3_all_parallel"

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 自动检测可用 GPU 列表
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPUS=($(seq 0 $(($GPU_COUNT - 1))))
else
    # 如果没有 nvidia-smi，默认使用 GPU 0
    GPUS=(0)
fi

echo "========================================================"
echo "Detected GPUs: ${GPUS[*]}"
echo "Tasks will be distributed across these GPUs."
echo "========================================================"

# 定义任务列表 (格式: "Method_Name Params...")
# 注意：这里把方法名和参数放在一起字符串中
tasks=(
    "fullkv --attn_implementation flash_attention_2"
    "fastkv --attn_implementation eager --tsp_rate 0.2 --tsp_idx 17"
    "snapkv --attn_implementation eager --window_size 8 --kernel_size 7 --pooling maxpool"
    "h2o --attn_implementation eager --window_size 8"
)

pids=()
gpu_idx=0

# 循环启动任务
for task_str in "${tasks[@]}"; do
    # 解析方法名（第一个单词）作为日志文件名前缀
    method=$(echo $task_str | awk '{print $1}')
    
    # 轮询分配 GPU
    current_gpu=${GPUS[$gpu_idx]}
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    
    log_file="$SAVE_DIR/run_${method}.log"
    
    echo "[Launched] $method on GPU $current_gpu. Log: $log_file"
    
    # 放在后台运行 (&)，并将输出重定向到各自的日志文件
    (
        CUDA_VISIBLE_DEVICES=$current_gpu python -m eval.run_longbench \
            --method $method \
            --model_path "$MODEL_PATH" \
            --save_dir "$SAVE_DIR" \
            --eviction_mode proportional \
            --retain_rate 0.1 \
            --dataset all \
            $(echo $task_str | cut -d' ' -f2-) \
            > "$log_file" 2>&1
    ) &
    
    # 记录后台进程 ID
    pids+=($!)
done

# 等待所有任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo "========================================================"
echo "All parallel evaluations finished."
echo "Calculating scores..."
echo "========================================================"

# 汇总分数 (使用第一张卡即可)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python -m eval.eval_longbench \
    --results_dir "$SAVE_DIR"
