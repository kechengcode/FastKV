#!/bin/bash

# ========================================================
# 8-GPU Parallel Evaluation Script for Qwen3-8B on LongBench
# Allocation: 2 GPUs per method
# Output: Automatically starts GPU burn script after completion
# ========================================================

# 设置模型路径 (请确保大小写正确)
MODEL_PATH="/root/autodl-tmp/models/qwen3-8b"
SAVE_DIR="outputs/results_longbench_qwen3_parallel_8gpu"
SCRIPT_BURN="run_burn.py"  # 烧机 Python 脚本
LOG_BURN="burn_output.log" # 烧机日志

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 定义一个辅助函数用于后台启动任务
# 参数: 1.方法名 2.GPU列表(逗号分隔) 3.后续参数
launch_task() {
    local method=$1
    local gpus=$2
    local log_file="$SAVE_DIR/${method}.log"
    shift 2 # 移除前两个参数，保留剩下的作为命令参数

    echo "--------------------------------------------------------"
    echo "Lauching [${method}] on GPUs [${gpus}]"
    echo "Logs > ${log_file}"
    
    # 核心命令: 指定 CUDA_VISIBLE_DEVICES 并后台运行 (&)
    (
        CUDA_VISIBLE_DEVICES=$gpus python -m eval.run_longbench \
            --method "$method" \
            --model_path "$MODEL_PATH" \
            --save_dir "$SAVE_DIR" \
            --eviction_mode proportional \
            --retain_rate 0.1 \
            --dataset all \
            --disable_thinking \
            "$@" > "$log_file" 2>&1
    ) &
    
    # 记录后台进程PID
    pids+=($!)
}

# 存储所有后台进程ID
pids=()

echo "========================================================"
echo "Starting Parallel Evaluation on 8 GPUs..."
echo "========================================================"

# 1. FullKV (Base) -> GPUs 0,1
launch_task fullkv "0,1" --attn_implementation "flash_attention_2"

# 2. FastKV -> GPUs 2,3
launch_task fastkv "2,3" --attn_implementation eager --tsp_rate 0.2 --tsp_idx 17

# 3. SnapKV -> GPUs 4,5
launch_task snapkv "4,5" --attn_implementation eager --window_size 8 --kernel_size 7 --pooling maxpool

# 4. H2O -> GPUs 6,7
launch_task h2o "6,7" --attn_implementation eager --window_size 8

echo "========================================================"
echo "All 4 tasks launched separately. Waiting for completion..."
echo "You can check progress in another terminal using: tail -f $SAVE_DIR/*.log"
echo "========================================================"

# 等待所有子进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo "========================================================"
echo "All evaluations finished. Calculating scores..."
echo "========================================================"

# 最后用卡0进行分数汇总
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_longbench \
    --results_dir "$SAVE_DIR"

echo "========================================================"
echo "Evaluation sequence complete."
echo "========================================================"

# ========================================================
# 自动启动 GPU Burn 这个程序
# ========================================================

if [ ! -f "$SCRIPT_BURN" ]; then
    echo "❌ 错误: 找不到文件 $SCRIPT_BURN"
    echo "请确认 run_burn.py 在当前目录下。"
    exit 1
fi

echo "🚀 检测到评测已结束，正在从后台启动 GPU 满载程序 ($SCRIPT_BURN) ..."

# 核心命令: nohup 后台运行烧机脚本
nohup python -u "$SCRIPT_BURN" > "$LOG_BURN" 2>&1 &

# 获取刚启动的烧机进程 PID
BURN_PID=$!

echo "✅ GPU 压力测试已启动 (PID: $BURN_PID)"
echo "📄 日志文件: $LOG_BURN"
echo "👀 请使用 'tail -f $LOG_BURN' 查看状态"
echo "🛑 若要停止，请运行 'pkill -f $SCRIPT_BURN'"
