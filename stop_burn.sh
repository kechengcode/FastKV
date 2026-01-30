#!/bin/bash

# 定义目标脚本名称 (需要与启动时一致)
SCRIPT_NAME="run_burn.py"

echo "🛑 正在停止 $SCRIPT_NAME 相关进程..."

# 使用 pkill 查找并杀死包含该脚本名的 python 进程
# -f : 匹配完整的命令行
# -e : 显示被杀死的进程
pkill -f -e "$SCRIPT_NAME"

if [ $? -eq 0 ]; then
    echo "✅ 已成功发送终止信号。"
    echo "提示: 可以运行 'nvidia-smi' 确认显卡负载是否已归零。"
else
    echo "⚠️ 未找到正在运行的 $SCRIPT_NAME 进程，或者进程已停止。"
fi
