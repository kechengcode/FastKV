import torch
import torch.multiprocessing as mp
import os
import time

# 配置参数
MATRIX_SIZE = 20000  # 矩阵大小，根据显存大小调整 (比如 A100/H100 可以设大点，4090/3090 设小点)
GPU_COUNT = 8        # 想要跑满的显卡数量

def stress_task(gpu_id):
    """
    单个 GPU 的压力测试函数
    """
    try:
        # 设置当前进程使用的 GPU
        device = torch.device(f"cuda:{gpu_id}")
        
        print(f"🚀 [GPU {gpu_id}] 开始向显存加载数据...")
        
        # 创建两个巨大的随机矩阵 (Float32)
        # 如果你想压榨 Tensor Core 并不太在意显存大小，可以改用 half()
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        
        print(f"🔥 [GPU {gpu_id}] 负载已拉满，开始死循环计算...")
        
        # 死循环计算矩阵乘法
        while True:
            # 矩阵乘法是计算密集型任务
            c = torch.mm(a, b)
            
            # 为了防止被编译器优化掉（虽然 PyTorch eager mode 不会），偶尔做个同步或简单操作
            # 但为了保持 100% 利用率，通常不需要显式同步，让 CUDA 队列塞满即可
            
    except RuntimeError as e:
        print(f"❌ [GPU {gpu_id}] 发生错误 (可能是显存不足): {e}")
    except KeyboardInterrupt:
        print(f"🛑 [GPU {gpu_id}] 停止。")

def main():
    # 检查系统是否有足够的 GPU
    available_gpus = torch.cuda.device_count()
    if available_gpus < GPU_COUNT:
        print(f"⚠️ 警告: 系统只有 {available_gpus} 张卡，但你要求跑 {GPU_COUNT} 张。将只运行 {available_gpus} 张。")
        target_gpus = available_gpus
    else:
        target_gpus = GPU_COUNT

    print(f"正在启动 {target_gpus} 个进程进行压力测试...")
    
    # 启动多进程
    processes = []
    mp.set_start_method('spawn', force=True) # CUDA 必须使用 spawn 方法
    
    for i in range(target_gpus):
        p = mp.Process(target=stress_task, args=(i,))
        p.start()
        processes.append(p)
        
    print("✅ 所有进程已启动。请使用 nvidia-smi 查看状态。")
    print("运行 'bash stop_burn.sh' 来停止脚本。")
    
    # 主进程保持存活
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("主进程收到终止信号...")

if __name__ == '__main__':
    main()
