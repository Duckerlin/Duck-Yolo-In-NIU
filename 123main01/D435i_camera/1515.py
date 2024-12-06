import psutil
import subprocess
import time

def get_gpu_usage():
    """ 获取 GPU 使用情况（通过 nvidia-smi）"""
    try:
        # 运行 nvidia-smi 命令并获取 GPU 信息
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        result = result.decode('utf-8').strip().split(',')
        gpu_usage = result[0].strip()  # GPU 使用率
        memory_used = result[1].strip()  # 已使用 GPU 内存
        memory_total = result[2].strip()  # 总 GPU 内存
        return gpu_usage, memory_used, memory_total
    except subprocess.CalledProcessError as e:
        return None, None, None

def get_cpu_usage():
    """ 获取 CPU 使用情况 """
    return psutil.cpu_percent(interval=1)  # 获取 CPU 使用百分比，设置 interval=1 秒

def monitor_resources():
    """ 监控 CPU 和 GPU 资源的使用情况并输出 """
    while True:
        # 获取 GPU 使用情况
        gpu_usage, memory_used, memory_total = get_gpu_usage()
        if gpu_usage is not None:
            print(f"GPU Usage: {gpu_usage}% | Memory Used: {memory_used}MB | Memory Total: {memory_total}MB")
        else:
            print("GPU is not available or nvidia-smi failed.")
        
        # 获取 CPU 使用情况
        cpu_usage = get_cpu_usage()
        print(f"CPU Usage: {cpu_usage}%")
        
        # 每隔 1 秒更新一次
        time.sleep(1)

if __name__ == "__main__":
    monitor_resources()
