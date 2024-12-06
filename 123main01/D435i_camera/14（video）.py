import time
import os
import subprocess
import torch
import psutil
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort
import cv2

# 获取CPU和GPU占用情况
def get_system_usage():
    # CPU 占用
    cpu_percent = psutil.cpu_percent(interval=0.1)  # 更小的时间间隔，实时性更高
    cpu_core_usage = psutil.cpu_percent(percpu=True, interval=0.1)
    
    # GPU 显存占用
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
        gpu_memory = result.stdout.strip().split('\n')
        gpu_usage = []
        for mem in gpu_memory:
            used, total = map(int, mem.split(','))
            gpu_usage.append({'used': used, 'total': total, 'percent': round(used / total * 100, 2)})
    except Exception as e:
        gpu_usage = [{'error': str(e)}]  # 如果 `nvidia-smi` 不可用，返回错误
    return cpu_percent, cpu_core_usage, gpu_usage


# 显示实时推理时间和系统资源
def display_usage(cpu_percent, cpu_core_usage, gpu_usage):
    print(f"CPU Usage: {cpu_percent:.2f}%")
    print(f"Per-Core Usage: {', '.join([f'{core:.2f}%' for core in cpu_core_usage])}")
    if gpu_usage and 'error' not in gpu_usage[0]:
        for i, gpu in enumerate(gpu_usage):
            print(f"GPU {i} - Used: {gpu['used']} MB / {gpu['total']} MB ({gpu['percent']}%)")
    else:
        print("GPU usage information not available.")

# 加载ONNX模型
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)  # 使用onnxruntime加载模型
    return session

# 加载PyTorch模型
def load_pytorch_model(model_path):
    model = torch.load(model_path)  # 加载PyTorch模型
    model.eval()  # 设置为评估模式
    return model

# 视频帧预处理
def preprocess_frame(frame, model):
    # 获取模型输入尺寸
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        n, c, h, w = model.get_inputs()[0].shape
    else:  # PyTorch模型
        n, c, h, w = model.input_size  # 假设PyTorch模型有input_size属性（需要根据具体模型调整）

    # 调整帧的大小
    frame_resized = cv2.resize(frame, (w, h))
    frame_resized = frame_resized.transpose((2, 0, 1))  # 将通道转为第一个维度
    frame_resized = frame_resized.reshape((n, c, h, w))
    frame_resized = np.float32(frame_resized)

    # 如果是PyTorch模型，需要转换为Tensor并进行标准化
    if isinstance(model, torch.nn.Module):
        frame_resized = torch.tensor(frame_resized)
        frame_resized = frame_resized / 255.0  # 归一化至[0, 1]
        frame_resized = frame_resized.permute(0, 3, 1, 2)  # 交换维度为 (batch_size, channels, height, width)
    return frame_resized

# 设置模型路径
onnx_model_path = r"D:/123/123main/pt/best.onnx"  # 修改为你的ONNX模型路径
pytorch_model_path = r"D:/123/123main/pt/best.pt"  # 修改为你的PyTorch模型路径

# 选择模型类型：ONNX 或 PyTorch
use_onnx = True  # 设置为True时使用ONNX模型，否则使用PyTorch模型

# 加载模型
if use_onnx:
    model = load_onnx_model(onnx_model_path)
else:
    model = load_pytorch_model(pytorch_model_path)

# 获取推理时间和系统占用情况
inference_time = []
cpu_percentages = []
gpu_usage = []
cpu_core_usages = []

# 视频路径
video_path = r'C:\Users\niu\duckee.mkv'  # 修改为你的输入视频路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果视频结束则跳出循环

    # 获取系统占用情况
    cpu_percent, cpu_core_usage, gpu_usage_info = get_system_usage()

    # 实时显示系统占用
    display_usage(cpu_percent, cpu_core_usage, gpu_usage_info)

    # 将占用情况记录下来
    cpu_percentages.append(cpu_percent)
    cpu_core_usages.append(cpu_core_usage)
    gpu_usage.append(gpu_usage_info)

    # 图像预处理
    frame_resized = preprocess_frame(frame, model)

    start_time = time.time()

    # 对于ONNX模型
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        result = model.run(None, {input_name: frame_resized})  # 推理

    # 对于PyTorch模型
    elif isinstance(model, torch.nn.Module):
        with torch.no_grad():
            result = model(frame_resized)  # 推理

    end_time = time.time()
    inference_time.append(end_time - start_time)

# 释放视频捕捉对象
cap.release()

# 计算平均推理时间
average_inference_time = sum(inference_time) / len(inference_time)
print(f"\nInference time is {average_inference_time:.4f} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps:.2f}")

# 计算系统资源的平均值
average_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
average_cpu_core_usage = [sum(core) / len(core) for core in zip(*cpu_core_usages)]
average_gpu_usage = []

for gpu_index in range(len(gpu_usage[0])):  # 针对多 GPU
    avg_used = sum(frame_usage[gpu_index]['used'] for frame_usage in gpu_usage) / len(gpu_usage)
    avg_total = gpu_usage[0][gpu_index]['total']  # 假设每帧的总显存相同
    avg_percent = round(avg_used / avg_total * 100, 2)
    average_gpu_usage.append({'used': avg_used, 'total': avg_total, 'percent': avg_percent})

# 打印统计结果
print(f"\nAverage CPU Usage: {average_cpu_percent:.2f}%")
print(f"Average CPU Core Usage: {', '.join([f'{core:.2f}%' for core in average_cpu_core_usage])}")
for i, gpu in enumerate(average_gpu_usage):
    print(f"GPU {i} - Avg Used: {gpu['used']:.2f} MB / {gpu['total']} MB ({gpu['percent']}%)")