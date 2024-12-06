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
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_core_usage = psutil.cpu_percent(percpu=True, interval=1)
    
    # GPU 显存占用
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    gpu_usage = []
    for mem in gpu_memory:
        used, free, total = map(int, mem.split(','))
        gpu_usage.append(f"Used: {used} MB, Free: {free} MB, Total: {total} MB")
    
    # 返回占用情况
    return cpu_percent, cpu_core_usage, gpu_usage

# 加载ONNX模型
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)  # 使用onnxruntime加载模型
    return session

# 加载PyTorch模型
def load_pytorch_model(model_path):
    model = torch.load(model_path)  # 加载PyTorch模型
    model.eval()  # 设置为评估模式
    return model

# 图像预处理
def preprocess_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    image_array = np.array(img)

    # 获取模型输入尺寸
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        n, c, h, w = model.get_inputs()[0].shape
    else:  # PyTorch模型
        n, c, h, w = model.input_size  # 假设PyTorch模型有input_size属性（需要根据具体模型调整）

    # 调整图像大小
    image_resized = cv2.resize(image_array, (w, h))
    image_resized = image_resized.transpose((2, 0, 1))  # 将通道转为第一个维度
    image_resized = image_resized.reshape((n, c, h, w))
    image_resized = np.float32(image_resized)

    # 如果是PyTorch模型，需要转换为Tensor并进行标准化
    if isinstance(model, torch.nn.Module):
        image_resized = torch.tensor(image_resized)
        image_resized = image_resized / 255.0  # 归一化至[0, 1]
        image_resized = image_resized.permute(0, 3, 1, 2)  # 交换维度为 (batch_size, channels, height, width)
    return image_resized

# 设置模型路径
onnx_model_path = r"D:/123/123main/pt/best.onnx"   # 修改为你的ONNX模型路径
pytorch_model_path = r"D:/123/123main/pt/best.pt" # 修改为你的PyTorch模型路径

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

source_dir = r'D:/test2'
sources = [f for f in os.listdir(source_dir)]

for source in sources:
    image_path = os.path.join(source_dir, source)
    image_resized = preprocess_image(image_path, model)  # 预处理图像

    # 获取系统占用情况
    cpu_percent, cpu_core_usage, gpu_usage_info = get_system_usage()

    # 将占用情况记录下来
    cpu_percentages.append(cpu_percent)
    cpu_core_usages.append(cpu_core_usage)
    gpu_usage.append(gpu_usage_info)

    start_time = time.time()

    # 对于ONNX模型
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        result = model.run(None, {input_name: image_resized})  # 推理

    # 对于PyTorch模型
    elif isinstance(model, torch.nn.Module):
        with torch.no_grad():
            result = model(image_resized)  # 推理

    end_time = time.time()

    print(result)
    inference_time.append(end_time - start_time)

# 计算平均推理时间
average_inference_time = sum(inference_time) / len(inference_time)
print(f"Inference time is {average_inference_time} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps}")

# 计算系统资源的平均值
average_cpu_percent = sum(cpu_percentages) / len(cpu_percentages)
average_cpu_core_usage = [sum(core) / len(core) for core in zip(*cpu_core_usages)]  # 计算每个核心的平均值
average_gpu_usage = [gpu_info for gpu_list in gpu_usage for gpu_info in gpu_list]  # 展开并计算平均

print(f"Average CPU Usage: {average_cpu_percent}%")
print(f"Average CPU Core Usage: {average_cpu_core_usage}")
for i, gpu in enumerate(average_gpu_usage):
    print(f"GPU {i} - {gpu}")
