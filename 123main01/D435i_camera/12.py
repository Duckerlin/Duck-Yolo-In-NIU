import time
import os
import subprocess
import torch
import psutil
from PIL import Image
import numpy as np
from openvino.runtime import Core  # OpenVINO 2.0 API
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
    
    # 打印结果
    print(f"CPU Usage: {cpu_percent}%")
    print(f"CPU Core Usage: {cpu_core_usage}")
    for i, gpu in enumerate(gpu_usage):
        print(f"GPU {i} - {gpu}")

# 初始化 OpenVINO 2.0
core = Core()  # 使用 openvino.runtime 的 Core 类
model_path = "/path/to/model.xml"
model = core.read_model(model_path)  # 读取模型
compiled_model = core.compile_model(model, device_name="CPU")  # 编译模型到 CPU

# 获取推理时间
inference_time = []
source_dir = r'D:/test2'
sources = [f for f in os.listdir(source_dir)]

for source in sources:
    image = os.path.join(source_dir, source)
    img = Image.open(image).convert('RGB')
    image_array = np.array(img)

    input_blob = next(iter(model.inputs))  # 获取输入blob
    n, c, h, w = model.input(input_blob).shape  # 获取输入维度
    image_resized = cv2.resize(image_array, (w, h))
    image_resized = image_resized.transpose((2, 0, 1))
    image_resized = image_resized.reshape((n, c, h, w))

    # 获取系统占用情况
    get_system_usage()

    start_time = time.time()
    result = compiled_model([image_resized])  # 使用编译的模型进行推理
    end_time = time.time()

    print(result)
    inference_time.append(end_time - start_time)

average_inference_time = sum(inference_time) / len(inference_time)
print(f"Inference time is {average_inference_time} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps}")
