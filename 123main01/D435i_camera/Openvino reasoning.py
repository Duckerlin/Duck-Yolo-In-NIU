import time
import os
from PIL import Image
import numpy as np
from openvino.inference_engine import IECore
import cv2

ie = IECore()
net = ie.read_network(model="/home/lz/yolov5/runs/train/best_20230921switch_openvino_model/best_20230921switch.xml", weights="/home/lz/yolov5/runs/train/best_20230921switch_openvino_model/best_20230921switch.bin")

# 加载推理插件
exec_net = ie.load_network(network=net, device_name="CPU")  # 可根据需要更改设备名称

inference_time = []

# Define path to the image file
source_dir = r'D:\test2'
sources = [f for f in os.listdir(source_dir)]

for source in sources:
    image = os.path.join(source_dir, source)
    img = Image.open(image).convert('RGB')
    image_array = np.array(img)

    # 预处理
    input_blob = next(iter(net.input_info))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    image_resized = cv2.resize(image_array, (w, h))
    image_resized = image_resized.transpose((2, 0, 1))  # 转换通道顺序
    image_resized = image_resized.reshape((n, c, h, w))

    start_time = time.time()
    # 进行推理
    result = exec_net.infer(inputs={input_blob: image_resized})

    end_time = time.time()

    print(result)
    inference_time.append(end_time-start_time)
average_inference_time = sum(inference_time) / len(inference_time)
print(f"Inference time is {average_inference_time} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps}")
