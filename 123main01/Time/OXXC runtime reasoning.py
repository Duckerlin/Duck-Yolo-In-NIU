import time
import os
import onnxruntime
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO

# 设置图像转换，修改为640x640，以匹配模型的输入要求
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 修改为 640x640 尺寸
    transforms.ToTensor(),
])

# 加载 .pt 文件并导出为 ONNX 格式
pt_model_path = r"D:/123/123main/pt/best.pt"  # 修改为您自己的 .pt 文件路径
onnx_model_path = r"D:/123/123main/pt/best.onnx"  # 修改为您想要保存的 ONNX 模型路径

# 如果 ONNX 文件不存在，则导出
if not os.path.exists(onnx_model_path):
    model = YOLO(pt_model_path)  # 加载 .pt 模型
    model.export(format="onnx", save_dir=os.path.dirname(onnx_model_path))  # 导出为 ONNX
    print(f"ONNX model exported to: {onnx_model_path}")

# 使用 ONNX Runtime 进行推理
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

# 获取输入和输出名称
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

inference_time = []

# 设置图片目录路径
source_dir = r'D:/test2'  # 修改为您的图像文件夹路径
sources = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  # 只处理图片文件

# 遍历所有图片进行推理
for source in sources:
    image_path = os.path.join(source_dir, source)  # 获取图片的完整路径

    # 加载图片并预处理
    img = Image.open(image_path).convert('RGB')
    image_tensor = transform(img).unsqueeze(0)

    # 转换为 NumPy 数组
    image_np = image_tensor.numpy()

    # 开始计时
    start_time = time.time()

    # 进行 ONNX 推理
    result = ort_session.run([output_name], {input_name: image_np})

    # 结束计时
    end_time = time.time()

    print(f"Result for {source}: {result}")
    inference_time.append(end_time - start_time)

# 计算平均推理时间和 FPS
average_inference_time = sum(inference_time) / len(inference_time)
print(f"Inference time is {average_inference_time} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps}")
