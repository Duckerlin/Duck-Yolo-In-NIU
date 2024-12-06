import torch
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("D:/123/123main/pt/best.pt")

# 打印模型结构
print(model)  # 这会打印出YOLOv8模型的详细结构
