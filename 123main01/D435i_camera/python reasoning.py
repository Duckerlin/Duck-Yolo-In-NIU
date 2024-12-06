import time
import os
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('D:/123/123main/pt/best.pt')

inference_time = []

# Define path to the image file
source_dir = r'D:\test2'
sources = [f for f in os.listdir(source_dir)]

for source in sources:
    image = os.path.join(source_dir, source)
    start_time = time.time()
    results = model(image)  # list of Results objects     , imgsz=640
    end_time = time.time()
    inference_time.append(end_time-start_time)
average_inference_time = sum(inference_time) / len(inference_time)
print(f"Inference time is {average_inference_time} seconds")
fps = 1.0 / average_inference_time
print(f"FPS is {fps}")
