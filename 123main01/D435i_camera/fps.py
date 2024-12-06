from ultralytics import YOLO
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# 初始化 YOLO 模型
model = YOLO("D:/123/123main/pt/best.pt")  # 使用 YOLOv8 模型

# 開啟攝影機
cap = cv2.VideoCapture(0)  # 攝影機 ID
data = []  # 用於存儲每幀數據

# 設置執行幀數限制
frame_limit = 10
frame_count = 0

while frame_count < frame_limit:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # 執行推論
    results = model.predict(frame, conf=0.5)

    end_time = time.time()

    # 計算每幀執行時間與 FPS
    inference_time = end_time - start_time
    fps = 1 / inference_time

    # 保存數據到表格
    data.append({
        "Frame": frame_count + 1,
        "Inference Time (s)": round(inference_time, 4),
        "FPS": round(fps, 2)
    })

    frame_count += 1

cap.release()

# 將數據保存為表格
df = pd.DataFrame(data)

# 使用 matplotlib 繪製表格
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# 保存表格為圖片
output_buffer = BytesIO()
plt.savefig(output_buffer, format='png')
output_buffer.seek(0)
image = Image.open(output_buffer)

# 保存圖片到本地
image.save("yolo_performance_table.png")
print("表格圖片已保存為 'yolo_performance_table.png'")
