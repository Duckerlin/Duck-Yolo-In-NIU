import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
import numpy as np

# 初始化 YOLOv8 模型
model = YOLO('D:/123/123main/pt/best.pt')  # 替換為你的 YOLOv8 模型檔案

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 視訊流

# 開始 RealSense 管道
pipeline.start(config)

try:
    while True:
        # 獲取 RGB 影像幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 將影像轉換為 numpy 陣列格式
        color_image = np.asanyarray(color_frame.get_data())

        # 使用 YOLOv8 進行推理
        results = model(color_image)
        annotated_image = results[0].plot()  # 在影像上繪製檢測結果

        # 顯示檢測結果
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_image)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止 RealSense 管道
    pipeline.stop()
    cv2.destroyAllWindows()
