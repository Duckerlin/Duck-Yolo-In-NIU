import csv
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import os

# 加载两个不同的模型
model_other = YOLO('D:/123/123main/pt/best.pt')
model_yolov8s = YOLO('D:/123/123main/pt/best(yolov8s).pt')

# 获取模型信息
print("Model Other Info:")
print(model_other.info(detailed=True))
print("Model YOLOv8s Info:")
print(model_yolov8s.info(detailed=True))

# 从 CSV 文件提取 epoch 行中的最后一个数据
def extract_epochs_from_csv(filename):
    if not os.path.exists(filename):
        print(f"[错误] CSV 文件未找到：{filename}")
        return "N/A"
    try:
        df = pd.read_csv(filename)
        if df.empty:
            print(f"[警告] CSV 文件为空：{filename}")
            return "N/A"
        return df.iloc[-1, 0]  # 获取最后一行第一列数据
    except pd.errors.EmptyDataError:
        print(f"[错误] CSV 文件为空：{filename}")
    except Exception as e:
        print(f"[错误] 无法加载 CSV 文件 {filename}：{e}")
    return "N/A"

# 从权重文件提取 epochs 信息
def extract_epochs_from_weights(filepath):
    if not os.path.exists(filepath):
        print(f"[错误] 权重文件未找到：{filepath}")
        return "N/A"
    try:
        weights = torch.load(filepath)
        return weights.get('epoch', "N/A")  # 提取 epoch 信息，若不存在则返回 N/A
    except Exception as e:
        print(f"[错误] 无法从权重文件加载 epochs 信息：{e}")
        return "N/A"

# 提取模型关键信息
def extract_model_data(model_info, model_name, epochs):
    if isinstance(model_info, tuple) and len(model_info) == 4:
        return {
            "name": model_name,
            "layers": model_info[0],
            "parameters": model_info[1],
            "GFLOPs": model_info[3],
            "epoch": epochs
        }
    return {}

# 分别加载两个模型的 result.csv 文件并提取 epochs 信息
epochs_other = extract_epochs_from_csv('D:/123/123main/runs/exp1 (other)/results.csv')
epochs_yolov8s = extract_epochs_from_csv('D:/123/123main/runs/exp (_init_)/results.csv')

# 如果无法从 CSV 文件中提取 epochs，则从权重文件提取
if epochs_other == "N/A":
    epochs_other = extract_epochs_from_weights('D:/123main/pt/best.pt')
if epochs_yolov8s == "N/A":
    epochs_yolov8s = extract_epochs_from_weights('D:/123main/best(yolov8s).pt')

# 打印提取的 epoch 信息
print(f"Epoch information for 'yolov8s_Swin + C3Ghost + DWC': {epochs_other}")
print(f"Epoch information for 'YOLOv8s': {epochs_yolov8s}")

# 提取数据并保存到 CSV
extracted_data = [
    extract_model_data(model_other.info(), "yolov8s_Swin + C3Ghost + DWC", epochs_other),
    extract_model_data(model_yolov8s.info(), "YOLOv8s", epochs_yolov8s)
]

def save_to_csv(extracted_data, filename):
    if extracted_data:
        keys = extracted_data[0].keys()
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(extracted_data)
        print(f"模型信息已保存到 {filename}")

# 保存模型信息到 CSV
save_to_csv(extracted_data, 'model_summary.csv')

# 使用 pandas 读取 CSV 文件并动态调整表格大小
def plot_table_from_csv(filename):
    try:
        df = pd.read_csv(filename)
        fig, ax = plt.subplots(figsize=(10, 5))  # 初始大小
        ax.axis('off')

        # 根据数据大小动态调整字体和比例
        row_count, col_count = df.shape
        font_size = max(10 - row_count // 5, 5)  # 动态字体调整，最小为 5
        scale_x = min(1.5, 10 / col_count)  # 动态表格宽度调整
        scale_y = min(1.5, 10 / row_count)  # 动态表格高度调整

        # 创建表格
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)  # 设置动态字体大小
        table.scale(scale_x, scale_y)  # 动态缩放比例

        # 使用 tight_layout() 避免显示不全
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[错误] 无法显示表格：{e}")

# 显示表格
plot_table_from_csv('model_summary.csv')
