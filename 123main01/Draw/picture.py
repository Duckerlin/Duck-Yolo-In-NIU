import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径列表
file_paths = [
    r"D:\123\123main\fps\fps_data(yolov8s).csv",
    r"D:\123\123main\fps\fps_data(dwc).csv",
    r"D:\123\123main\fps\fps_data(ECA).csv",
    r"D:\123\123main\fps\fps_data(swin).csv",
    r"D:\123\123main\fps\fps_data(biformer).csv",
    r"D:\123\123main\fps\fps_data(swin + dwc).csv",
    r"D:\123\123main\fps\fps_data(swin + ghost).csv"
]

# 自定义命名列表，按顺序与文件路径一一对应
custom_names = [
    "YOLOv8s",   
    "YOLOv8 + DWC",
    "YOLOv8 + ECA",
    "YOLOv8 + Swin",
    "YOLOv8 + Biformer",
    "YOLOv8 + Swin & DWC",
    "YOLOv8 + Swin & Ghost"
]

# 初始化结果表格
result_data = {
    "name": [],  # 文件名称
    "fps": [],  # FPS 数值
    "frame time": []  # Frame Time 数值
}

# 遍历处理每个 CSV 文件
for idx, file_path in enumerate(file_paths):
    try:
        # 读取 CSV 文件并清理列名
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        
        # 检查是否包含足够的列数和行数
        if df.shape[1] >= 3 and len(df) > 0:
            # 取第二列的最后一个值作为 FPS
            fps = df.iloc[-1, 1]
            # 取第三列的最后一个值作为 Frame Time
            frame_time = df.iloc[-1, 2]
            
            # 使用自定义命名
            result_data["name"].append(custom_names[idx])  # 使用自定义命名
            result_data["fps"].append(fps)
            result_data["frame time"].append(frame_time)
        else:
            raise ValueError(f"File {file_path} has insufficient columns or rows.")
        
    except Exception as e:
        # 捕获错误，输出错误信息
        print(f"Error processing {file_path}: {e}")
        result_data["name"].append(custom_names[idx])  # 使用自定义命名
        result_data["fps"].append("Error")
        result_data["frame time"].append("Error")

# 将结果转换为 DataFrame
result_df = pd.DataFrame(result_data)

# 保存结果为新的 CSV 文件
result_df.to_csv(r"D:\123\123main\fps\summary_table.csv", index=False)
print(result_df)

# -------- Matplotlib 表格可视化 --------

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 4))  # 调整图形大小
ax.axis('off')  # 隐藏坐标轴

# 创建表格
table = ax.table(
    cellText=result_df.values,
    colLabels=result_df.columns,
    cellLoc='center',
    loc='center'
)

# 调整表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)  # 调整字体大小
table.scale(1.0, 1.5)  # 调整表格比例

# 使用 tight_layout 来避免显示不全
plt.tight_layout()

# 显示图表
plt.show()
