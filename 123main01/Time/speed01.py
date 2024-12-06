import time
import os
import csv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
import onnxruntime as ort
from openvino.runtime import Core
import matplotlib.pyplot as plt  # 添加此行


# 导出模型为 ONNX 和 OpenVINO 格式
def export_models(model_path):
    model = YOLO(model_path)
    export_dir = os.path.dirname(model_path)

    exported_paths = {}
    # 导出 ONNX
    onnx_path = os.path.join(export_dir, "model.onnx")
    model.export(format="onnx", imgsz=640, device="cpu")
    exported_paths['onnx'] = onnx_path

    # 导出 OpenVINO FP32
    openvino_fp32_path = os.path.join(export_dir, "openvino_fp32")
    model.export(format="openvino", imgsz=640, device="cpu")
    exported_paths['openvino_fp32'] = openvino_fp32_path

    # 导出 OpenVINO INT8
    openvino_int8_path = os.path.join(export_dir, "openvino_int8")
    model.export(format="openvino", imgsz=640, int8=True, device="cpu")
    exported_paths['openvino_int8'] = openvino_int8_path

    return exported_paths


# 测试推理时间
def test_inference_time(model_format, model_path, input_data, device="CPU"):
    if model_format == 'pytorch':
        model = YOLO(model_path)
        model.fuse()  # 加速推理
        start = time.time()
        model(input_data)
        end = time.time()

    elif model_format == 'onnx':
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        start = time.time()
        session.run(None, {input_name: input_data})
        end = time.time()

    elif model_format.startswith("openvino"):
        ie = Core()
        compiled_model = ie.compile_model(model_path, device_name=device)
        infer_request = compiled_model.create_infer_request()
        input_blob = next(iter(compiled_model.inputs))
        start = time.time()
        infer_request.infer({input_blob: input_data})
        end = time.time()

    else:
        raise ValueError(f"不支持的模型格式: {model_format}")

    return end - start


# 保存推理时间到 CSV
def save_inference_times(inference_times, filename):
    rows = [{"format": fmt, "time (s)": t} for fmt, t in inference_times.items()]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["format", "time (s)"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"推理时间已保存到 {filename}")


# 主程序入口
def main():
    # 替换为你的自定义模型路径
    model_path = 'D:/123main/pt/best.pt'  # 替换为你的 .pt 文件路径
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)  # 模拟输入数据

    # 导出模型到其他格式
    exported_paths = export_models(model_path)

    # 添加 PyTorch 原生模型路径到结果中
    exported_paths['pytorch'] = model_path

    # 测试推理时间
    inference_times = {}
    for model_format, path in exported_paths.items():
        device = 'cuda' if model_format == 'tensorrt' else 'cpu'
        inference_times[model_format] = test_inference_time(model_format, path, input_data, device=device)

    # 保存推理时间
    save_inference_times(inference_times, "inference_times.csv")

    # 使用 pandas 加载并展示结果
    df = pd.read_csv("inference_times.csv")
    print(df)

    # 绘制表格
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.show()


# 运行主程序
if __name__ == "__main__":
    main()
