import torch
import subprocess
import os
from ultralytics import YOLO

def convert_pt_to_onnx(pt_model_path, onnx_model_path, input_size=(1, 3, 640, 640)):
    """
    将 .pt 模型转换为 .onnx 格式
    :param pt_model_path: 输入的 .pt 文件路径
    :param onnx_model_path: 输出的 .onnx 文件路径
    :param input_size: 输入大小 (batch_size, channels, height, width)
    """
    # 加载 YOLO 模型
    model = YOLO(pt_model_path)  # 使用 ultralytics YOLO 加载模型

    # 设置为评估模式
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(*input_size)  # 根据模型输入大小调整

    # 导出为 ONNX 格式
    torch.onnx.export(model.model, dummy_input, onnx_model_path, export_params=True,
                      opset_version=12, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Successfully converted {pt_model_path} to {onnx_model_path}")

def convert_onnx_to_openvino(onnx_model_path, openvino_model_dir):
    """
    使用 OpenVINO Model Optimizer 将 ONNX 模型转换为 OpenVINO 格式（.xml 和 .bin 文件）
    :param onnx_model_path: 输入的 ONNX 文件路径
    :param openvino_model_dir: 输出的 OpenVINO 模型文件目录
    """
    # 确保目录存在
    os.makedirs(openvino_model_dir, exist_ok=True)

    # OpenVINO Model Optimizer 转换命令
    mo_command = [
        "mo", "--input_model", onnx_model_path,
        "--output_dir", openvino_model_dir,
        "--disable_nhwc_to_nchw",  # 禁用 NHWC 到 NCHW 的转换，如果有需要可以注释掉
    ]

    # 运行 Model Optimizer
    try:
        subprocess.run(mo_command, check=True)
        print(f"Successfully converted {onnx_model_path} to OpenVINO format in {openvino_model_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during OpenVINO conversion: {e}")
        
def main():
    # 设置文件路径
    pt_model_path = r"D:/123/123main/pt/best.pt"  # 需要转换的 .pt 模型路径
    onnx_model_path = r"D:/123/123main/pt/best.onnx"  # 转换后的 .onnx 路径
    openvino_model_dir = r"D:/123/123main/pt/openvino_model"  # OpenVINO 格式的输出目录
    # 转换 PyTorch 模型为 ONNX 格式
    convert_pt_to_onnx(pt_model_path, onnx_model_path)

    # 转换 ONNX 模型为 OpenVINO 格式
    convert_onnx_to_openvino(onnx_model_path, openvino_model_dir)

if __name__ == "__main__":
    main()
