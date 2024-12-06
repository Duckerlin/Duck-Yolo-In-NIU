import torch
from ultralytics import YOLO

def train_and_export_model():
    # 使用 yolov8s.yaml 加载模型配置
    model = YOLO("D:/123(1)/123main01/ultralytics/cfg/models/v8/yolov8s.yaml", task='detect')  # 指定任务为检测

    # 加载训练好的权重文件 yolov8s.pt
    model.load("D:/123(1)/123main01/yolov8s.pt")  # 使用 load 方法加载权重
    # 允许所有层在训练中更新
    model.freeze = False
    
    # 训练模型
    model.train(data="D:/123(1)/123main01/ultralytics/cfg/datasets/mydata.yaml", epochs=700)
     # 保存修改后的模型权重为 .pt 文件
    torch.save(model.state_dict(), "modified_yolov8s.pt")  # 保存为新的模型pt 文件


    # 导出模型为 ONNX 格式
    # model.export(format="onnx")  # 选择导出格式

if __name__ == '__main__':
    train_and_export_model()
    

