from ultralytics import YOLO

if __name__ == '__main__':
    # 模型路径
    pt_file_address = r"D:\123main\pt\best.pt"  # 修改为你的模型路径

    # YAML 文件路径（数据集配置）
    yaml_file_address = r"D:\123main\ultralytics\cfg\datasets\mydata.yaml"  # 修改为你的数据配置文件路径

    # 加载模型
    model = YOLO(pt_file_address)

    # 执行验证，并指定 YAML 文件路径
    metrics = model.val(data=yaml_file_address, workers=0)  # 避免多线程冲突

    # 打印模型指标
    print("Precision:", metrics.box.p)  # Precision
    print("Recall:", metrics.box.r)  # Recall
    print("mAP@0.5:", metrics.box.map50)  # mAP@0.5
    print("mAP@0.75:", metrics.box.map75)  # mAP@0.75

    # 打印更精确的 Precision 值
    print("Precision (detailed):", metrics.results_dict['metrics/precision(B)'])
