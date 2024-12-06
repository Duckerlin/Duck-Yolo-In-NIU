# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：plot_results.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
"""
import pandas as pd
import matplotlib.pyplot as plt


# 训练结果列表
results_files = [
    'D:/123main/runs/exp1/results.csv',
    'D:/123main/runs/exp/results.csv',

]

# 与results_files顺序对应
custom_labels = [
    'yolov8s-EMAttention',
    'yolov8s',


]

#
def plot_comparison(metrics, labels, custom_labels, layout=(2, 2)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10))  # 创建网格布局
    axes = axes.flatten()  # 将子图对象展平，方便迭代

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        for file_path, custom_label in zip(results_files, custom_labels):
            df = pd.read_csv(file_path)

            # 清理列名中的多余空格
            df.columns = df.columns.str.strip()

            # 检查 'epoch' 列是否存在
            if 'epoch' not in df.columns:
                print(f"'epoch' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 检查目标指标列是否存在
            if metric_key not in df.columns:
                print(f"'{metric_key}' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 在对应的子图上绘制线条
            axes[i].plot(df['epoch'], df[metric_key], label=f'{custom_label}')

        axes[i].set_title(f' {metric_label}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric_label)
        axes[i].legend()

    plt.tight_layout()  # 自动调整子图布局，防止重叠
    plt.show()


if __name__ == '__main__':


    # 精度指标
    metrics = [
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]

    labels = [
        'Precision', 'Recall', 'mAP@50', 'mAP@50-95'
    ]

    # 调用通用函数绘制精度对比图
    plot_comparison(metrics, labels, custom_labels, layout=(2, 2))


    # 损失指标
    loss_metrics = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
    ]

    loss_labels = [
        'Train Box Loss', 'Train Class Loss', 'Train DFL Loss', 'Val Box Loss', 'Val Class Loss', 'Val DFL Loss'
    ]

    # 调用通用函数绘制损失对比图
    plot_comparison(loss_metrics, loss_labels, custom_labels, layout=(2, 3))
