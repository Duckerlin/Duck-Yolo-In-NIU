# -*- coding: utf-8 -*-
"""
@Auth ： 掛科邊緣
@File ：plot_results.py
@IDE ：PyCharm
@Motto:學習新思想，爭做新青年
"""

import pandas as pd
import matplotlib.pyplot as plt

# 訓練結果文件列表與模型標簽 (各種指標之圖表)
results_files = [
    'D:/123/123main/runs/exp (_init_)/results.csv',
    'D:/123/123main/runs/exp1 (other)/results.csv',
]

# 與results_files順序對應
custom_labels = [
    'yolov8s',
    'yolov8s-AKConv Lightest',
]

# PR 和 F1 曲線文件路徑（for F1、PR曲線）
pr_csv_dict = {
    'YOLOv8s': r'D:\123\123main\runs\exp (PR  _init_)\PR_curve.csv',
    'YOLOv8s-AKConv Lightest': r'D:\123\123main\runs\exp1 (PR other)\PR_curve.csv',
}

f1_csv_dict = {
    'YOLOv8s': r'D:\123\123main\runs\exp (F1 _init_)\F1_curve.csv',
    'YOLOv8s-AKConv Lightest': r'D:\123\123main\runs\exp1 (F1 other)\F1_curve.csv',
}

# 通用繪圖函數
def plot_comparison(metrics, labels, custom_labels, layout=(2, 2)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10))  # 創建網格布局
    axes = axes.flatten()  # 將子圖對象展平，方便叠代

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        for file_path, custom_label in zip(results_files, custom_labels):
            df = pd.read_csv(file_path)

            # 清理列名中的多余空格
            df.columns = df.columns.str.strip()

            # 檢查 'epoch' 列是否存在
            if 'epoch' not in df.columns:
                print(f"'epoch' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 檢查目標指標列是否存在
            if metric_key not in df.columns:
                print(f"'{metric_key}' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 在對應的子圖上繪制線條
            axes[i].plot(df['epoch'], df[metric_key], label=f'{custom_label}')

        axes[i].set_title(f'{metric_label}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric_label)
        axes[i].legend()

    plt.tight_layout()  # 自動調整子圖布局，防止重疊
    plt.show()


# 繪制 PR 曲線
def plot_PR():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname, res_path in pr_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    #fig.savefig("pr.png", dpi=250)
    plt.show()


# 繪制 F1 曲線
def plot_F1():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname, res_path in f1_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
   # fig.savefig("F1.png", dpi=250)
    plt.show()


if __name__ == '__main__':
    # 繪制精度對比圖
    metrics = [
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]
    labels = [
        'Precision', 'Recall', 'mAP@50', 'mAP@50-95'
    ]
    plot_comparison(metrics, labels, custom_labels, layout=(2, 2))

    # 繪制損失對比圖
    loss_metrics = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
    ]
    loss_labels = [
        'Train Box Loss', 'Train Class Loss', 'Train DFL Loss', 'Val Box Loss', 'Val Class Loss', 'Val DFL Loss'
    ]
    plot_comparison(loss_metrics, loss_labels, custom_labels, layout=(2, 3))

    # 繪制 PR 曲線
    plot_PR()

    # 繪制 F1 曲線
    plot_F1()


