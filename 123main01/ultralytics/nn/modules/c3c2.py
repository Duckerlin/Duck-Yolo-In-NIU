import torch
import torch.nn as nn

class C3c2(nn.Module):  # 改善通道數
    def __init__(self, in_channels, out_channels):
        super(C3c2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv1(x) + self.conv2(x))
