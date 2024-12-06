import torch
import torch.nn as nn

def check_channels(x, expected_channels):
    actual_channels = x.shape[1]
    print(f"Expected channels: {expected_channels}, Actual channels: {actual_channels}")
    return actual_channels == expected_channels

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dim_conv3 = min(in_channels, 96)  # 根据输入通道数动态设置
        self.partial_conv3 = nn.Conv2d(in_channels, self.dim_conv3, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        return self.forward_slicing(x)

    def forward_slicing(self, x):
        x = x.clone()
        print("x1:", x.shape)
        
        # 这里使用 dim_conv3 确保不会超出输入的通道数
        actual_channels = x.shape[1]
        self.dim_conv3 = min(actual_channels, 96)
        print("Updated dim_conv3:", self.dim_conv3)

        # 确保切片不会超出范围
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        print("x2:", x.shape)
        
        x = self.conv(x)
        print("x3:", x.shape)
        return x


class YourModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvModule(in_channels, out_channels)
        self.conv2 = ConvModule(out_channels, out_channels)

    def forward(self, x):
        expected_channels = x.shape[1]
        print("Input shape:", x.shape)

        x = self.conv1(x)
        if not check_channels(x, self.conv1.conv.out_channels):
            raise ValueError("Channel mismatch after conv1!")

        x = self.conv2(x)
        if not check_channels(x, self.conv2.conv.out_channels):
            raise ValueError("Channel mismatch after conv2!")

        return x

# 示例：创建模型并运行前向传播
if __name__ == "__main__":
    model = YourModel(in_channels=3, out_channels=64)
    input_tensor = torch.randn(1, 3, 224, 224)  # 示例输入
    output = model(input_tensor)
    print("Output shape:", output.shape)
