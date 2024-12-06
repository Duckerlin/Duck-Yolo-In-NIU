import torch

# 创建一个简单的模型进行测试
model = torch.nn.Linear(10, 10).cuda()

# 尝试将模型转换为半精度
model.half()

# 创建半精度输入数据
input_data = torch.randn(10).cuda().half()

# 执行前向传播
output = model(input_data)

# 输出结果
print("Output:", output)
