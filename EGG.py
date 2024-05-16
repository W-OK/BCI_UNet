import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("path/to/PyEEG")

# 文件夹路径
train_dir = Path("./Fake_BCI_Dataset/train")
test_dir = Path("./Fake_BCI_Dataset/test")

# 创建文件夹（如果不存在）
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.delta_generator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )
        self.theta_generator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )
        self.alpha_generator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )
        self.beta_generator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )
        self.gamma_generator = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_size),
            nn.Tanh()
        )

        # 将模型的权重参数转换为双精度（Double）类型
        for module in [self.delta_generator, self.theta_generator, self.alpha_generator, self.beta_generator, self.gamma_generator]:
            for layer in module.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.data = layer.weight.data.double()
                    layer.bias.data = layer.bias.data.double()

    def forward(self, x, noise_type):
        x = x.double()  # 将输入数据类型更改为双精度（Double）
        if noise_type == 'delta':
            x = self.delta_generator(x)
        elif noise_type == 'theta':
            x = self.theta_generator(x)
        elif noise_type == 'alpha':
            x = self.alpha_generator(x)
        elif noise_type == 'beta':
            x = self.beta_generator(x)
        elif noise_type == 'gamma':
            x = self.gamma_generator(x)
        return x

# 生成器和优化器
input_size = 1000
output_size = 10000
generator = Generator(input_size, output_size)
optimizer = optim.Adam(generator.parameters(), lr=0.001)

# 数据预处理（使用虚构的 EEG 数据代替实际数据）
def generate_fake_data(batch_size, output_size):
    return torch.randn(batch_size, output_size, dtype=torch.double, requires_grad=True)  # 确保数据类型为双精度，并且需要梯度

# 添加更多噪声和变化
def add_noise(signal):
    noise = np.random.normal(0, 0.1, signal.shape)  # 正态分布噪声
    return signal + noise

# 数据预处理（平滑处理）
def smooth(signal, window_len=11, window='hanning'):
    s = np.r_[2 * signal[0] - signal[window_len:1:-1], signal, 2 * signal[-1] - signal[-1:-window_len:-1]]
    if window == 'flat':  # 移动平均
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[window_len // 2:len(y) - window_len // 2]

# 训练生成器
num_epochs = 30000
batch_size = 1
losses = []

for epoch in range(num_epochs):
    noise = torch.randn(batch_size, input_size, dtype=torch.double, requires_grad=True)  # 确保噪声数据类型为双精度，并且需要梯度
    # 随机选择要生成的波段
    noise_type = np.random.choice(['delta', 'theta', 'alpha', 'beta', 'gamma'])
    fake_data = generator(noise, noise_type)

    # 添加更多噪声和变化
    fake_data_numpy = fake_data.squeeze().detach().numpy()
    fake_data_numpy = add_noise(fake_data_numpy)
    fake_data_numpy = smooth(fake_data_numpy)

    # 数据预处理（归一化）
    fake_data = torch.tensor(fake_data_numpy, dtype=torch.double, requires_grad=True).unsqueeze(0)  # 确保数据类型为双精度，并且需要梯度
    fake_data = (fake_data - torch.min(fake_data)) / (torch.max(fake_data) - torch.min(fake_data))

    # 计算损失
    loss = nn.L1Loss()(fake_data, generate_fake_data(batch_size, output_size))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# 可视化训练过程中的损失值
plt.plot(losses)
plt.title('Generator Training Loss')
plt.xlabel('Epochs')
plt.ylabel('L1 Loss')
plt.show()

# 生成并保存图像
num_images_per_folder = 1000
num_images = 2 * num_images_per_folder

for i in range(num_images):
    noise = torch.randn(1, input_size, dtype=torch.double, requires_grad=True)  # 确保噪声数据类型为双精度，并且需要梯度
    noise_type = np.random.choice(['delta', 'theta', 'alpha', 'beta', 'gamma'])
    fake_data = generator(noise, noise_type)

    # 添加更多噪声和变化
    fake_data_numpy = fake_data.squeeze().detach().numpy()
    fake_data_numpy = add_noise(fake_data_numpy)
    fake_data_numpy = smooth(fake_data_numpy)

    # 数据预处理（归一化）
    fake_data = torch.tensor(fake_data_numpy, dtype=torch.double, requires_grad=True).unsqueeze(0)  # 确保数据类型为双精度，并且需要梯度
    fake_data = (fake_data - torch.min(fake_data)) / (torch.max(fake_data) - torch.min(fake_data))

    # 生成图像
    plt.figure(figsize=(10, 4))
    plt.plot(fake_data.squeeze().detach().numpy())
    plt.title(f'Generated EEG Signal - Image {i + 1} ({noise_type} band)')
    plt.axis('off')

    # 保存图像到训练或测试文件夹
    if i < num_images_per_folder:
        plt.savefig(train_dir / f"generated_image_{i + 1}.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(test_dir / f"generated_image_{i + 1 - num_images_per_folder}.png", bbox_inches='tight', pad_inches=0)

    plt.close()

    if (i + 1) % 10 == 0:
        print(f'Generated image {i + 1}/{num_images}, Loss: {loss.item()}')
