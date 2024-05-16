import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Subset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# EEG 数据集定义
class BCIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(self.image_filenames[idx].split("_")[-1].split(".")[0])  # Convert label to integer
        label_tensor = torch.tensor(label).float() / 100.0  # Convert label to float and scale to [0, 1]

        # 修改标签张量的形状以匹配模型输出的形状
        label_tensor = label_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return image, label_tensor



# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)  # 应该沿着倒数第一个维度进行 softmax 操作
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x
        return out

# 用于BCI任务的改进UNet，具有额外的特征通道和更深的架构，以及自注意力机制
class BCI_UNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_feature_channels=5):
        super(BCI_UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Attention module
        self.attention = SelfAttention(512)

        # Decoder
        self.dec1 = self.conv_block(512 + 256, 256)  # Updated decoder input channels
        self.dec2 = self.conv_block(256 + 128, 128)  # Updated decoder input channels
        self.dec3 = self.conv_block(128 + 64, 64)    # Updated decoder input channels

        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2)

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, feature_channels):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.dropout(F.relu(e1))  # Apply dropout after ReLU
        e2 = self.enc2(self.pool(e1))
        e2 = self.dropout(F.relu(e2))  # Apply dropout after ReLU
        e3 = self.enc3(self.pool(e2))
        e3 = self.dropout(F.relu(e3))  # Apply dropout after ReLU
        e4 = self.enc4(self.pool(e3))

        # Attention mechanism
        e4 = self.attention(e4)

        # Decoder with skip connections
        d1 = torch.cat([self.upsample(e4), e3], 1)
        d1 = self.dec1(d1)
        d1 = self.dropout(F.relu(d1))  # Apply dropout after ReLU

        d2 = torch.cat([self.upsample(d1), e2], 1)
        d2 = self.dec2(d2)
        d2 = self.dropout(F.relu(d2))  # Apply dropout after ReLU

        d3 = torch.cat([self.upsample(d2), e1], 1)
        d3 = self.dec3(d3)
        d3 = self.dropout(F.relu(d3))  # Apply dropout after ReLU

        # Output layer
        out = self.out_conv(d3)
        return out

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加批量归一化层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加批量归一化层
            nn.ReLU(inplace=True)
        )

    def visualize_attention(self, input_image):
        self.eval()
        with torch.no_grad():
            input_image = input_image.to(device)
            feature_channels = torch.randn(1, 5, input_image.size(2), input_image.size(3)).to(device)
            e1 = self.enc1(input_image)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            e4 = self.attention(e4)
            attention_map = self.attention.gamma.squeeze().cpu().numpy()

            if np.isnan(attention_map).any():
                attention_map = np.nan_to_num(attention_map)

            attention_map = torch.tensor(attention_map)
            attention_map = torch.clamp(attention_map, min=0.0)
            attention_map /= torch.max(attention_map)

            weighted_image = torch.mul(input_image, attention_map.unsqueeze(0).unsqueeze(0))

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(input_image.squeeze().cpu().numpy().transpose(1, 2, 0))
            axs[0].set_title('Input Image')

            axs[1].imshow(weighted_image.squeeze().cpu().numpy().transpose(1, 2, 0))
            axs[1].set_title('Weighted Image')

            plt.show()


# 数据转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# 加载最佳模型的权重
checkpoint_path = 'best_model_checkpoint.pth'

# 创建DataLoader
dataset = BCIDataset(data_dir="./Fake_BCI_Dataset/train", transform=transform)
test_dataset = BCIDataset(data_dir="./Fake_BCI_Dataset/test", transform=transform)

# 重新划分数据集
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# 使用random_split函数重新划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建新的数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 迁移学习：加载预训练的ResNet模型
pretrained_model = models.resnet18(pretrained=True)
num_ftrs = pretrained_model.fc.in_features

# 修改BCI_UNet的输出层以匹配ResNet模型中的类数
model = BCI_UNet(3, 1, num_ftrs).to(device)

# 从预训练的ResNet加载权重到BCI_UNet的编码器层
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 使用二分类的交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

# 训练循环
num_epochs = 20  # 根据需要调整
best_val_loss = float('inf')

# 训练循环
for epoch in range(num_epochs):
    # 训练
    model.train()
    total_train_loss = 0
    # 在训练循环中
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)

        # 生成额外的特征通道（随机噪声或其他相关特征）
        feature_channels = torch.randn(inputs.size(0), 5, inputs.size(2), inputs.size(3)).to(device)

        outputs = model(inputs, feature_channels)
        outputs = torch.sigmoid(outputs)  # 使用 sigmoid 激活函数处理输出

        # 根据需要调整标签张量的形状以匹配模型输出的形状
        labels = labels.repeat(1, 1, inputs.size(2), inputs.size(3))  # 或者使用 labels = labels.view(batch_size, 1, height, width)

        # 计算损失
        loss = criterion(outputs, labels.float() / 100.0)  # 计算二元交叉熵损失
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0)  # 设置裁剪的阈值为1.0

        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        # 验证循环
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 生成额外的特征通道（随机噪声或其他相关特征）
            feature_channels = torch.randn(inputs.size(0), 5, inputs.size(2), inputs.size(3)).to(device)

            outputs = model(inputs, feature_channels)

            # 根据需要调整标签张量的形状以匹配模型输出的形状
            labels = labels.repeat(1, 1, inputs.size(2),
                                   inputs.size(3))  # 或者使用 labels = labels.view(batch_size, 1, height, width)

            # 计算损失
            loss = criterion(outputs, labels.float() / 100.0)  # 计算二元交叉熵损失

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')

    scheduler.step(average_val_loss)

# 在整个数据集上进行最终的可视化
total_test_loss = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        feature_channels = torch.randn(inputs.size(0), 5, inputs.size(2), inputs.size(3)).to(device)
        outputs = model(inputs, feature_channels)
        labels = labels.repeat(1, 1, inputs.size(2), inputs.size(3))  # 或者使用 labels = labels.view(batch_size, 1, height, width)
        labels = labels.float() / 100.0  # 转换成浮点型并缩放到 [0, 1] 范围内
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

average_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {average_test_loss:.4f}")

# 在整个数据集上进行最终的可视化
sample_image, _ = test_dataset[0]  # 获取测试集的第一个样本图像
sample_image = sample_image.unsqueeze(0).to(device)  # 将图像添加一个维度并移到设备上

# 调用 visualize_attention 方法
model.visualize_attention(sample_image)

# 在整个数据集上进行最终的可视化
sample_image, sample_label = test_dataset[0]  # 获取测试集的第一个样本图像和标签
sample_image = sample_image.unsqueeze(0).to(device)  # 将图像添加一个维度并移到设备上

# 使用训练好的模型获取模型的预测值
with torch.no_grad():
    model.eval()
    feature_channels = torch.randn(1, 5, sample_image.size(2), sample_image.size(3)).to(device)
    predicted_output = model(sample_image, feature_channels)
    predicted_output = torch.sigmoid(predicted_output)  # 使用 sigmoid 激活函数处理输出

# 将预测输出的张量转换为 NumPy 数组，并提取所需的信息
predicted_output_value = predicted_output.cpu().numpy().squeeze()  # 将张量转换为 NumPy 数组，并去除多余的维度

# 输出预测值和真实值
print("Predicted Output:", predicted_output_value)
print("True Label:", sample_label.item() / 100.0)  # 由于标签经过了缩放，需要除以100来得到真实值

# 从验证集中获取真实值
true_values = []  # 用于存储真实值
for inputs, labels in val_loader:
    true_values.extend(labels.cpu().numpy().flatten().tolist())  # 将标签转换为 NumPy 数组并扁平化

# 从模型预测结果中获取预测值
predicted_values = []  # 用于存储预测值
with torch.no_grad():
    model.eval()
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        feature_channels = torch.randn(inputs.size(0), 5, inputs.size(2), inputs.size(3)).to(device)
        outputs = model(inputs, feature_channels)
        outputs = torch.sigmoid(outputs)  # 使用 sigmoid 激活函数处理输出
        predicted_values.extend(outputs.cpu().numpy().flatten().tolist())  # 将预测输出转换为 NumPy 数组并扁平化

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(true_values, label='True Values', color='blue')
plt.plot(predicted_values, label='Predicted Values', color='red', linestyle='dashed')
plt.xlabel('Sample Index')  # 样本序号或时间
plt.ylabel('EEG Signal Prediction')  # 脑电信号预测值
plt.title('Comparison of Predicted Values and True Values on Validation Set')
plt.legend()
plt.grid(True)
plt.show()