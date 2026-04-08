# ==============================
# 训练脚本 (SimpleCNN + TensorBoard)
# ==============================

# ------------------------------
# 1. 导入必要库
# ------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

# ------------------------------
# 2. 全局配置
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU/CPU
EPOCHS = 100
BATCH_SIZE = 128

print(f"Using device: {DEVICE}")

# ------------------------------
# 3. 数据预处理与加载
# ------------------------------
'''
TODO rewrite the data input
# 定义训练集数据增强和标准化
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪（带填充）
    transforms.RandomHorizontalFlip(),         # 随机水平翻转（50% 概率）
    transforms.ToTensor(),                     # 转为 PyTorch 张量 [0,1]
    transforms.Normalize(                      # 使用 CIFAR-10 官方均值/标准差
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# 测试集只做标准化（不增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# 加载 CIFAR-10 数据集（注意：root 应指向包含 cifar-10-batches-py/ 的目录）
trainset = torchvision.datasets.CIFAR10(
    root='data/cifar10',      # 数据根目录
    train=True,               # 加载训练集
    download=False,           # 不下载（假设已存在）
    transform=train_transform # 应用训练变换
)

testset = torchvision.datasets.CIFAR10(
    root='data/cifar10',
    train=False,              # 加载测试集
    download=False,
    transform=test_transform  # 应用测试变换（无增强）
)

# 创建数据加载器
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,             # 训练时打乱顺序
    num_workers=2,
    pin_memory=True if DEVICE.type == 'cuda' else False  # GPU 加速
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,    # 测试也用大 batch 提速
    shuffle=False,            # 测试无需打乱
    num_workers=2,
    pin_memory=True if DEVICE.type == 'cuda' else False
)
'''
# ------------------------------
# 4. 定义模型：简单 CNN
# ------------------------------
class SimpleCNN(nn.Module):
    """一个用于 CIFAR-10 分类的轻量级 CNN"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 特征提取层（卷积 + 激活 + 池化）
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入: 32x32x3 → 输出: 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # → 16x16x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → 16x16x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)        # → 8x8x64
        )
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # 展平后输入
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)           # CIFAR-10 有 10 个类别
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)           # 提取特征
        x = torch.flatten(x, 1)        # 展平: [B, 64, 8, 8] → [B, 64*8*8]
        x = self.classifier(x)         # 分类
        return x

# ------------------------------
# 5. 初始化模型、优化器、损失函数
# ------------------------------
model = SimpleCNN().to(DEVICE)  # ✅ 先实例化，再 .to()

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 适用于多分类任务

# ------------------------------
# 6. TensorBoard 日志记录器
# ------------------------------
writer = SummaryWriter(log_dir='runs/cifar10_simplecnn')

# ------------------------------
# 7. 评估函数（计算测试准确率）
# ------------------------------
def evaluate(model, dataloader, device):
    """
    在测试集上评估模型准确率
    Args:
        model: 待评估模型
        dataloader: 测试数据加载器
        device: 设备（CPU/GPU）
    Returns:
        float: 准确率（0~1）
    """
    model.eval()  # 切换到评估模式（关闭 Dropout/BatchNorm 更新）
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ------------------------------
# 8. 训练循环
# ------------------------------
print("开始训练...")
for epoch in range(EPOCHS):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # 将数据移动到指定设备
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播 + 优化
        loss.backward()
        optimizer.step()
        
        # 累计损失（用于日志）
        running_loss += loss.item()
        
        # 每 100 个 batch 记录一次平均损失
        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / 100
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Loss/train', avg_loss, global_step)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
    
    # 每个 epoch 结束后评估测试集
    test_acc = evaluate(model, testloader, DEVICE)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Test Accuracy: {test_acc:.4f}")

# ------------------------------
# 9. 清理资源
# ------------------------------
writer.close()
print("训练完成！TensorBoard 日志已保存至 runs/cifar10_simplecnn")
print("运行以下命令查看结果：\n  tensorboard --logdir=runs")
