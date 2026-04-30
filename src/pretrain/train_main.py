# train_main.py
"""
Pytorch模型预训练代码：
- 在独特的训练集上进行训练
- 利用RTX 2070 Super进行开发
- 利用tensorboard监控
"""

import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp import autocast, GradScaler
# ------------------------------
# 1. 全局配置（RTX 2070 Super 专属）
# ------------------------------
TIMELOCAL = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BASE_BATCH_SIZE = 512          # RTX 2070 Super 8GB 显存可轻松支持
NUM_WORKERS = 2                # Windows 避免内存爆炸（原4→2）
PIN_MEMORY = True
MIXED_PRECISION = True
LOG_DIR = "./runs/pretrain_%s".format(TIMELOCAL)

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------
# 2. 数据预处理
# ------------------------------
print("⚙️  加载并预处理数据集...")

"""
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar10', train=True, download=False, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='./data/cifar10', train=False, download=False, transform=transform_test
)

# 关键：NUM_WORKERS=2 避免 Windows 页面文件不足
trainloader = DataLoader(
    trainset,
    batch_size=BASE_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True  # 减少 worker 重建开销
)

testloader = DataLoader(
    testset,
    batch_size=BASE_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)
"""
# ------------------------------
# 3. 模型、优化器、损失函数
# ------------------------------
print("🧠 构建 ResNet18 模型...")

model = torchvision.models.resnet18(weights=None, num_classes=10)
model = model.to(DEVICE)

# 启用 cuDNN 自动调优（加速 CNN）
torch.backends.cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, fused=True)  # fused=True 减少 kernel launch

# 混合精度训练组件
scaler = GradScaler("cuda",enabled=MIXED_PRECISION)

# TensorBoard 日志
writer = SummaryWriter(log_dir=LOG_DIR)

# ------------------------------
# 4. 训练函数
# ------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # 混合精度训练（关键！）
        with autocast("cuda",enabled=MIXED_PRECISION):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 缩放损失 + 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(trainloader)} "
                  f"Loss: {running_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.2f}%")

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    return epoch_loss, epoch_acc

# ------------------------------
# 5. 测试函数
# ------------------------------
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', acc, epoch)
    print(f"\n✅ Test Epoch {epoch+1}: Loss={avg_loss:.3f}, Acc={acc:.2f}%\n")
    return avg_loss, acc

# ------------------------------
# 6. 主训练循环
# ------------------------------
if __name__ == "__main__":
    print(f"   训练开始时间：{TIMELOCAL}")
    print(f"🚀 开始训练！设备: {DEVICE}")
    print(f"   Batch Size: {BASE_BATCH_SIZE}")
    print(f"   Num Workers: {NUM_WORKERS}")
    print(f"   Mixed Precision: {'ON' if MIXED_PRECISION else 'OFF'}\n")

    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        test_loss, test_acc = test(epoch)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 模型已保存 (Acc: {best_acc:.2f}%)")

    writer.close()
    print(f"\n🎉 训练完成！最佳测试准确率: {best_acc:.2f}%")
    print(f"📊 TensorBoard 日志: tensorboard --logdir={LOG_DIR}")