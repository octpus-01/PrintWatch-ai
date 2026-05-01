# train_main.py
"""
PyTorch 模型预训练主程序
- 支持多模型遍历训练
- 针对 RTX 2070 Super (8GB) 显存优化
- 集成 TensorBoard 监控与断点续训
"""
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import gc

from config import TRAIN_CONFIG, DATA_CONFIG, EXPERIMENTS

# ------------------------------
# 1. 全局环境初始化
# ------------------------------
TIMELOCAL = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
DEVICE = torch.device(TRAIN_CONFIG["device"] if torch.cuda.is_available() else "cpu")

# 防止显存碎片化导致 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# 启用 cuDNN 自动调优（加速卷积计算）
torch.backends.cudnn.benchmark = True

# 确保日志和模型保存目录存在
os.makedirs(TRAIN_CONFIG["log_dir"], exist_ok=True)
os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

# ------------------------------
# 2. 数据预处理与加载
# ------------------------------
def get_dataloaders(batch_size):
    print("⚙️  加载并预处理数据集...")
    
    # ResNet18 标准预处理流程（归一化参数来自 ImageNet）
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(DATA_CONFIG["img_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(DATA_CONFIG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 假设你的数据集按文件夹分类存放（ImageFolder 格式）
    # 例如：data/defect_dataset/train/class1/xxx.jpg
    trainset = ImageFolder(root=os.path.join(DATA_CONFIG["data_root"], 'train'), transform=transform_train)
    testset = ImageFolder(root=os.path.join(DATA_CONFIG["data_root"], 'val'), transform=transform_test)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"], pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=TRAIN_CONFIG["num_workers"], pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    return trainloader, testloader, len(trainset.classes)

# ------------------------------
# 3. 训练与测试核心函数
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        # 混合精度训练（大幅降低 8GB 显存占用）
        with autocast(device_type='cuda', enabled=TRAIN_CONFIG["mixed_precision"]):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch}] Batch {batch_idx}/{len(dataloader)} "
                  f"Loss: {running_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    return epoch_loss, epoch_acc

def test(model, dataloader, criterion, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(dataloader)
    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', acc, epoch)
    print(f"✅ Test Epoch {epoch}: Loss={avg_loss:.3f}, Acc={acc:.2f}%")
    return avg_loss, acc

# ------------------------------
# 4. 单个实验运行逻辑
# ------------------------------
def run_experiment(exp_config):
    print(f"\n{'='*40} 开始实验: {exp_config['name']} {'='*40}")
    
    # 1. 准备数据
    trainloader, testloader, num_classes = get_dataloaders(exp_config["batch_size"])
    
    # 2. 构建模型
    model = exp_config["model_fn"](**exp_config["params"]).to(DEVICE)
    
    # 3. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"], fused=True)
    scaler = GradScaler(device='cuda', enabled=TRAIN_CONFIG["mixed_precision"])
    
    # 4. TensorBoard 日志
    log_path = os.path.join(TRAIN_CONFIG["log_dir"], f"{exp_config['name']}_{TIMELOCAL}")
    writer = SummaryWriter(log_dir=log_path)
    
    best_acc = 0
    # 5. 训练循环
    for epoch in range(exp_config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, scaler, epoch, writer)
        test_loss, test_acc = test(model, testloader, criterion, epoch, writer)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], f"{exp_config['name']}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存 (Acc: {best_acc:.2f}%)")

    writer.close()
    
    # 6. 实验结束，彻底清理显存，防止下一个模型 OOM
    del model, optimizer, criterion, scaler
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🧹 实验 {exp_config['name']} 结束，显存已清理。\n")

# ------------------------------
# 5. 主程序入口
# ------------------------------
if __name__ == "__main__":
    print(f"🚀 训练开始时间：{TIMELOCAL}")
    print(f"🖥️  当前设备: {DEVICE}")
    print(f"⚡ 混合精度: {'ON' if TRAIN_CONFIG['mixed_precision'] else 'OFF'}\n")

    # 遍历配置中的所有实验
    for exp in EXPERIMENTS:
        try:
            run_experiment(exp)
        except Exception as e:
            print(f"❌ 实验 {exp['name']} 发生错误: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    print("\n🎉 所有实验运行完成！")
    print(f"📊 查看日志命令: tensorboard --logdir={TRAIN_CONFIG['log_dir']}")