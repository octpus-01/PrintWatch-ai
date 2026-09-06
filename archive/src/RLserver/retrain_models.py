# train_ghostnet.py
"""
GhostNet-100 单模型训练脚本
基于原有环境配置 (RTX 2070 Super) 优化，保留所有显存与训练参数设置
"""

import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import gc
import torch.multiprocessing as mp

# 引入 timm 库用于加载 GhostNet 模型
# 如果报错，请在终端运行: pip install timm
import timm 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 强制加载损坏图片

# ... (之前的 imports)

# ------------------------------
# 辅助类：用于为数据集的子集指定不同的 Transform
# ------------------------------
class TransformSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        # 1. 获取原始数据（不经过 ImageFolder 的 transform）
        img, label = self.dataset[self.indices[idx]]
        
        # 2. 应用当前子集特定的 transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ------------------------------
# 1. 全局配置参数 (保留原设置)
# ------------------------------

TRAIN_CONFIG = {
    "device": "cuda",
    "epochs": 10,
    "base_batch_size": 32,
    "num_workers": 0,
    "persistent_workers": False,
    "pin_memory": False,
    "mixed_precision": True,
    "learning_rate": 0.0001,
    "weight_decay": 1e-4,
    "log_dir": "./runs",
    "checkpoint_dir": "./checkpoints",
}

DATA_CONFIG = {
    "data_root": r"D:\downloads\k\Kaggle_3D_Print_Defect_Dataset",
    "img_size": 224,
    "num_classes": 6,
}

# ------------------------------
# 2. 环境初始化
# ------------------------------

TIMELOCAL = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
DEVICE = torch.device(TRAIN_CONFIG["device"] if torch.cuda.is_available() else "cpu")

# 针对 RTX 2070 Super 的显存优化策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True  # 加速卷积计算

# 确保目录存在
os.makedirs(TRAIN_CONFIG["log_dir"], exist_ok=True)
os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

# ------------------------------
# 3. 数据加载模块 (已修复类型报错与逻辑Bug)
# ------------------------------

def get_dataloaders(batch_size):
    print("⚙️ 加载并预处理数据集...")
    
    # 训练集数据增强
    transform_train = transforms.Compose([
        transforms.Resize(DATA_CONFIG["img_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试集/验证集标准化处理
    transform_test = transforms.Compose([
        transforms.Resize(DATA_CONFIG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. 加载全量数据，注意：这里先不设置 transform，设为 None
    full_dataset = ImageFolder(root=DATA_CONFIG["data_root"], transform=None)
    
    # 2. 手动划分索引 (80% 训练, 20% 验证)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 生成随机索引并打乱
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 3. 使用自定义的 TransformSubset 分别包装训练集和验证集
    # 这样彻底解决了 Pylance 报错，也避免了训练集和验证集共享同一个 transform 的问题
    trainset = TransformSubset(full_dataset, train_indices, transform=transform_train)
    valset = TransformSubset(full_dataset, val_indices, transform=transform_test)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=TRAIN_CONFIG["num_workers"], 
        pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    testloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, 
        num_workers=TRAIN_CONFIG["num_workers"], 
        pin_memory=TRAIN_CONFIG["pin_memory"]
    )

    print(f"✅ 识别到的类别标签: {full_dataset.class_to_idx}")
    return trainloader, testloader, len(full_dataset.classes)


# ------------------------------
# 4. 训练与验证核心逻辑
# ------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        
        # 混合精度训练加速
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

        # TensorBoard 实时记录
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Train/Running_Loss', running_loss/(batch_idx+1), global_step)
        writer.add_scalar('Train/Batch_Acc', 100.*correct/total, global_step)

        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch}] Batch {batch_idx}/{len(dataloader)} "
                  f"Loss: {running_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Epoch_Accuracy', epoch_acc, epoch)
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
# 5. 主程序入口
# ------------------------------

if __name__ == "__main__":
    print(f"🚀 GhostNet-100 训练开始时间：{TIMELOCAL}")
    print(f"🖥️ 当前设备: {DEVICE}")
    print(f"⚡ 混合精度: {'ON' if TRAIN_CONFIG['mixed_precision'] else 'OFF'}\n")
    
    # Windows 多进程设置
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy('file_system')

    # 1. 准备数据
    trainloader, testloader, num_classes = get_dataloaders(TRAIN_CONFIG["base_batch_size"])
    
    # 2. 构建 GhostNet-100 模型
    print("🏗️ 正在构建 GhostNet-100 模型...")
    try:
        # 使用 timm 加载 ghostnet_100，pretrained=True 使用 ImageNet 预训练权重
        model = timm.create_model('ghostnet_100', pretrained=True, num_classes=DATA_CONFIG["num_classes"])
    except Exception as e:
        print(f"❌ 模型加载失败，请检查是否安装 timm 库 (pip install timm)。错误: {e}")
        exit()
        
    model = model.to(DEVICE)

    # 3. 优化器与损失
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], 
                      weight_decay=TRAIN_CONFIG["weight_decay"], fused=True)
    scaler = GradScaler(device='cuda', enabled=TRAIN_CONFIG["mixed_precision"])

    # 4. 日志记录器
    log_path = os.path.join(TRAIN_CONFIG["log_dir"], f"GhostNet_100_{TIMELOCAL}")
    writer = SummaryWriter(log_dir=log_path)
    
    best_acc = 0
    print(f"{'='*40} 开始训练 {'='*40}")

    # 5. 训练循环
    for epoch in range(TRAIN_CONFIG["epochs"]):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, scaler, epoch, writer)
        test_loss, test_acc = test(model, testloader, criterion, epoch, writer)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], f"GhostNet_100_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 最佳模型已保存 (Acc: {best_acc:.2f}%)")

    writer.close()
    
    # 清理显存
    del model, optimizer, criterion, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🎉 训练完成！")
    print(f"📊 查看日志命令: tensorboard --logdir={TRAIN_CONFIG['log_dir']}")
