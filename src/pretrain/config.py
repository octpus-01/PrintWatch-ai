# config.py
from models import get_resnet18

# 全局训练配置
TRAIN_CONFIG = {
    "device": "cuda",
    "epochs": 15,
    "base_batch_size": 32,      # RTX 2070S 8GB 显存的安全值（开启混合精度后可适当调大）
    "num_workers": 0,
    "pin_memory": True,
    "mixed_precision": True,     # 开启混合精度训练
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "log_dir": "./runs",
    "checkpoint_dir": "./checkpoints",
}

# 数据预处理与路径配置
DATA_CONFIG = {
    "data_root": "./data/raw/defect_data",  # 替换为你实际的 3D 打印缺陷数据集路径
    "img_size": 224,                       # ResNet18 标准输入尺寸
    "num_classes": 5,                     # 你的缺陷类别数量
}

# 实验列表（方便主程序遍历多个模型）
EXPERIMENTS = [
    {
        "name": "ResNet18_Defect_Pretrain",
        "model_fn": get_resnet18,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": True},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    }
]