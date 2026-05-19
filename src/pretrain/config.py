from models import get_resnet18

# 全局训练配置
TRAIN_CONFIG = {
    "device": "cuda",
    "epochs": 10,
    "base_batch_size": 32,  
    "num_workers": 0,
    "persistent_workers": False,            # ✅ Windows 必须 0，否则卡死
    "pin_memory": False,         # ✅ 必须关，否则爆显存
    "mixed_precision": True,     # 保持开启
    "learning_rate": 0.0001,     # ✅ 从 0.001 改成 0.0001（loss 不抖动）
    "weight_decay": 1e-4,
    "log_dir": "./runs",
    "checkpoint_dir": "./checkpoints",
}

# 数据预处理与路径配置
DATA_CONFIG = {
    "data_root": r"D:\downloads\k\Kaggle_3D_Print_Defect_Dataset",  # 加 r 防止路径错误
    "img_size": (360, 620),       # ✅ 等比例、不拉伸、稳定训练
    "num_classes": 6,             # ✅ 正确 6 类
}

# 实验列表
EXPERIMENTS = [
    {
        "name": "ResNet18_Defect_Pretrain",
        "model_fn": get_resnet18,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": True},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    }
]