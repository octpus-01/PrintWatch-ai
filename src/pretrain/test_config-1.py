# config.py

from test_models import (
    get_yolov26_nano, get_faster_net_p2, get_convnext_tiny_yolo26, 
    get_pp_lcnet_picodet, get_repvgg_yolov6s, get_vmamba_detect, 
    get_mobilevit_s_yolo26, get_edgenext_yolo8, get_levit, 
    get_swin_tiny_maskrcnn, get_resnet18_cbam, get_resnet18_simam, 
)

# 全局训练配置
TRAIN_CONFIG = {
    "device": "cuda",
    "epochs": 5,
    "base_batch_size": 32,      # RTX 2070S 8GB 显存的安全值（开启混合精度后可适当调大）
    "num_workers": 0,
    "pin_memory": False,
    "mixed_precision": True,     # 开启混合精度训练
    "learning_rate": 0.0001,
    "weight_decay": 1e-4,
    "log_dir": "./runs",
    "checkpoint_dir": "./checkpoints",
}

# 数据预处理与路径配置
DATA_CONFIG = {
    "data_root": r"D:\downloads\k\Kaggle_3D_Print_Defect_Dataset",  # 替换为你实际的 3D 打印缺陷数据集路径
    "img_size": 224,                       # 标准输入尺寸
    "num_classes": 6,                     # 你的缺陷类别数量
}

# 实验列表（方便主程序遍历多个模型）
EXPERIMENTS = [
    {
        "name": "YOLOv26_Nano_End_to_End",
        "model_fn": get_yolov26_nano,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "FasterNet_P2_Detect_Head",
        "model_fn": get_faster_net_p2,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "ConvNeXt_Tiny_YOLOv26_Head",
        "model_fn": get_convnext_tiny_yolo26,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "PP_LCNet_PicoDet",
        "model_fn": get_pp_lcnet_picodet,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "RepVGG_YOLOv6s",
        "model_fn": get_repvgg_yolov6s,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "VMamba_Visual_Mamba_Detect_Head",
        "model_fn": get_vmamba_detect,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"] // 2,  # Reduce batch size due to model complexity
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "MobileViT_S_YOLOv26",
        "model_fn": get_mobilevit_s_yolo26,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "EdgeNeXt_YOLOv8",
        "model_fn": get_edgenext_yolo8,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "LeViT_Vision_Transformer",
        "model_fn": get_levit,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": False},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },

    {
        "name": "ResNet_18_CBAM",
        "model_fn": get_resnet18_cbam,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": True},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    },
    {
        "name": "ResNet_18_SimAM",
        "model_fn": get_resnet18_simam,
        "params": {"num_classes": DATA_CONFIG["num_classes"], "pretrained": True},
        "batch_size": TRAIN_CONFIG["base_batch_size"],
        "epochs": TRAIN_CONFIG["epochs"],
    }
]
