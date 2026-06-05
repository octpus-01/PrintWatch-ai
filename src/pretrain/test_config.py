# test_config.py
import os

# --- 修复 1: 设置环境变量以解决 CUDA 异步报错问题 ---
# 这能让报错信息指向确切的代码行，而不是汇编代码
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 修复导入路径，确保 test_models 在同一目录或正确路径下
from test_models import (
    get_yolov26_nano,
    get_faster_net_p2,
    get_convnext_tiny_yolo26,
    get_pp_lcnet_picodet,
    get_repvgg_yolov6s,
    get_vmamba_detect,
    get_mobilevit_s_yolo26,
    get_edgenext_yolo8,
    get_levit,
    get_swin_tiny_maskrcnn,
    get_resnet18_cbam,
    get_resnet18_simam,
)

# 全局训练配置
TRAIN_CONFIG = {
    "device": "cuda",
    "epochs": 5,
    "base_batch_size": 32,
    "num_workers": 0,
    "pin_memory": False,
    "mixed_precision": True,
    "learning_rate": 0.0001,
    "weight_decay": 1e-4,
    "log_dir": "./runs",
    "checkpoint_dir": "./checkpoints",
}

# 数据预处理与路径配置
DATA_CONFIG = {
    "data_root": r"D:\downloads\k\Kaggle_3D_Print_Defect_Dataset",
    "img_size": 224,
    "num_classes": 6,  # 确保类别数正确
}

# 实验列表
EXPERIMENTS = [
    # --- 第一梯队优化 (ResNet系列) ---
    {
        "name": "ResNet_18_SE",
        "model_fn": get_resnet18_simam,  # 这里借用 SimAM 的结构示例，实际建议在 models 中实现 SE
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },
    {
        "name": "ResNet_34_CBAM",
        "model_fn": get_resnet18_cbam,  # 注意：这里需要修改 get_resnet18_cbam 函数支持 resnet34
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },

    # --- 第二梯队架构调整 (ConvNeXt/EdgeNeXt) ---
    {
        "name": "ConvNeXt_Tiny_YOLOv8",
        "model_fn": get_convnext_tiny_yolo26,  # 复用，Head 差异在模型内部处理
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },
    {
        "name": "EdgeNeXt_Small_YOLOv8",
        "model_fn": get_edgenext_yolo8,
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },

    # --- 第三梯队扩容 (轻量级模型加大) ---
    {
        "name": "RepVGG_A1_YOLOv6s",
        "model_fn": get_repvgg_yolov6s,
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },
    {
        "name": "PP_LCNet_1.0x_PicoDet",
        "model_fn": get_pp_lcnet_picodet,
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    },
    {
        "name": "YOLOv26_Small_E2E",
        "model_fn": get_yolov26_nano,
        "params": {"num_classes": DATA_CONFIG["num_classes"]}
    }
]

# --- 增加配置校验 ---
def validate_config():
    for exp in EXPERIMENTS:
        assert "name" in exp, "实验缺少 'name' 字段"
        assert "model_fn" in exp, f"实验 {exp['name']} 缺少 'model_fn' 字段"
        assert "params" in exp, f"实验 {exp['name']} 缺少 'params' 字段"
    print(f"[配置检查] 共发现 {len(EXPERIMENTS)} 个实验配置，通过。")

if __name__ == "__main__":
    validate_config()