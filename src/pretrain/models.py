# models.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

"""
class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=192, depth=12, num_classes=10):
        super().__init__()
        # 这里写你的 ViT 模型具体结构
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        # ... 其他层 ...
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        # ... 前向传播逻辑 ...
        return self.head(x)

class ResNetLike(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 这里写你的 CNN 模型具体结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # ... 其他层 ...
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # ... 前向传播逻辑 ...
        return self.fc(x.mean([2, 3])) # 简单的全局平均池化

"""

def get_resnet18(num_classes=2, pretrained=False):
    """
    构建 ResNet18 模型
    Args:
        num_classes: 分类类别数
        pretrained: 是否加载 ImageNet 预训练权重
    """
    model = resnet18(weights="DEFAULT" if pretrained else None)
    
    # 修改最后的全连接层以适应自定义分类数量
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model