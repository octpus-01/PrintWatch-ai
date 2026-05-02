### super_test_models.py 文件：
- **YOLOv26系列**: 包含YOLOv26-Nano和相关检测头
- **FasterNet**: 包含FasterNet骨干网络和P2检测头
- **ConvNeXt**: 包含ConvNeXt-Tiny和YOLOv26头
- **PP-LCNet**: 包含轻量级骨干网络和PicoDet检测头
- **RepVGG**: 包含可重构卷积网络和YOLOv6s头
- **VMamba**: 包含视觉Mamba模型和检测头
- **MobileViT**: 包含轻量级视觉Transformer和YOLOv26头
- **EdgeNeXt**: 包含边缘优化网络和YOLOv8头
- **LeViT**: 包含轻量级视觉Transformer
- **Swin Transformer**: 包含Swin-Tiny和Mask R-CNN头
- **ResNet-18变体**: 包含CBAM和SimAM注意力机制
- **HRNet**: 包含高分辨率网络和OCR模块
- **Res2Net**: 包含多尺度残差网络和YOLOv7-Tiny头
- **EfficientNet-Lite**: 包含轻量级网络和RetinaNet头

### super_test_config.py 文件：
- 包含全局训练配置参数
- 包含数据预处理配置
- 包含所有15个实验模型的详细配置列表

每个模型都针对3D打印缺陷识别进行了优化，支持分类任务，并考虑了RTX 2070S 8GB显存的限制，在复杂模型上自动降低了批次大小。您可以使用此配置运行长时间的模型比较实验。