# Raspi 4B use
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

#TODO:在树莓派上测试！

# =================配置区域=================
ONNX_MODEL_PATH = "model.onnx"  # 你的ONNX模型文件名
INPUT_SIZE = (224, 224)         # 模型期望的输入尺寸 (宽, 高)
CONFIDENCE_THRESHOLD = 0.5      # 置信度阈值
IMGDIR = "/data"


def preprocess_image(image, input_size):
    """
    将 picamera2 采集的图像转换为模型所需的输入格式
    """
    # 1. 缩放图像 (OpenCV 默认是 BGR，如果模型是 RGB 训练的需要转换)
    # 这里假设模型是 RGB 训练，所以将 BGR 转为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, input_size)
    
    # 2. 归一化到 
    normalized = resized.astype(np.float32) / 255.0
    
    # 3. 标准化 (假设使用 ImageNet 的均值和方差，如果是自定义模型请修改)
    # 格式：(H, W, C) -> (C, H, W)
    # 这一步将数据从 通道在后 变为 通道在前
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 减去均值并除以方差
    # 注意：这里需要针对每个通道操作，np.transpose 将 (H,W,C) 变为 (C,H,W)
    normalized = (np.transpose(normalized, (2, 0, 1)) - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)
    
    # 4. 增加批次维度 (1, C, H, W)
    input_tensor = np.expand_dims(normalized, axis=0)
    
    return input_tensor

def main():
    # 1. 初始化 ONNX Runtime 会话
    try:
        # 强制使用 CPU，树莓派通常没有 CUDA
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs().name
        print(f"✅ ONNX 模型加载成功，输入节点名称: {input_name}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 2. 初始化摄像头
    picam2 = Picamera2()
    # 配置预览模式，分辨率可以设小一点以提高推理速度，比如 640x480
    config = picam2.create_preview_configuration(main={"size": (840, 640)})
    picam2.configure(config)
    picam2.start()
    print("📷 摄像头已启动，正在预热...")
    time.sleep(2) # 等待摄像头自动曝光稳定

    print("🚀 开始推理 (按 Ctrl+C 退出)...")
    
    try:
        while True:
            # 3. 获取图像 (picamera2 捕获的是 numpy 数组)
            frame = picam2.capture_array()
            stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            output_img = picam2.capture_file(IMGDIR + "/inferdata/inferimg_%s".format(stamp))

            
            # 4. 预处理
            model_input = preprocess_image(frame, INPUT_SIZE)
            
            # 5. 执行推理
            # run 返回的是一个列表，包含所有输出节点
            outputs = session.run(None, {input_name: model_input})
            result = outputs

            # 简单示例：如果是分类模型，打印概率最高的类别
            class_id = np.argmax(result)
            score = np.max(result)
            print(f"类别: {class_id}, 置信度: {score:.4f}")

    except KeyboardInterrupt:
        print("\n停止推理...")
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()


