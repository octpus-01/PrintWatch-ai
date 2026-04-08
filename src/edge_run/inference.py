# Raspi 4B use
import numpy as np

import os
import onnxruntime as ort
import time

def load_camera(data_dir):
    """
    从原始二进制文件加载 CIFAR-10 测试集
    返回: images (N, 32, 32, 3), labels (N,)
    """
#TODO WRITE THE INPUT

    # 转换为 (N, 3, 32, 32) -> (N, 32, 32, 3)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images.astype(np.float32), np.array(labels)

def normalize_image(image):
    """
    应用与训练时相同的归一化
    CIFAR-10 常用: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    """
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    return ((image / 255.0 - mean) / std).astype(np.float32)

def main():
    # 1. 加载测试数据
    print("正在加载 CIFAR-10 测试集...")
    images, labels = load_cifar10_test_data('data/cifar-10-batches-py')
    print(f"加载完成: {images.shape[0]} 张图像")
    
    # 2. 预处理：归一化 + 转为 channel-first (N, 3, 32, 32)
    print("正在预处理图像...")
    processed_images = np.stack([
        normalize_image(img).transpose(2, 0, 1)  # (32,32,3) -> (3,32,32)
        for img in images
    ], axis=0)  # shape: (10000, 3, 32, 32)
    
    # 确保数据类型为 float32
    processed_images = processed_images.astype(np.float32)
    
    # 3. 加载 ONNX 模型
    print("正在加载 ONNX 模型...")
    session = ort.InferenceSession("models/resnet18_model.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 4. 批量推理（避免内存溢出）
    print("开始推理...")
    batch_size = 32
    correct = 0
    total = 0
    num_batches = images.shape[0] // batch_size + 1
    pbar = tqdm(total=num_batches, desc="推理", unit="batch")

    start_time = time.time()
    for i in range(0, len(processed_images), batch_size):
        batch_images = processed_images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # 推理
        outputs = session.run([output_name], {input_name: batch_images})
        predictions = np.argmax(outputs[0], axis=1) #type:ignore
        
        # 统计正确数
        correct += np.sum(predictions == batch_labels)
        total += len(batch_labels)
        
        pbar.update(1)

    pbar.close()
    end_time = time.time()    
    # 5. 计算准确率
    accuracy = correct / total
    duration = round(end_time - start_time, 4)
    print(f"\n✅ 最终准确率: {accuracy:.4f} ({correct}/{total})，耗时{duration}s")

if __name__ == "__main__":
    main()




