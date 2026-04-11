import paho.mqtt.client as mqtt
import cv2
import base64
import json
import time
import os
from PIL import Image
import numpy as np

# ====================== MQTT 配置 ======================
MQTT_BROKER = "broker.emqx.io"  
MQTT_PORT = 1883
MQTT_TOPIC = "rl/image/label"
CLIENT_ID = "python_publisher"

# ====================== 数据路径（你的图片+标注） ======================
# 格式：图片路径 → 对应标签
IMAGE_DIR = "images/"  # 新建此文件夹放图片
LABEL_MAP = {
    "cat.jpg": 0,
    "dog.jpg": 1,
    "car.jpg": 2,
    # 按你的分类添加
}

class MQTTPublisher:
    def __init__(self):
        self.client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

    def encode_image(self, img_path):
        """图片转base64字符串"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))  # 统一尺寸（和训练一致）
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def send_data(self, img_name, label):
        """发送 图片+标注"""
        img_data = self.encode_image(os.path.join(IMAGE_DIR, img_name))
        
        # 组装JSON数据
        payload = {
            "image": img_data,
            "label": label,
            "timestamp": time.time()
        }

        # 转JSON并发送
        msg = json.dumps(payload)
        self.client.publish(MQTT_TOPIC, msg, qos=0)
        print(f"✅ 发送成功 | 图片：{img_name} | 标签：{label}")

if __name__ == "__main__":
    publisher = MQTTPublisher()
    
    # 循环发送测试数据
    for img_name, label in LABEL_MAP.items():
        publisher.send_data(img_name, label)
        time.sleep(1)  # 间隔1秒发送