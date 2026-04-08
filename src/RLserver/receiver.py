import paho.mqtt.client as mqtt
import json
import base64
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ====================== 导入你之前的强化学习代码 ======================
from rl_agent import RLAgent, ClassifyEnv

# ====================== MQTT 配置 ======================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "rl/image/label"
CLIENT_ID = "python_subscriber_rl"

# ====================== 训练配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class MQTTSubscriberRL:
    def __init__(self):
        # 初始化强化学习智能体
        self.agent = RLAgent(num_classes=NUM_CLASSES, pretrained_model=None)
        self.client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
        
        # 绑定回调函数
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print("✅ 已连接MQTT服务器，等待接收图片数据...")
        client.subscribe(MQTT_TOPIC)

    def decode_image(self, img_data):
        """base64 转回图片 tensor"""
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return transform(img).unsqueeze(0).to(DEVICE)

    def on_message(self, client, userdata, msg):
        """接收数据 → 解码 → 强化学习训练"""
        payload = json.loads(msg.payload.decode())
        img_base64 = payload["image"]
        true_label = payload["label"]
        
        # 1. 恢复图像
        state = self.decode_image(img_base64)
        
        # 2. 强化学习决策
        action = self.agent.select_action(state)
        
        # 3. 奖励函数（正确+1，错误-1）
        reward = 1.0 if action == true_label else -1.0
        
        # 4. 存入经验池
        next_state = state  # 分类任务单步完成
        done = True
        self.agent.buffer.push(state, action, reward, next_state, done)
        
        # 5. 执行强化学习更新
        self.agent.update()
        
        print(f"📥 接收并训练 | 标签：{true_label} | 预测：{action} | 奖励：{reward}")

    def start(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_forever()

if __name__ == "__main__":
    subscriber = MQTTSubscriberRL()
    subscriber.start()