好的，使用 MQTT 在 Python 中传输图片和标注数据是一个很实用的方案。核心思路是：**将图片文件读取为字节流（bytes），然后与标注信息一起打包成一个结构化的消息（例如 JSON），再通过 MQTT 发送出去。接收端收到消息后，解析并分别保存图片和标注数据到指定文件夹。**

以下是详细的步骤和代码示例：

### **1. 核心概念与准备**

*   **MQTT Broker**: 你需要一个 MQTT 服务器（Broker）来中转消息。对于本地开发和测试，推荐使用开源的 `Mosquitto`。
    *   **安装 Mosquitto (Ubuntu/Debian)**: `sudo apt install mosquitto mosquitto-clients`
    *   **启动服务**: `sudo systemctl start mosquitqitto`
    *   **默认地址**: `localhost` 或 `127.0.0.1`, 端口 `1883`。
*   **Python 库**: 使用 `paho-mqtt` 库，这是最流行的 Python MQTT 客户端。
    *   **安装**: `pip install paho-mqtt`

### **2. 消息格式设计**

为了同时传输图片和文本标注，我们需要定义一个清晰的消息结构。这里推荐使用 **JSON** 格式，并将图片数据进行 **Base64 编码**（因为 JSON 不能直接包含二进制数据）。

```json
{
    "image_name": "print_001.jpg",
    "image_data": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAg...", // 这里是图片的Base64编码字符串
    "annotation": "No defect detected"
}
```

### **3. 发送端 (Publisher) 代码**

这个脚本负责读取本地图片，将其与标注信息打包并通过 MQTT 发送。

```python
# publisher.py
import base64
import json
import os
from paho.mqtt import client as mqtt_client

# --- 配置 ---
BROKER = 'localhost'  # MQTT Broker 地址
PORT = 1883           # MQTT Broker 端口
TOPIC = '3dprinter/monitoring'  # 主题
IMAGE_PATH = './input_image.jpg'  # 要发送的图片路径
ANNOTATION_TEXT = 'Layer 5: Potential warping detected.'  # 你的标注文本

def connect_mqtt():
    """连接到 MQTT Broker"""
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def publish_image_and_annotation(client):
    """读取图片、编码、打包并发布消息"""
    try:
        # 1. 读取图片为二进制
        with open(IMAGE_PATH, "rb") as image_file:
            image_bytes = image_file.read()

        # 2. 将二进制数据编码为 Base64 字符串
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # 3. 创建包含图片和标注的 JSON 消息
        payload = {
            "image_name": os.path.basename(IMAGE_PATH),
            "image_data": image_b64,
            "annotation": ANNOTATION_TEXT
        }
        message = json.dumps(payload)

        # 4. 发布消息
        result = client.publish(TOPIC, message)
        status = result[0]
        if status == 0:
            print(f"Message sent to topic `{TOPIC}`")
        else:
            print(f"Failed to send message to topic `{TOPIC}`")

    except FileNotFoundError:
        print(f"Error: Image file '{IMAGE_PATH}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    client = connect_mqtt()
    client.loop_start()  # 启动网络循环
    publish_image_and_annotation(client)
    client.loop_stop()   # 停止网络循环
```

### **4. 接收端 (Subscriber) 代码**

这个脚本订阅指定的主题，接收消息，并将图片和标注分别保存到指定的文件夹。

```python
# subscriber.py
import base64
import json
import os
from paho.mqtt import client as mqtt_client

# --- 配置 ---
BROKER = 'localhost'
PORT = 1883
TOPIC = '3dprinter/monitoring'
OUTPUT_IMAGE_DIR = './received_images/'  # 图片保存目录
OUTPUT_ANNOTATION_DIR = './annotations/' # 标注保存目录

# 确保输出目录存在
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATION_DIR, exist_ok=True)

def connect_mqtt():
    """连接到 MQTT Broker"""
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker! Subscribing...")
            client.subscribe(TOPIC)
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def save_message(msg):
    """解析接收到的消息并保存图片和标注"""
    try:
        # 1. 将接收到的字节消息解码为字符串，再解析为 JSON 对象
        payload = json.loads(msg.payload.decode('utf-8'))
        
        image_name = payload['image_name']
        image_b64 = payload['image_data']
        annotation = payload['annotation']

        # 2. 保存图片
        image_path = os.path.join(OUTPUT_IMAGE_DIR, image_name)
        image_bytes = base64.b64decode(image_b64)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        print(f"Image saved to: {image_path}")

        # 3. 保存标注 (可以保存为同名的 .txt 文件)
        annotation_path = os.path.join(OUTPUT_ANNOTATION_DIR, f"{os.path.splitext(image_name)[0]}.txt")
        with open(annotation_path, "w") as f:
            f.write(annotation)
        print(f"Annotation saved to: {annotation_path}")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing message: {e}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")

def on_message(client, userdata, msg):
    """当收到新消息时调用此函数"""
    print(f"Received message on topic `{msg.topic}`")
    save_message(msg)

if __name__ == '__main__':
    client = connect_mqtt()
    client.on_message = on_message  # 设置消息回调
    client.loop_forever()  # 保持连接并监听消息
```

### **5. 如何运行**

1.  **启动 Broker**: 确保 `mosquitto` 服务正在运行。
2.  **启动接收端**: 在终端运行 `python subscriber.py`。它会连接到 Broker 并开始监听 `3dprinter/monitoring` 主题。
3.  **启动发送端**: 在另一个终端运行 `python publisher.py`。它会发送一条包含图片和标注的消息。
4.  **检查结果**: 查看 `./received_images/` 和 `./annotations/` 文件夹，你应该能看到接收到的文件。

### **重要注意事项**

*   **性能**: Base64 编码会使数据体积增加约 33%。对于高分辨率或高帧率的视频流，这可能会成为瓶颈。如果性能是关键，可以考虑只发送图像差异或使用更高效的二进制协议（但这会增加复杂性）。
*   **安全性**: 本地开发时 `localhost` 是安全的。但在生产环境中，务必为你的 MQTT Broker 配置用户名/密码认证 (`client.username_pw_set(...)`) 并考虑使用 TLS 加密。
*   **错误处理**: 上述代码包含了基本的错误处理，但在实际项目中，你需要更健壮的机制来处理网络中断、消息丢失等情况。