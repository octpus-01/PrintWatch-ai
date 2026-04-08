import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import torchvision.transforms as T
from torchvision import datasets, models

# ======================
# 超参数
# ======================
BATCH_SIZE = 32
LR = 1e-4          # 微调用小学习率
GAMMA = 0.95       # 奖励折扣
MEMORY_SIZE = 20000
EPS_START = 0.3    # 探索率低（因为已有预训练模型）
EPS_END = 0.01
EPS_DECAY = 5000
TARGET_UPDATE = 5  # 目标网络更新间隔
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 1. 图像分类主干网络（你自己的模型替换这里）
# ======================
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 这里用简单CNN演示，你可以换成 ResNet50 / ViT / 你训好的模型
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)  # 适配 CIFAR10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return x

# ======================
# 2. DQN 强化学习主体（包装分类模型）
# ======================
class DQN(nn.Module):
    def __init__(self, classifier, num_classes):
        super().__init__()
        self.backbone = classifier  # 你的预训练分类模型
        self.fc_out = nn.Linear(num_classes, num_classes)  # 输出Q值

    def forward(self, x):
        feat = self.backbone(x)
        q_value = self.fc_out(feat)  # 每个类别对应一个Q值
        return q_value

# ======================
# 3. 经验回放池
# ======================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.cat(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.cat(next_state),
            torch.tensor(done)
        )

    def __len__(self):
        return len(self.buffer)

# ======================
# 4. 强化学习智能体
# ======================
class RLAgent:
    def __init__(self, num_classes, pretrained_model=None):
        self.n_actions = num_classes
        
        # 加载你的分类模型（关键！）
        self.classifier = ImageClassifier(num_classes=num_classes).to(DEVICE)
        if pretrained_model is not None:
            self.classifier.load_state_dict(pretrained_model)  # 载入已训好权重

        # 构建 DQN 与 目标网络
        self.policy_net = DQN(self.classifier, num_classes).to(DEVICE)
        self.target_net = DQN(self.classifier, num_classes).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(MEMORY_SIZE)
        self.step_count = 0

    def select_action(self, state, explore=True):
        # 输入：图像tensor
        # 输出：分类动作（类别）
        with torch.no_grad():
            q_values = self.policy_net(state)
            
        # 强化学习经典 epsilon-greedy
        if explore:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-self.step_count / EPS_DECAY)
            if random.random() < eps:
                return random.randint(0, self.n_actions-1)
        return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # 采样批次数据
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = states.to(DEVICE)
        next_states = next_states.to(DEVICE)
        actions = actions.to(DEVICE)
        rewards = rewards.to(DEVICE)
        dones = dones.to(DEVICE)

        # 计算 Q 当前值
        q_pred = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            q_target = rewards + GAMMA * next_q * (~dones)

        # MSE 损失
        loss = F.mse_loss(q_pred.squeeze(), q_target)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ======================
# 5. 图像分类环境（奖励机制）
# ======================
class ClassifyEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(dataset)

    def step(self, action, true_label):
        """
        强化学习核心：奖励函数
        分类正确：+1 奖励
        分类错误：-1 惩罚
        """
        reward = 1.0 if action == true_label else -1.0
        done = True
        return reward, done

    def get_sample(self, idx):
        img, label = self.dataset[idx]
        img = img.unsqueeze(0)  # 增加batch维度
        return img, label

# ======================
# 6. 训练入口（可直接运行）
# ======================
if __name__ == "__main__":
    # 1. 加载数据集（CIFAR10 演示）
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    num_classes = 10

    # 2. 初始化环境与智能体
    env = ClassifyEnv(dataset)
    agent = RLAgent(num_classes=num_classes, pretrained_model=None)

    # 3. 强化学习训练
    EPISODES = 5000
    print("开始强化学习微调图像分类模型...")
    for episode in range(EPISODES):
        idx = np.random.randint(env.size)
        state, true_label = env.get_sample(idx)
        action = agent.select_action(state.to(DEVICE))
        reward, done = env.step(action, true_label)

        next_state, _ = env.get_sample(np.random.randint(env.size))
        agent.buffer.push(state, action, reward, next_state, done)
        agent.update()

        if episode % 100 == 0:
            print(f"Episode {episode:4d} | Reward: {reward:2.0f}")

    # 4. 保存强化学习优化后的模型
    torch.save(agent.classifier.state_dict(), "rl_finetuned_classifier.pth")
    print("强化学习完成，模型已保存！")