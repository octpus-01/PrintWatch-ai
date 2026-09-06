# PrintWatchAI 边缘端 — 服务端对接指南 (API_INTEGRATION_GUIDE)

## 1. 概述

本项目构建「边缘端推理 → 数据上报 → 服务端重训 → 模型下发」的闭环系统。边缘端（树莓派）通过 HTTP 与服务端进行数据同步与模型更新，实现端云协同的主动学习。

```
┌──────────────┐  ① POST /api/v1/sync_data（高置信度推理数据）   ┌──────────────┐
│  边缘端(树莓派) │ ────────────────────────────────────────────▶ │    服务端     │
│  推理 + 数据采集 │                                                │ 存储/重训/发布  │
│              │  ② GET /api/v1/get_latest_model（拉取新模型）     │              │
│              │ ◀────────────────────────────────────────────── │              │
└──────────────┘                                                 └──────────────┘
```

## 2. 全局规范

| 项目 | 说明 |
| :--- | :--- |
| 协议 | HTTP/1.1 或 HTTPS |
| 数据格式 | JSON（`get_latest_model` 除外，返回模型二进制） |
| 字符编码 | UTF-8 |
| 鉴权方式 | 所有请求 Header 携带 `Authorization: Bearer <token>` |
| 设备标识 | 由请求体 `device_id` 字段标识，服务端据此区分设备 |

鉴权失败返回 `401`，需边缘端重新配置 Token。

## 3. 接口详情

### 3.1 批量上报推理数据

**POST `/api/v1/sync_data`**

边缘端将筛选后的高置信度推理数据（图片 + 标签 + 置信度）上报，用于模型重训。

**请求体：**

```json
{
  "device_id": "pi-camera-001",
  "records": [
    {
      "timestamp": "2026-08-17T13:48:00.123456+00:00",
      "label": "cat",
      "confidence": 0.85,
      "image_base64": "/9j/4AAQSkZJRgABAQEA...",
      "model_path": "models/server_v2.onnx"
    }
  ]
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `device_id` | string | 是 | 设备唯一标识 |
| `records[].timestamp` | string | 是 | ISO-8601 时间戳（UTC） |
| `records[].label` | string | 是 | 预测标签（CIFAR-10 类别名） |
| `records[].confidence` | number | 是 | 最大置信度（0~1），边缘端仅上报 ≥ 0.5 的数据 |
| `records[].image_base64` | string | 是 | 原始帧 JPEG 的 Base64 编码 |
| `records[].model_path` | string | 否 | 产生该预测的模型路径/版本 |

**响应（200 OK）：**

```json
{
  "code": 200,
  "message": "success",
  "synced_count": 1
}
```

**批量上限：** 单次请求建议不超过 50 条（约 3MB）。

### 3.2 获取最新模型

**GET `/api/v1/get_latest_model`**

边缘端定时调用，获取服务端训练完成的新模型。

**查询参数：**

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `current_version` | string | 否 | 边缘端当前模型版本号，用于服务端判断是否有更新 |

**响应：**

- **有新版模型** → `200 OK`，`Content-Type: application/octet-stream`，Body 为模型二进制（`.onnx`）。响应 Header 必须携带：

  ```
  X-Model-Version: v2
  ```

  边缘端会将该模型保存为 `models/server_<version>.onnx` 并自动热更新，同时清理本地 7 天前的旧数据。

- **无新模型** → `304 Not Modified`（无 Body）。

## 4. 错误码字典

| 状态码 | 描述 | 边缘端处理建议 |
| :--- | :--- | :--- |
| 200 | 请求成功 | 正常处理 |
| 304 | 无新模型 | 忽略，继续使用当前模型 |
| 400 | 请求参数错误（如 Base64 格式错误） | 检查本地数据序列化逻辑 |
| 401 / 403 | 鉴权失败 | 检查 Token 是否配置/过期 |
| 404 | 接口路径不存在 | 检查服务端路由 |
| 408 / 429 | 超时 / 请求过于频繁 | 指数退避重试 |
| 500 / 502 / 503 / 504 | 服务端内部错误 | 触发重试机制，稍后重试 |

## 5. 交互时序

1. 边缘端每 5 秒推理一次，每 4 次（约 20 秒）筛选置信度 ≥ 0.5 的帧，存入本地 SQLite。
2. 边缘端每 10 分钟（`SYNC_INTERVAL`）将未同步数据批量 `POST /api/v1/sync_data`；成功后标记为已同步。
3. 边缘端每 30 分钟（`MODEL_CHECK_INTERVAL`）`GET /api/v1/get_latest_model`；有新版则自动热更新。
4. 新模型加载成功后，边缘端删除本地 7 天前的历史数据，为新模型重训保留近一周数据。
