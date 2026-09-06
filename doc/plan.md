# PrintWatch 项目 — OpenCode 完整开发计划（v3.2 定稿版）
**版本**：v3.2（基于 v3.1 检查报告合并全部 12 处修正）
**生成日期**：2026-09-06
**硬件状态**：GPU（RTX 2070 Super）与树莓派 4B 明日上线
**开发机**：Windows CPU 开发机（Python 3.12，torch 实际版本以 `uv run python -c "import torch; print(torch.__version__)"` 输出为准并回填此处：`___`）
**数据集**：Kaggle 3D Print Defect Dataset（187,770 样本 / 6 类）
**包管理**：uv（规范见附录 B）
**代码托管**：GitHub 私仓
---
## 变更摘要
原计划“一周 CPU 期 → 一周 GPU 迁移期 → 等待树莓派权限”的三段式节奏被打断——GPU 与树莓派**同时上线**。计划压缩为**双轨并行 7 天模式**：GPU 端（预训练 + RL 搜索）与边缘端（真机标定 + 兜底部署）同步推进，EdgeSimulator 从“联调替代品”降级为“回归测试工具”。
### v3.1 → v3.2 变更记录
| # | 变更 | 级别 |
|---|------|------|
| 1 | §6 全部命令补 `uv run` 前缀 | 阻塞 |
| 2 | Day 0 基准缩减为 GhostNet-only，新增 Day 2 追加任务 0.2b 补测三族 | 阻塞 |
| 3 | 新增 Day 0 步骤 0.0a（源端打包：manifest / index_v2 / class_stats） | 阻塞 |
| 4 | 统一 data.root / index_csv 三处配置路径，GPU yaml 补字段 | 矛盾 |
| 5 | 搜索空间维度统一为 ~30 维 | 矛盾 |
| 6 | 删除 §13 时态错误节（内容并入 §0 与 §14） | 矛盾 |
| 7 | 发布门控定义初始 best_f1 来源（baseline 模型 metrics_json） | 矛盾 |
| 8 | resnet_cbam 族 FLOPs 冲突预案（Day 2 实测决策） | 矛盾 |
| 9 | torch 版本号笔误修正（以实际输出为准） | 次要 |
| 10 | reward_rms 定义补充（RunningMeanStd 跨 session 恢复） | 次要 |
| 11 | NCNN 表述统一为“仅可行性调研，不实现” | 次要 |
| 12 | 契约测试真机参数依赖说明（conftest fixture 时序） | 次要 |
---
## 0. 当前状态（Baseline）
### 已完成工作（D1-D3 + D4 进行中）
| 项 | 状态 | 说明 |
|---|------|------|
| 仓库治理 | ✅ D1 完成 | `.python-version` 锁定 Python 3.12，旧代码归档至 `archive/` |
| 配置抽象 | ✅ D1 完成 | `configs/base.yaml` + `cpu_dev.yaml`，`src/common/config.py` 加载器 |
| DB 迁移 | ✅ D1 完成 | `scripts/init_db_v2.py` 幂等迁移脚本 |
| 契约测试 | ✅ D2 完成 | 15 用例全绿（auth / sync_data / get_latest_model），基于 simulator |
| EdgeSimulator | ✅ D2 完成 | 含分层采样、双缓冲、telemetry、baseline 保护；定位为回归测试工具 |
| 数据管线 | ✅ D3 完成 | `data/pretrain_index.csv`（187,770 样本 / 6 类 / 分层采样），dataset.py + transforms.py |
| 服务端 RLserver | ✅ 已联调 | sync_data / get_latest_model / 鉴权 / 版本管理 |
| Track A 训练器 | 🔄 D4 进行中 | trainer.py / losses.py / run_pretrain.py / builder-GhostNet |
| Track B RL 搜索 | ⬜ 未开始 | space / reward / latency_table / workers / gate |
| GPU 环境 | ⏳ 明日上线 | RTX 2070 Super |
| 树莓派 | ⏳ 明日上线 | Raspberry Pi 4B |
### 冻结承诺
1. **API 协议向后兼容扩展**（已联调行为不修改，只加字段/加端点）
2. **GhostNet 兜底永不被覆盖**（`/models/baseline/` 只读，任何发布流程不可触碰）
3. **所有训练代码走 `PW_ENV` 配置抽象**，禁止硬编码 device
4. **所有命令统一 `uv run` 前缀**（附录 B 规范，GPU 机与 CPU 机一致执行）
---
## 1. 架构总览
```
┌────────────── 云端（RTX 2070 Super）──────────────┐
│  RLserver(Flask+SQLite) ← retrain.py 扩展点        │
│  ├─ 触发门控 → RL搜索(DQN + 4 worker) → 发布门控    │
│  ├─ 预训练管线（Track A, 复用上轮数据/参数）          │
│  └─ 延迟查找表（真机标定数据）                        │
└──────────┬────────────────────┬─────────────────┘
     sync_data↑            get_latest_model↓
┌──────────┴────────────────────┴─────────────────┐
│  树莓派 4B：ONNX Runtime 双缓冲热更新               │
│  ├─ 本地常驻 GhostNet-100 兜底（三级回退链终点）       │
│  ├─ 真机延迟标定模式（Day 0 用途）                   │
│  └─ 数据采集（置信度分层上报策略）                    │
└─────────────────────────────────────────────────┘
```
**奖励函数（冻结）**：
```
R = 0.6·macro_F1 + 0.25·exp(-5·max(0, edge_lat/2 - 1)) + 0.1·(1-quant_drop) - 0.05·struct_dist
```
> 归一化：`reward_rms`（RunningMeanStd）在 `reward.py` 内维护，跨搜索 session 从 replay buffer 快照恢复，避免每周数据分布漂移导致奖励尺度漂移。
**发布门控（冻结）**：
- 累积测试集 F1 ≥ history.best_f1 + 1%
  - **初始 best_f1 定义**：首次发布时 history 为空，取 A3 预训练 GhostNet 基线在累积测试集上的 F1（从 models 表 `status='baseline'` 行的 `metrics_json` 读取）
- p95 边缘延迟 < 2.0s
- 量化掉点 < 2%
- 不触碰 baseline（`model_family='ghostnet' and status='baseline'` 的行受发布 API 保护）
---
## 2. 代码仓库结构
```
printwatch/
├── pyproject.toml                # uv 依赖真源（含 pytorch-cu121 index）
├── uv.lock                       # 锁定文件（进 Git）
├── .gitignore                    # 数据/模型/产物排除，CSV/manifest 例外进库
│
├── configs/
│   ├── base.yaml                 # 公共配置（data.root 占位，被子配置覆盖）
│   ├── cpu_dev.yaml              # CPU: smoke 参数、小分辨率、本机数据路径
│   └── gpu_train.yaml            # GPU: 完整参数、本机数据路径
│
├── src/RLserver/                 # ── 已有代码，只做增量修改 ──
│   ├── main.py
│   ├── api.py                    # [修改] 新增 /telemetry 路由
│   ├── database.py               # [修改] ground_truth 等字段
│   ├── models_store.py           # [修改] config_json/metrics_json 元数据 + baseline 保护
│   ├── retrain.py                # [重写] RL 搜索调度入口
│   └── ...
│
├── src/pretrain/                 # ── Track A: 预训练管线 ──
│   ├── dataset.py                # CSV 索引 Dataset + resolve_path() 路径兼容
│   ├── transforms.py
│   ├── trainer.py                # device 抽象、AMP 可选、rng checkpoint、早停
│   ├── losses.py                 # FocalLoss / KDLoss（含手工算例单测）
│   └── run_pretrain.py           # CLI 入口
│
├── src/search/                   # ── Track B: RL 搜索引擎 ──
│   ├── space.py                  # 四族搜索空间（~30 维扁平向量）
│   ├── builder.py                # action → 模型实例化（timm）
│   ├── latency_table.py          # 延迟查找表（真机标定数据驱动）
│   ├── simulator.py              # EdgeSimulator（回归测试工具）
│   ├── reward.py                 # 复合奖励 + RunningMeanStd
│   ├── gate.py                   # 发布门控 + 三级回退
│   ├── agents/dqn_agent.py       # Double DQN
│   ├── workers/eval_worker.py    # 候选评估子进程
│   └── run_search.py             # CLI 入口
│
├── tests/
│   ├── contract/test_api_v1.py   # 15 用例（simulator 版；真机 target 由 conftest 扩展）
│   ├── conftest.py               # [Day 5 前扩展] target fixture: simulator/real_pi
│   ├── test_smoke_pretrain.py
│   ├── test_smoke_search.py
│   └── test_gate_and_rollback.py
│
├── scripts/
│   ├── migrate_dataset.py        # manifest / verify / rebuild-csv / integrity / migrate
│   ├── seed_latency_table.py     # 真机标定数据 → 查找表
│   ├── init_db_v2.py
│   ├── gpu_migrate_check.py
│   ├── pi_benchmark.py           # 树莓派延迟基准
│   └── check_latency_margin.py
│
└── docs/
    ├── PLAN.md                   # 本文档
    └── MIGRATION.md              # 迁移检查清单
```
---
## 3. 环境与配置抽象（v3.2：路径统一）
所有环境差异收敛到配置文件 + 环境变量，代码零分支。**路径规范（本次修正核心）**：`base.yaml` 的 `data.root` 为占位，**每台机器的子配置各自覆盖为实际路径**；CSV 一律存相对路径。
### configs/base.yaml（公共）
```yaml
data:
  index_csv: data/pretrain_index_v2.csv   # 相对路径版索引（进 Git）
  num_classes: 6
```
### configs/cpu_dev.yaml
```yaml
device: cpu
amp: false
data:
  root: D:/kaggle_dataset/            # ← CPU 开发机实际路径（示例）
  img_size: 64
  num_workers: 2
train: { batch_size: 16, epochs: 2, smoke: true }
search: { smoke: true, n_candidates: 2, eval_workers: 1 }
```
### configs/gpu_train.yaml（v3.2：补齐缺失字段）
```yaml
device: cuda
amp: true
data:
  root: E:/kaggle_dataset/            # ← GPU 机实际路径（迁移校验通过后填写）
  img_size: 224
  num_workers: 4
train: { batch_size: 64, epochs: 50, smoke: false }
search: { smoke: false, n_candidates: 300, eval_workers: 4, cuda_mps: true }
```
### 统一加载器（src/common/config.py）
```python
import os, yaml
def load(env=None):
    env = env or os.getenv("PW_ENV", "cpu_dev")
    cfg = yaml.safe_load(open(f"configs/{env}.yaml"))
    base = yaml.safe_load(open("configs/base.yaml"))
    cfg = {**base, **cfg}                                # 子配置覆盖 base
    cfg["data"] = {**base["data"], **cfg.get("data", {})}
    cfg["device"] = os.getenv("PW_DEVICE", cfg["device"])
    return cfg
```
**验收标准**：`PW_ENV=gpu_train` 切换后，无需改任何一行 Python 代码即可在 GPU 上重跑相同命令。
---
## 4. 执行计划：双轨并行 7 天
### Day 0（明天）：硬件上线日
**目标**：两台硬件可用 + 基础设施标定完成。今天不做任何研究性工作。**严格按 0.0 → 0.0a → 0.0b → 0.1 → 0.2 → 0.3 顺序执行。**
| 序 | 任务 | 验收标准 | 预估 |
|---|------|---------|------|
| 0.0 | **GPU 机环境搭建**：clone 仓库 → `uv sync` → 依赖冒烟 import；核对 GPU 机 torch 版本与 uv.lock 一致 | `uv run python -c "import torch; print(torch.cuda.is_available())"` = True | 0.5h |
| 0.0a | **源端（CPU 开发机）打包**：`migrate_dataset.py manifest`（生成 data/manifest.json）+ `rebuild-csv`（生成 data/pretrain_index_v2.csv 相对路径版）+ class_stats 快照 → `git push` | GitHub 上可见 manifest.json / pretrain_index_v2.csv / class_stats.json 三文件 | 0.5h |
| 0.0b | **数据集迁移校验（GPU 机）**：拷贝数据（rsync --checksum / robocopy）→ `migrate_dataset.py migrate --verify all` | 三层校验全绿（SHA256 全部一致 / 结构统计一致 / 200 张抽样解码成功），报告归档；填写 gpu_train.yaml 的 data.root | 1~2h |
| 0.1 | **GPU 环境自检**：`scripts/gpu_migrate_check.py` | 全项 ✓（含 §14 清单全部检查项） | 1h |
| 0.2 | **树莓派真机延迟基准（缩减版）**：**GhostNet × 160/192/224 × fp32**，各 100 次取 p50/p95/p99 → `latency_benchmarks.json` | 数据落盘；GhostNet@224 p95 实测确认 | 1h |
| 0.3 | **GhostNet 兜底烧录**：树莓派 `/models/baseline/ghostnet_v1.onnx` + 只读保护；契约测试 **simulator 版全绿 + 真机 curl 抽验**（`--target=real_pi` fixture 依赖 Day 5 前的 conftest 扩展，Day 0 不阻塞） | 兜底就位；simulator 回归 15 用例全绿；真机双接口手动验证通过 | 1h |
| — | **0.2b（顺延至 Day 2）**：A4 + B2 完成后，补测 **LeViT/RegNet/ResNet+CBAM × 各分辨率 × fp32/int8**，增量合并进 latency_benchmarks.json | 查找表覆盖全搜索空间 | 1.5h |
**Day 0 关键产出**：`latency_benchmarks.json`（真机数据）。它取代“文献基准 + 插值估算”，奖励函数的速度项从此可信。
**关键判断**：若 GhostNet@224 真机 p95 > 1.6s（安全余量 < 20%），立刻下调搜索空间 `input_res` 上限为 192。
**风险预案**：
- 0.2 延迟超标 → 搜索空间收紧（分辨率降档 + prune_ratio 下限提高）+ 启动 **NCNN 可行性调研（仅评估，不实现，与 §16 删减决定一致）**
- 树莓派与云端不同网段 → 手机热点组局域网；标定数据本身与网络无关
---
### Day 1-2：Track A GPU 正式预训练（与 Track B 开发并行）
| 任务 | 内容 | 验收 |
|------|------|------|
| A1 | D4 收尾：`trainer.py`（AMP / rng checkpoint / 早停）+ `losses.py` 单测 + `run_pretrain.py` CLI | CPU smoke 通过后 `PW_ENV=gpu_train` 复跑通过 |
| A2 | **上轮 checkpoint 接入**：`--resume_from` 加载上轮实验权重，key 映射校验 | 加载后 val F1 与上轮报告一致（±0.5%） |
| A3 | **GhostNet-100 正式训练**：224 / 50 epoch / FocalLoss / 全量数据；**训练完成后注册 `status='baseline'` 并写入 metrics_json（含累积测试集 F1，作为门控初始基准）** | 第一个正式基线；训练曲线 CSV 归档；models 表 baseline 行就位 |
| A4 | builder 补全 LeViT / RegNetY-040 / ResNet-34+CBAM 三族（CBAM stage mask 注入 / width 缩放 / 剪枝挂点） | 4 族 × 10 随机 action 前向通过 |
| A5 | 三族冒烟训练（10 epoch） | 三条 loss 曲线正常下降 |
| A4.5 | **（Day 2 晚）执行 0.2b**：三族真机延迟补测，`seed_latency_table.py` 合并 | latency_table 覆盖全空间 |
**A5 若某族不收敛**：记录现象降低优先级，不阻塞主线；在 `space.py` 中缩小该族参数范围，让 RL 学会回避极端配置。
**resnet_cbam FLOPs 决策点（Day 2 实测）**：ResNet-34 基础约 3.6 GFLOPs，若压缩后全组合仍超 2.0G 硬闸门（该族被 RL 永久回避，搜索空间形同虚设），则二选一：
- **方案 i（推荐）**：对该族单独放宽 FLOPs 上限至 2.5G，延迟由查找表软惩罚约束
- **方案 ii**：该族 `input_res` 限至 160，保持 2.0G 全局约束
决策写入 `space.py` 注释与 MIGRATION.md。
---
### Day 2-4：Track B RL 搜索引擎
| 任务 | 内容 | 验收 |
|------|------|------|
| B1 | `space.py` + `reward.py` + `latency_table.py`（GhostNet 真机数据 + 三族**占位插值**，0.2b 后替换为实测值） | 奖励手工算例对齐；查表覆盖全部 action 组合 |
| B2 | `eval_worker.py`：训练→INT8 量化→ONNX 导出（先 `model.cpu().eval()`）→查表延迟→奖励全流程；FLOPs 硬约束前置拦截 | 单候选 GPU 耗时实测归档；INT8 导出工具链就位（0.2b 依赖此产出） |
| B3 | `run_search.py`：贝叶斯冷启动 25 候选 → Double DQN 主循环；spawn 模式 + 显存软限制 0.22；**replay buffer 落盘支持断点续搜** | 50 候选跑通，replay buffer 正常积累 |
| B4 | `gate.py` + 发布链路：过门控候选 → `publish_model()`（写 config_json/metrics_json）；初始 best_f1 从 baseline 行读取；baseline 保护规则单测 | 门控拒绝/通过两路径 + 空 history 首发布路径测试通过 |
**单候选训练耗时 → 搜索预算核算表**（Day 2 实测后填写）：
| 实测耗时 | 4 路并行 | 候选/小时 | 一夜(8h) | 应对策略 |
|---------|---------|----------|---------|---------|
| ___ min/候选 | ___ | ___ | ___ | 若一夜 < 60 候选：epoch 降至 30 或分辨率降至 192 |
---
### Day 5：首闭环日（里程碑）
```
当晚触发第一次完整自动搜索（200~300 候选，通宵）
   ↓ 次日早晨
top-1 候选过发布门控（初始 best_f1 = A3 GhostNet 基线）
   ↓
conftest.py 扩展 target fixture → 契约回归（simulator + 真机双跑）
   ↓
publish_model() → 树莓派 get_latest_model 真机拉取
   ↓
真机热更新 + 10 张校准图自检 + 延迟实测
   ↓ 成功: 记为 v_next, 遥测接通
   ↓ 失败: 三级回退链实战演练（这本身就是验收项）
```
**Day 5 晚即第一次“数据→搜索→发布→真机热更新”全自动闭环**。无论成功或回退，均产出系统韧性实测证据（论文素材）。
---
### Day 6-7：进化闭环运转 + 加固
| 任务 | 内容 |
|------|------|
| C1 | `retrain.py` 扩展点接入完整搜索流程 + 触发门控（已标注未用数据 ≥500 条或 7 天周期） |
| C2 | 标注最小闭环：`ground_truth` 字段 + CSV 导出工具（人工核对用，不做 UI） |
| C3 | 蒸馏接入：新候选训练以上一版部署模型为 teacher（KDLoss 已备好） |
| C4 | 坏模型实弹演练：故意 publish 伪造 ONNX → 双缓冲回退 → baseline 兜底 → 遥测记录 `load_result` |
| C5 | 监控告警：telemetry 入库 + 延迟/F1 异常告警日志 |
| C6 | 文档：MIGRATION.md 实测版定稿；搜索预算表定稿；实验记录归档；**tag `day5-closedloop`** |
---
## 5. 并行工作分配建议
| 工作流 | 内容 | 依赖 |
|--------|------|------|
| 🅰️ GPU 轨 | A1→A3→A5，随后支援 B2/B3 | Day 0 自检通过 |
| 🅱️ 搜索轨 | B1→B2→B3→B4（与 A 轨共享 builder） | Day 0 延迟数据 |
| 🅲 边缘轨 | 0.2→0.3→真机契约测试→Day 5 真机对接→C4 | 树莓派在网 |
**单人执行按天序串行**：Day 0 全部 → A1 → B1 → A3（GPU 后台）+ B2（前台）→ B3/B4 → Day 5 闭环。
---
## 6. Day 0 即刻行动清单（v3.2：全部统一 `uv run`）
```bash
# ── 源端（CPU 开发机）──
# 0.0a 源端打包
uv run python scripts/migrate_dataset.py manifest \
  --csv data/pretrain_index.csv --data-root D:/kaggle_dataset --out data/manifest.json
uv run python scripts/migrate_dataset.py rebuild-csv \
  --csv data/pretrain_index.csv --out data/pretrain_index_v2.csv
uv run python scripts/migrate_dataset.py class-stats \
  --csv data/pretrain_index_v2.csv --out data/class_stats.json
git add . && git commit -m "day0a: dataset manifest + relative-path index" && git push
# ── GPU 机 ──
# 0.0 环境搭建
git clone git@github.com:<you>/printwatch.git && cd printwatch
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
# 0.0b 数据迁移校验（数据先经 robocopy/rsync 拷至 E:/kaggle_dataset）
uv run python scripts/migrate_dataset.py migrate \
  --csv data/pretrain_index_v2.csv --data-root E:/kaggle_dataset \
  --manifest data/manifest.json --verify all
# 0.1 GPU 自检
PW_ENV=gpu_train uv run python scripts/gpu_migrate_check.py
# ── 树莓派侧 ──
# 0.2 缩减版基准（GhostNet only，三族待 0.2b）
uv run python scripts/pi_benchmark.py --families ghostnet \
  --res 160 192 224 --quant fp32 --runs 100 --out latency_benchmarks.json
# 0.3 兜底烧录
scp models/ghostnet_baseline.onnx pi@<ip>:/models/baseline/
# 回归（simulator 版 + 真机抽验）
uv run pytest tests/contract/ -q
uv run python scripts/check_latency_margin.py   # GhostNet@224 p95 必须 < 1.6s
```
> **注**：`--target=real_pi` 的 pytest fixture 由 conftest.py 扩展提供，安排在 Day 5 之前实现；Day 0 用 simulator 版全绿 + 上方真机 curl 抽验替代。
---
## 7. 搜索空间定义（space.py，冻结）
```python
SEARCH_SPACE = {
    "model_family":  {"type": "categorical", "choices": ["ghostnet", "levit", "regnet", "resnet_cbam"]},
    "input_res":     {"type": "categorical", "choices": [160, 192, 224]},
    "width_mult":    {"type": "discrete", "range": (0.35, 1.0), "steps": 8},
    "prune_ratio":   {"type": "discrete", "range": (0.0, 0.6),  "steps": 8},
    "levit_embed":   {"type": "categorical", "choices": [128, 192, 256]},
    "regnet_depth":  {"type": "categorical", "choices": [12, 16, 20, 24]},
    "cbam_stages":   {"type": "multi_binary", "n": 4},
    "quant":         {"type": "categorical", "choices": ["fp32", "int8_dynamic"]},
}
```
动作编码：全部离散化为 one-hot 拼接扁平向量（**约 30 维**，四族全空间；CPU 冒烟期对空间做子集采样），DQN 输入输出同维度；含 `validate_action()` 校验函数。
**resnet_cbam FLOPs 预案**：若 Day 2 实测该族全组合超 2.0G 硬闸门，按 §Day 1-2 决策点处理（放宽至 2.5G 或限 res 160），决策记录于本节注释。
---
## 8. 奖励函数（reward.py，公式冻结）
```python
def compute_reward(metrics: dict, cfg) -> float:
    alpha, beta, gamma, eta, lam = 0.6, 0.25, 0.10, 0.05, 5.0
    acc_term  = alpha * metrics["macro_f1"]
    lat_term  = beta * math.exp(-lam * max(0.0, metrics["edge_latency_s"]/2.0 - 1))
    stab_term = gamma * (1.0 - metrics["quant_drop"])
    reg_term  = -eta * metrics["struct_dist_to_deployed"]
    r = acc_term + lat_term + stab_term + reg_term
    return (r - reward_rms.mean) / (reward_rms.std + 1e-8)
# reward_rms: RunningMeanStd 实例，模块级维护；
# 跨搜索 session 从 replay buffer 快照（runs/*/rms.pkl）恢复。
```
**硬约束前置**：候选 FLOPs 超上限（默认 2.0G INT8 折算，resnet_cbam 按 §7 决策）**不进训练**，直接返回 -1.0 并记录日志。
---
## 9. 发布门控（gate.py）
```python
def passes_gate(candidate, history) -> bool:
    # history.best_f1 为空时（首次发布），从 models 表
    # status='baseline' 行的 metrics_json 读取初始基准（A3 产出）
    baseline_f1 = history.best_f1 if history.best_f1 is not None else load_baseline_f1()
    return (
        candidate.f1 >= baseline_f1 + 0.01
        and candidate.edge_latency_p95 < 2.0
        and candidate.quant_drop < 0.02
        and not is_baseline_ghostnet(candidate)   # 兜底永不被覆盖
    )
```
三级回退链：新模型加载失败 → 回退上一成功版本（双缓冲，已验证）→ 仍失败 → 强制加载 `/models/baseline/ghostnet_v1.onnx`（只读，不可被任何发布动作覆盖）。
---
## 10. 数据库 Schema 迁移（init_db_v2.py，幂等）
```sql
-- models 表
ALTER TABLE models ADD COLUMN config_json  TEXT;
ALTER TABLE models ADD COLUMN metrics_json TEXT;
ALTER TABLE models ADD COLUMN model_family TEXT;
ALTER TABLE models ADD COLUMN status       TEXT DEFAULT 'active';
-- records 表
ALTER TABLE records ADD COLUMN ground_truth      TEXT;               -- NULL=未标注
ALTER TABLE records ADD COLUMN annotation_status TEXT DEFAULT 'pending';
ALTER TABLE records ADD COLUMN sample_tier       TEXT;               -- uncertain/normal_high/defect_high
ALTER TABLE records ADD COLUMN latency_ms        REAL;
-- 遥测表
CREATE TABLE telemetry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL, model_version TEXT NOT NULL,
  latency_p50_ms REAL, latency_p95_ms REAL,
  inference_count_24h INTEGER, load_result TEXT,
  error_detail TEXT, memory_peak_mb REAL, created_at TEXT NOT NULL
);
```
---
## 11. 风险登记簿（v3.2）
| 风险 | 概率 | 影响 | 对策 |
|------|-----|------|------|
| 树莓派与服务器不同局域网 | 中 | Day 0 标定受阻 | 手机热点组网；标定数据与网络无关 |
| 上轮 checkpoint 与 timm 权重命名不匹配 | 高 | A2 延误半天 | Day 0 晚提前跑 key 对比脚本，写映射表 |
| 某族（如 LeViT）量化掉点 > 2% | 中 | 该族奖励被压制 | 属预期，RL 学会回避；该族强制 QAT |
| GPU 单候选训练耗时 > 25min | 中 | 搜索预算缩水 | Day 2 实测后按预算表降 epoch/分辨率 |
| Windows spawn 多进程死锁 | 低 | 通宵搜索中断 | worker 超时守护 + replay buffer 落盘断点续搜 |
| 首闭环门控全拒 | 中 | Day 5 无发布 | **不算失败**：门控有效，分析 top-3 归因（延迟 or F1），调 α/β 后二轮 |
| 数据拷贝中断/坏块 | 中 | 迁移失败 | rsync --checksum / robocopy 重试 + manifest 校验兜底，按坏文件清单增量重拷 |
| uv.lock 解析 CUDA index 失败 | 低 | 依赖安装失败 | 回退 `uv pip install torch --index-url .../cu121` 后 `uv sync --inexact` |
| resnet_cbam 全组合被 FLOPs 闸门拦截 | 中 | 该族搜索空间形同虚设 | Day 2 决策点：放宽至 2.5G 或限 res 160（见 §7） |
---
## 12. 与旧版差异对照
| 项 | v2.0 | v3.2 |
|---|------|------|
| 时间线 | CPU 周 → GPU 周 → 等树莓派 | **双轨并行 7 天** |
| 延迟查找表初值 | 文献基准 + 插值 | **Day 0 GhostNet 真机标定 + Day 2 三族补测（0.2b）** |
| EdgeSimulator 定位 | 联调替代品（核心） | 回归测试工具（真机接管联调） |
| 预训练 | 只做 CPU 冒烟 | **Day 1-2 GPU 正式训练出基线** |
| 契约测试 | 只对 simulator | simulator 回归 + Day 0 真机 curl 抽验 + Day 5 双跑 |
| 首次端到端闭环 | 原 Day 12 之后 | **Day 5 夜** |
| 数据源 | data/inbox/ | Kaggle 数据集（相对路径索引 + manifest 校验迁移） |
| 动作维度 | ~30 维 | **~30 维（四族全空间），CPU 冒烟期子集采样** |
| 命令规范 | 混用 python | **统一 `uv run` 前缀** |
| 门控基准 | 未定义首次发布基准 | **初始 best_f1 = baseline 模型 metrics_json** |
---
## 13. GPU 迁移检查清单（MIGRATION.md 骨架）
```bash
# scripts/gpu_migrate_check.py 输出示例
[✓] torch.cuda.is_available() = True (RTX 2070 Super, 8GB)
[✓] torch 版本与 uv.lock 一致
[✓] AMP autocast 前向/反向正常
[✓] 4 进程 × memory_fraction(0.22) 并发训练不 OOM
[✓] DataLoader num_workers=4 在 Windows spawn 模式正常
[✓] CUDA MPS 启用成功（可选）
[✓] timm 四族模型 .to(cuda) 前向 OK
[✓] ONNX 导出在 GPU 权重下正常（导出前强制 .cpu()）
```
**已知迁移风险点**：
1. **Windows multiprocessing**：spawn 模式，worker 必须可 pickle（顶层函数），入口 `if __name__ == "__main__"` 保护——CPU 期即按 spawn 开发，不留 Linux fork 依赖。
2. **ONNX 导出**：统一先 `model.cpu().eval()` 再导出。
3. **AMP**：仅训练加速，**评估指标强制 FP32**，避免 macro-F1 数值不稳定。
4. **数据迁移**：三层校验通过前，gpu_train.yaml 的 data.root 不允许投入使用。
---
## 14. 给 OpenCode 的执行指令模板
每个 session 按以下格式下发（示例为 Day 0 优先任务）：
> 参照 docs/PLAN.md：实现 `scripts/pi_benchmark.py`、`scripts/check_latency_margin.py`、`scripts/gpu_migrate_check.py` 与 `scripts/migrate_dataset.py`（manifest / rebuild-csv / class-stats / migrate --verify all 四个子命令）。
> 要求：
> 1. `pi_benchmark.py` 支持 `--families ghostnet`（0.2 缩减版；三族参数预留，0.2b 启用），输出 p50/p95/p99 至 latency_benchmarks.json
> 2. `migrate_dataset.py` 三层校验任一失败即非零退出并定位第一个坏文件
> 3. 全部代码无硬编码 device/路径，遵循 `src/common/config.py` 抽象
> 4. 所有脚本可通过 `uv run` 执行，兼容 Windows spawn
> 5. `uv run pytest tests/ -q` 全绿，不破坏 tests/contract/ 现有 15 用例
---
## 15. 本方案相对上一版的删减说明
- **删**：标注工作台 UI（`ground_truth` 字段 + CSV 导出临时替代）
- **删**：遥测反哺查找表的完整闭环（保留 telemetry 表与 `/telemetry` 路由占位，Day 5 遥测接通后启用）
- **删**：NCNN 实现分支（真机延迟基准出来前不动；仅保留“延迟超标时启动可行性调研”预案）
- **留**：sample_tier 分层上报逻辑在 EdgeSimulator 中实现，真机恢复后边缘端照抄
- **增**：Track A 与 builder 复用设计（A/B 共用结构生成器，避免搜索空间与预训练配置漂移）
---
## 附录 A：每日任务速查表
| 天 | 轨道 | 核心任务 | 里程碑 |
|----|------|---------|--------|
| **Day 0** | 基础设施 | 0.0 环境搭建 / 0.0a 源端打包 / 0.0b 数据校验 / 0.1 GPU 自检 / 0.2 GhostNet 延迟基准 / 0.3 兜底烧录 | `latency_benchmarks.json`（GhostNet 部分）产出 |
| **Day 1** | A 轨 | trainer 收尾 + checkpoint 接入 + GhostNet-100 正式训练（注册 baseline） | GPU 预训练跑通 |
| **Day 1** | B 轨 | space / reward / latency_table（真机数据 + 三族占位） | 奖励算例对齐 |
| **Day 2** | A 轨 | builder 补全三族 + 冒烟训练 + **0.2b 三族真机补测** | 4 族前向通过；查找表全覆盖 |
| **Day 2** | B 轨 | eval_worker + run_search（50 候选）+ **resnet_cbam FLOPs 决策** | 搜索跑通；预算表定稿 |
| **Day 3** | B 轨 | gate + 发布链路（含空 history 首发布路径） | 门控三路径测试通过 |
| **Day 4** | 合并 | 全链路联调 + 300 候选搜索准备 + conftest target fixture | 搜索就绪 |
| **Day 5** | 全轨 | 首次完整闭环（搜索→门控→发布→真机热更新） | **里程碑：首次闭环** |
| **Day 6** | C 轨 | 进化闭环 + 蒸馏 + 坏模型演练 | 三级回退真机验证 |
| **Day 7** | C 轨 | 监控告警 + 文档归档 + tag `day7-v1` | 实验记录归档 |
---
## 附录 B：文件传输与包管理规范
### 包管理：uv
- 唯一真源 = `pyproject.toml` + `uv.lock`（均进 Git），任何机器 `uv sync` 一键还原
- torch CUDA 版通过 `[tool.uv.index]` pytorch-cu121 指定，**禁止手工 pip 安装 torch**
- **所有运行命令统一 `uv run` 前缀**（含 pytest、scripts、模块入口）
- torch 实际版本回填：CPU 机 `___` / GPU 机 `___`（由 §6 冒烟命令输出）
### 代码传输：GitHub 私仓
- 数据/模型/训练产物不进 Git（.gitignore）；**例外进库**：`data/pretrain_index_v2.csv`（相对路径索引）、`data/manifest.json`（SHA256 清单）、`data/class_stats.json`（结构快照）
- 分支策略（单人）：main 直接开发 + 里程碑打 tag（`d4-smoke` / `day0-hardware` / `day5-closedloop` / `day7-v1`）
- 每完成一个 Day 强制 push；GPU 机开工前必先 `git pull`
### 数据集迁移三层校验（migrate_dataset.py）
1. **manifest 校验**：源端生成 SHA256 清单 → 目标端逐文件核对（缺一即失败并定位）
2. **结构校验**：行数 187,770 / split 比例 / 6 类分布与 class_stats.json 快照一致
3. **可用性校验**：随机 200 张 PIL 解码 + sha256 去重检查
4. **路径规范**：CSV 自 v2 起存相对路径，`dataset.py` 用 `cfg.data.root` 拼接；旧绝对路径由 `resolve_path()` 自动兼容并告警
---
**文档定稿声明**：v3.2 已合并检查报告全部 12 处修正（§变更记录），自本版起作为 OpenCode 执行的唯一真源文档；后续变更以增量修订（v3.3+）追加，禁止直接改写冻结承诺（§0）与冻结公式（§1、§7、§8、§9）。
