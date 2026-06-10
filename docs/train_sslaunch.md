# Pi0.5 训练指南（本环境）

本文档适用于在当前集群环境下，使用 openpi 框架对 pi0.5 进行 fine-tuning，涵盖数据格式要求、训练前准备、单卡调试和 sslaunch 多卡提交。

## 1. 训练数据格式要求

openpi 训练使用 **LeRobot v2.1** 格式的本地数据集。数据集路径默认为 `~/.cache/huggingface/lerobot/<org>/<dataset_name>/`。

### 目录结构

```
~/.cache/huggingface/lerobot/<org>/<dataset_name>/
├── meta/
│   ├── info.json          # 数据集元信息（特征定义、episode 数量、fps 等）
│   ├── episodes.jsonl     # 每个 episode 的元数据（index, task, length）
│   ├── tasks.jsonl        # 任务列表（task_index → 自然语言描述）
│   └── stats.json         # 特征统计（可选，openpi 用自己的 norm_stats）
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...            # 每个 episode 一个 parquet 文件
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_left_wrist/
        └── observation.images.cam_right_wrist/
```

### info.json 关键字段

```json
{
  "codebase_version": "v2.1",
  "robot_type": "aloha",
  "total_episodes": 1000,
  "total_frames": 679710,
  "fps": 30,
  "splits": { "train": "0:1000" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.state": { "dtype": "float32", "shape": [14] },
    "action": { "dtype": "float32", "shape": [14] },
    "observation.images.cam_high": { "dtype": "video", "shape": [3, 480, 640] },
    "observation.images.cam_left_wrist": { "dtype": "video", "shape": [3, 480, 640] },
    "observation.images.cam_right_wrist": { "dtype": "video", "shape": [3, 480, 640] }
  }
}
```

> 480 * 640 这个分辨率是不是有点高呢?

### 特征命名约定

openpi 的 ALOHA 数据管线（`LeRobotAlohaDataConfig`）期望以下 key：

| 特征 | 说明 |
|------|------|
| `observation.state` | 本体感知状态（float32），双臂: `[left_j1..j6, left_gripper, right_j1..j6, right_gripper]`，共 14 维 |
| `action` | 动作（float32），与 state 同维同语义 |
| `observation.images.cam_high` | 头部/高位相机 |
| `observation.images.cam_left_wrist` | 左腕相机 |
| `observation.images.cam_right_wrist` | 右腕相机 |

这些 key 通过 TrainConfig 中的 `repack_transforms` 映射到模型内部表示。如果你的数据集 key 不同，需在 config 中调整 `RepackTransform`。

### 关于 gripper 值域

pi0.5 base model 预训练时 gripper 使用 `[0.0, 1.0]` 范围（0=全开, 1=全闭）。如果你的数据集 gripper 已在此范围内，可设 `adapt_to_pi=False` 跳过 ALOHA→PI 的 gripper 线性变换。

## 2. 定义训练 Config

在 `src/openpi/training/config.py` 中添加 `TrainConfig`，注册到 `_CONFIGS` 列表。示例（X-One 赠送数据）：

```python
TrainConfig(
    name="pi05_xone",                          # 唯一名称，CLI 通过此名引用
    model=pi0_config.Pi0Config(pi05=True),     # pi0.5 模型
    data=LeRobotAlohaDataConfig(
        repo_id="xone/xone26",                 # 对应 ~/.cache/huggingface/lerobot/xone/xone26/
        adapt_to_pi=False,                      # 是否应用 ALOHA→PI gripper 变换
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",                 # 复用 base 预训练的归一化统计
        ),
        base_config=DataConfig(prompt_from_task=True),  # 从 tasks.jsonl 读 prompt
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"  # 加载 base 预训练权重
    ),
    num_train_steps=20_000,
    batch_size=64,           # 8 卡时全局 batch size
)
```

### 常用 TrainConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_steps` | 30000 | 训练总步数 |
| `batch_size` | 32 | 全局 batch size（跨所有 GPU） |
| `save_interval` | 1000 | 每 N 步保存 checkpoint |
| `keep_period` | 5000 | `step % keep_period == 0` 的 checkpoint 不会被清理 |
| `overwrite` | False | 是否覆盖已有 checkpoint 目录 |
| `resume` | False | 是否从上次 checkpoint 恢复训练 |
| `wandb_enabled` | True | 是否启用 W&B 日志 |
| `fsdp_devices` | 1 | FSDP 分片设备数（>1 时启用 FSDP，降低单卡显存但可能稍慢） |
| `checkpoint_base_dir` | `./checkpoints` | checkpoint 基目录 |
| `exp_name` | (必填) | 实验名，CLI 传入，checkpoint 存入 `<base_dir>/<config_name>/<exp_name>/` |

大多数时候改一下 repo_id 就行了。

## 3. 训练前准备

### 3.1 一键准备脚本

如果数据集和资源尚未就绪，运行准备脚本：

```bash
cd /mnt/vepfs/base2/rongyinze/codebases/openpi
bash scripts/prepare_xone.sh
```

此脚本会自动完成：
1. 安装 Python 依赖（`uv sync`，需代理）
2. 下载 pi05_base checkpoint (~12GB) 和 norm stats（需代理访问 GCS）
3. 下载 PaliGemma tokenizer（需代理访问 GCS）
4. 从 BOS 同步 LeRobot 数据集（内网，无需代理）
5. 验证所有资源就绪

### 3.2 手动准备

如果不使用一键脚本，需确保以下资源存在：

| 资源 | 路径 | 来源 |
|------|------|------|
| Python 依赖 | openpi/.venv/ | `uv sync` |
| pi05_base checkpoint | `~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params/` | GCS 自动下载 |
| Norm stats | `~/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets/trossen/` | GCS 自动下载 |
| PaliGemma tokenizer | `~/.cache/openpi/big_vision/paligemma_tokenizer.model` | GCS 自动下载 |
| LeRobot 数据集 | `~/.cache/huggingface/lerobot/<org>/<name>/` | 自行转换或同步 |

GCS 资源首次使用时 openpi 会自动下载，但需要代理：

```bash
export https_proxy=192.168.48.27:18000
```

如果自动下载太慢，可使用 gsutil（需先 `uv pip install gsutil`）手动下载：

```bash
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base/params ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

### 3.3 计算归一化统计

如果不复用 base model 的 norm stats（即 config 中未设置 `AssetsConfig`），需先计算：

```bash
uv run scripts/compute_norm_stats.py --config-name <your_config_name>
```

如果复用 base 的 norm stats（推荐，当机器人与 ALOHA 动作空间兼容时），此步可跳过。

## 4. 单卡本地调试

在本机（有 GPU）快速验证 config 是否正确、数据管线是否通畅：

```bash
cd /mnt/vepfs/base2/rongyinze/codebases/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_xone \
  --exp-name=debug \
  --overwrite \
  --num-train-steps 5 \
  --batch-size 2 \
  --no-wandb-enabled
```

关键参数说明：
- `pi05_xone`：config 名称（positional arg）
- `--exp-name=debug`：实验名，checkpoint 存到 `checkpoints/pi05_xone/debug/`
- `--num-train-steps 5`：只跑 5 步验证
- `--batch-size 2`：小 batch 防止 OOM
- `--no-wandb-enabled`：关闭 W&B（tyro CLI 的布尔参数否定形式）
- `--overwrite`：覆盖已有同名 checkpoint 目录
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：允许 JAX 使用 90% GPU 显存

预期输出：
- Step 0 loss 约 0.1185（pi0.5 初始 loss）
- 5 步内 loss 略有下降
- 无报错即说明数据、模型、管线均正常

## 5. sslaunch 多卡集群训练

### 5.1 提交训练任务

项目提供了 `scripts/train_xone_cluster.sh`，封装了资源校验、环境变量设置和训练启动：

```bash
cd /mnt/vepfs/base2/rongyinze/codebases/openpi

# 用法: bash scripts/train_xone_cluster.sh <config_name> <exp_name>
# exp_name 会自动追加时间戳，如 xone26_sft → xone26_sft_20260430_082511

http_proxy= https_proxy= no_proxy= sslaunch submit \
  -c b200 \
  -q embody1 \
  -j rongyinze-pi05-xone-sft \
  -e BOS_UPLOAD_DIR=bos://base2-test/rongyinze/codebases/openpi/checkpoints \
  -e UPLOAD_KEEP_PERIOD=5000 \
  --no-log \
  -- \
  scripts/train_xone_cluster.sh pi05_xone xone26_sft
```

`train_xone_cluster.sh` 会自动完成：
1. 检查所有预下载资源（checkpoint、norm stats、tokenizer、数据集），缺失则 fail fast
2. 设置 `HF_HUB_OFFLINE=1` 阻止训练中意外的网络访问
3. 设置 NCCL 环境变量（`NCCL_P2P_DISABLE=1` 等）
4. 使用 `--fsdp-devices 8` 启用 FSDP 跨 8 卡分片
5. 训练日志同时输出到 `logs/<config_name>/<exp_name>.log`
6. 透传 `BOS_UPLOAD_DIR` / `UPLOAD_KEEP_PERIOD` 给 `train.py`，启用 checkpoint 自动上传 BOS（见 [6.4](#64-自动上传-bos--本地只保留-1-个-checkpoint)）；若设了 `BOS_UPLOAD_DIR` 但 megfile CLI 不存在则 fail fast

> `-e KEY=VALUE` 可多次使用，把环境变量注入 pod；上面两个变量是开启 BOS 自动上传的开关。`--no-log` 让 submit 创建任务后立即返回（训练在 k8s 独立运行），省得 submit 客户端一直挂着流式日志。

如需训练 `adapt_to_pi=True` 的对比实验：

```bash
sslaunch submit \
  -c b200 \
  -q a6000 \
  -j rongyinze-pi05-xone-sft \
  -- \
  scripts/train_xone_cluster.sh pi05_xone_adapt xone26_sft_adapt
```

### 5.2 多卡注意事项

- openpi 训练脚本自动检测所有可用 GPU，使用 JAX 的 data parallelism
- `batch_size` 是**全局** batch size，会自动分配到各卡（例如 batch_size=64，8 卡时每卡 8）
- 如果单卡显存不够，设 `--fsdp-devices N`（N>1）启用 FSDP 模型分片

## 6. 训练产出

### 6.1 Checkpoint 目录结构

checkpoint 存储路径为 `<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/`：

```
checkpoints/
└── pi05_xone/                              # config name
    └── pick_place_sft_20260607_173228/     # exp name
        ├── config.json                     # ← 本次训练的完整 TrainConfig（train.py 自动写，见 6.5）
        └── 19999/                          # step number (0-indexed)
            ├── params/                     # 模型权重 (~12GB)
            ├── train_state/                # 优化器状态（恢复训练用）
            ├── assets/                     # 归一化统计等资源
            │   └── trossen/
            │       └── norm_stats.json
            └── _CHECKPOINT_METADATA        # orbax checkpoint 元数据
```

`config.json` 在 **exp 目录层**（每次训练一份），与各 `<step>/` 平级——它描述整个 run，不随 step 变化。BOS 上的目录结构与本地一致（`config.json` 也会上传到 `<BOS_UPLOAD_DIR>/<config_name>/<exp_name>/config.json`）。

### 6.2 Checkpoint 保留策略

- 每 `save_interval`（默认 1000）步保存一次
- 旧 checkpoint 会被自动删除，**除非** `step % keep_period == 0`（默认 5000）
- 最终步（`num_train_steps - 1`）始终保留
- 使用 `--overwrite` 可清除已有同名实验的所有 checkpoint

> 底层是 orbax `CheckpointManager`，初始化时固定 `max_to_keep=1`，并把 config 的 `keep_period` 透传进去（见 `src/openpi/training/checkpoints.py`）。
> 因此：`keep_period=5000` 时本地会累积 step 0/5000/10000/… 多个 checkpoint；若设 `keep_period=None`，则本地任何时候只保留**最新 1 个** checkpoint（旧的在下一次 save 时被删）。后者配合 6.4 的 BOS 自动上传，可做到"本地只占 1 份磁盘、需要长期保留的 checkpoint 都在 BOS"。

### 6.3 W&B 监控

训练默认上报到 W&B 项目 `openpi`（可通过 `--project-name` 修改）。W&B run ID 保存在 checkpoint 目录下的 `wandb_id.txt`，`--resume` 时自动恢复同一 run。
不过 `train_xone_cluster.sh` 默认了 `--no-wandb-enabled`。

### 6.4 自动上传 BOS & 本地只保留 1 个 checkpoint

集群本地磁盘有限，而一个 pi0.5 checkpoint（params + train_state + assets）约 **42GB**。为此 `train.py` 支持在训练过程中把指定 checkpoint 自动上传到 BOS，并配合 `keep_period=None` 让本地始终只占 1 份。

#### 开关：两个环境变量

| 环境变量 | 含义 | 默认 |
|---|---|---|
| `BOS_UPLOAD_DIR` | BOS 基础路径前缀；**留空则完全不上传** | `""`（关闭） |
| `UPLOAD_KEEP_PERIOD` | 上传周期：`step % UPLOAD_KEEP_PERIOD == 0` 的 step 会上传 | `5000` |

实际上传目标为 `<BOS_UPLOAD_DIR>/<config_name>/<exp_name>/<step>/`，例如：

```
bos://base2-test/rongyinze/codebases/openpi/checkpoints/pi05_xone/pick_place_sft_20260607_173228/5000/
```

#### 上传哪些 step

`step % UPLOAD_KEEP_PERIOD == 0` 的 step **以及最终 step**（`num_train_steps - 1`）。
以 `num_train_steps=20000`、`UPLOAD_KEEP_PERIOD=5000` 为例，BOS 上最终会有 4 个 checkpoint：`5000/ 10000/ 15000/ 19999/`。

#### 配合 `keep_period=None` 实现本地只留 1 份

在 TrainConfig 里设 `keep_period=None`（本环境的 `pi05_xone` 已这样配置），orbax `max_to_keep=1` 就会让本地任何时刻只保留最新 1 个 checkpoint。上传时序（save_interval=1000）：

| step | 本地 save | 是否上传 BOS | 本地保留 | BOS 累计 |
|---|---|---|---|---|
| 5000 | ✓ | ✓（先 `wait_until_finished` 再 sync） | {5000} | {5000} |
| 6000 | ✓ | — | {6000}（orbax 删 5000） | {5000} |
| 10000 | ✓ | ✓ | {10000} | {5000,10000} |
| 15000 | ✓ | ✓ | {15000} | {5000,10000,15000} |
| 19999 | ✓ | ✓ | {19999} | {5000,10000,15000,19999} |

关键点：上传发生在 `save_state` 之后、下一次 save（删除旧 ckpt）之前，且上传前会 `checkpoint_manager.wait_until_finished()` 确保 async 落盘完成，所以上传到 BOS 的内容一定完整。

#### 失败处理

`_upload_checkpoint_to_bos`（`scripts/train.py`）用 `subprocess` 调用 megfile CLI（`/mnt/vepfs/base2/rongyinze/miniconda3/bin/megfile sync -f`），**上传失败只 log error、不中断训练**——checkpoint 仍在本地（直到下一次 save 才被删），可事后手动 `megfile sync` 补传。日志关键字：

```
Uploading checkpoint step 5000: <local> -> <bos>
BOS upload complete for step 5000        # 成功
BOS upload failed for step 5000: ...      # 失败（含 stderr）
```

> megfile 凭证读自 `~/.aws/credentials` + `~/.config/megfile/megfile.conf`。容器内 `$HOME=/home/rongyinze` 与宿主 `/mnt/vepfs/base2/rongyinze` 是同一挂载，所以凭证天然可用。

### 6.5 TrainConfig 自动记录（config.json）

orbax 的 checkpoint 只存 `params / train_state / assets`，**不存 TrainConfig**。而 TrainConfig 定义在代码里（`config.py` 的 `_CONFIGS`），会被覆盖/改动——一旦改了，旧 checkpoint 再用同名 config 加载时就对不上当初的训练设置（典型如 `repo_id`、`keep_period`）。为此 `train.py` 在训练启动时自动把当前**完整 TrainConfig** 落盘成 `config.json`，让 checkpoint 自描述。

- **写什么**：`dataclasses.asdict(config)` 序列化为 JSON；transforms / weight_loader / filter 等非 JSON 原生字段用 `default=str` 转成字符串表示。涵盖 `name / data.repo_id / model / batch_size / num_train_steps / keep_period / optimizer / lr_schedule` 等全部字段。
- **何时写**：`main()` 里 checkpoint 目录初始化后立即写一次（不依赖 wandb，`--no-wandb-enabled` 也会写）。仅 `process_index==0` 执行，写失败只 log、不影响训练。
- **写到哪**：本地 exp 目录 `<checkpoint_dir>/config.json`；若开了 BOS 上传，同时 `megfile cp` 一份到 `<BOS_UPLOAD_DIR>/<config_name>/<exp_name>/config.json`。
- **实现**：`scripts/train.py` 的 `_save_train_config(config, directory, bos_target_dir)`。

> **无需改启动命令**：该记录是 `train.py` 自动行为，不引入任何新的 CLI 参数或环境变量，5.1 的提交命令照旧。

用途：拿到一个 checkpoint 时，读它的 `config.json` 即可知道是哪个 config、哪个数据集、什么超参产出的——尤其在覆盖式复用 config name（如本仓库直接改 `pi05_xone` 的 `repo_id` 训不同数据集）时，避免靠记忆/猜测。

> 关于本仓库如何用于 pi 的推理，直接看 README 即可。