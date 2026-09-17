# 中文 | [English](README.md)

# tron1-rl-isaaclab

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 LimX **TRON1** 双足机器人强化学习训练栈，使用 PPO 训练 locomotion 策略。本仓库扩展了 Isaac Lab 模板，支持 TRON1 机器人变体的 Sim-to-Real 训练。

## 环境要求

- Isaac Sim + Isaac Lab，且 isaaclab / isaaclab_tasks / isaaclab_rl 可被 import
- 使用与 Isaac Sim 配套的 Python（本机为 Isaac Sim 5.0 / Python 3.11）
- GPU（推荐 >= 12 GB 显存，用于多环境训练）

## 安装

```bash
# 1. clone 仓库
git clone https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab

# 2. editable install extension 与 vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

> **说明：** 机器人模型资产（USD 文件）已内置在 `exts/bipedal_locomotion/bipedal_locomotion/assets/usd/` 中，覆盖 SF_TRON1A 和 WF_TRON1A 两种变体，无需额外下载模型文件。

## 训练

任务 ID 在 exts/bipedal_locomotion/ 中注册。

```bash
# Solefoot (SF)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 4096 --headless

# Wheelfoot (WF)
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-Blind-Flat-v0 --num_envs 4096 --headless
```

常用选项：

- --resume True --checkpoint_path <path> -- 从指定的 .pt checkpoint 恢复训练
- --video -- 启用视频录制
- --max_iterations N -- 覆盖最大迭代次数

日志路径：logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/

## 翻倒起身（Recovery）

代码位于 `exts/bipedal_locomotion/bipedal_locomotion/tasks/recovery/`，与 `locomotion` 平级，包含环境配置、`mdp/` 和 `agents/`。复用 WF 机器人与基础配置；旧任务 ID、checkpoint 和 RSL-RL 兼容修复保留。腿关节使用有限角度及物理限位，轮子可连续转动。

当前 `model_21000.pt` 已通过原任务 33 阶段仿真评估（99.91%，16,896 回合）。随机关节姿态使用下面的 Fallen 扩展独立评估与续训。迁移后已验证训练、旧模型推理和无窗口录制，GUI 问题尚未解决。

在仓库根目录执行（以下为本机路径）：

```bash
SIM_PY=/home/myoukin/isaacsim/python.sh
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="$PWD/logs/rsl_rl/wf_tron_1a_getup_bounded/2026-09-17_12-43-05_continuous_e10a7a122f48/model_21000.pt"

# 按需继续训练，另存新 checkpoint
"$SIM_PY" scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-GetUp-Bounded-v0 --num_envs 4096 \
  --resume True --checkpoint_path "$CHECKPOINT" \
  --max_iterations 2000 --save_interval 100 --headless

# 播放旧模型并录制约 15 秒视频
"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Limx-WF-GetUp-Bounded-Play-v0 --checkpoint_path "$CHECKPOINT" \
  --num_envs 1 --getup_stage 32 --getup_fixed_stage \
  --headless --video --video_length 750
```

视频写入 checkpoint 同级的 `videos/play/`，重复录制前需保留已有同名文件。独立评估使用 `scripts/rsl_rl/evaluate_getup.py`（参数见 `--help`）。自动训练入口为 `manage_getup.py start/status/stop`：每 2000 次更新评估，自动晋级和记录；已有状态目录会恢复原进度，达标后自动停止。评估记录位于 `logs/getup_continuous/`。

### 随机关节倒地（Fallen）

`recovery/fallen_pose_env_cfg.py` 定义被动姿态生成环境及新任务 `Isaac-Limx-WF-Recovery-Fallen-v0`（播放加 `-Play`）。生成时关闭腿的位置驱动，随机合法关节和身体朝向，自然落稳后记录状态；训练时恢复正常执行器，从姿态库 reset。数据文件位于 `data/recovery/fallen_v1.pt`，请与日志分开保留。

训练集 4,096 个、测试集 1,536 个状态独立生成并检查重复。按实际关节偏离默认姿态的程度分三档自动晋级，混合 30% 原任务姿态；这三档是关节姿态难度，不是原来的身体倾角阶段。每 2,000 次更新评估完整测试集及原任务，退化暂停，连续两次达标停止。测试集用于开发验收，不代表任意姿态或实机都已覆盖。

```bash
# 生成新姿态库（已有文件时拒绝覆盖）
"$SIM_PY" scripts/recovery/generate_fallen_poses.py --headless \
  --train_poses 4096 --test_poses 1536 --seed 20260920 --output data/recovery/fallen_v1.pt

# 首次启动后台训练；已有状态则恢复，重复启动不会另开进程
"$SIM_PY" scripts/recovery/manage_recovery.py start \
  --checkpoint "$CHECKPOINT" --pose_bank data/recovery/fallen_v1.pt \
  --num_envs 4096 --chunk_iterations 2000 --iterations 6000

# 检查数据完整性、实时迭代和最近评估
"$SIM_PY" scripts/recovery/check_recovery.py
# 在当前训练/评估段结束后暂停
"$SIM_PY" scripts/recovery/manage_recovery.py stop
```

控制器记录在 `logs/recovery_fallen/continuous/`，新 checkpoint 在 `logs/rsl_rl/wf_tron_1a_fallen/`。旧模型全测试集评估可用 `scripts/recovery/evaluate_fallen_poses.py --checkpoint "$CHECKPOINT" --pose_bank data/recovery/fallen_v1.pt --output logs/recovery_fallen/evaluation.json --headless`（用 `$SIM_PY` 运行）。播放新任务沿用 `play.py`，传入新 task ID、`--pose_bank data/recovery/fallen_v1.pt` 和所需 checkpoint。

## 机器人形态

| 形态 | 末端 | Task ID 前缀 |
|---|---|---|
| SF_TRON1A | sole foot (ankle pitch) | Isaac-Limx-SF-... |
| WF_TRON1A | wheel | Isaac-Limx-WF-... |

## 相关仓库

| 仓库 | 描述 |
|---|---|
| [tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | TRON1 机器人模型文件 |
| [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym) | TRON1 Isaac Gym RL 训练 |
| [tron1-rl-deploy-ros](https://github.com/limxdynamics/tron1-rl-deploy-ros) | TRON1 RL 部署（ROS） |
| [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python) | TRON1 RL 部署（Python） |

## 许可证

[Apache 2.0](LICENCE)。
