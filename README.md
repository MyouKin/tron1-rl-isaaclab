# English | [中文](README_cn.md)

# tron1-rl-isaaclab

Reinforcement learning training stack for the LimX **TRON1** bipedal robot, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and using PPO to train locomotion policies. This repository extends the Isaac Lab template to support sim-to-real training for TRON1 robot variants.

## Requirements

- Isaac Sim + Isaac Lab, with isaaclab / isaaclab_tasks / isaaclab_rl importable
- Use the Python bundled with Isaac Sim (locally tested with Isaac Sim 5.0 / Python 3.11)
- GPU (>= 12 GB VRAM recommended for multi-env training)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab

# 2. Editable install of the extension and vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

> **Note:** Robot model assets (USD files) for SF_TRON1A and WF_TRON1A are bundled inside `exts/bipedal_locomotion/bipedal_locomotion/assets/usd/`. No additional model download is required.

## Training

Task IDs are registered in exts/bipedal_locomotion/.

```bash
# Solefoot (SF)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 4096 --headless

# Wheelfoot (WF)
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-Blind-Flat-v0 --num_envs 4096 --headless
```

Common options:

- --resume True --checkpoint_path <path> -- resume from a specific .pt checkpoint
- --video -- enable video recording
- --max_iterations N -- override the maximum iteration count

Log path: logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/

## Recovery (GetUp)

Recovery lives in `exts/bipedal_locomotion/bipedal_locomotion/tasks/recovery/`, alongside `locomotion`, with its own environment, MDP terms and PPO configuration. It reuses the WF base configuration and preserves existing task IDs, checkpoints and RSL-RL compatibility fixes. Leg joints have finite physical limits; wheels remain continuous.

Use `Isaac-Limx-WF-GetUp-Bounded-v0` for training and `Isaac-Limx-WF-GetUp-Bounded-Play-v0` for playback. The existing `model_21000.pt` passed all 33 curriculum stages (99.91% across 16,896 simulated episodes); arbitrary fallen joint configurations remain untested. Training, old-checkpoint inference and headless video recording were verified after the directory migration; desktop GUI remains unresolved.

See [Recovery commands in the Chinese README](README_cn.md#翻倒起身recovery) for resume and recording commands. `manage_getup.py start/status/stop` controls automatic training and evaluation; records are kept in `logs/getup_continuous/`.

## Robot Morphologies

| Morphology | End-effector | Task ID Prefix |
|---|---|---|
| SF_TRON1A | sole foot (ankle pitch) | Isaac-Limx-SF-... |
| WF_TRON1A | wheel | Isaac-Limx-WF-... |

## Related Repositories

| Repository | Description |
|---|---|
| [tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | TRON1 robot model files |
| [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym) | TRON1 RL training with Isaac Gym |
| [tron1-rl-deploy-ros](https://github.com/limxdynamics/tron1-rl-deploy-ros) | TRON1 RL deployment (ROS) |
| [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python) | TRON1 RL deployment (Python) |

## License

[Apache 2.0](LICENCE).
