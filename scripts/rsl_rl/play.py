"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import sys

# Kit's GUI extensions reorder sys.path. Pin this repository's encoder-enabled
# RSL-RL package before startup instead of accidentally importing rsl_rl_lib 3.x.
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "rsl_rl"))
import rsl_rl

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many simulation steps.")
parser.add_argument("--export", action="store_true", help="Explicitly export policy/encoder before playback.")
parser.add_argument("--real_time", action="store_true", help="Limit playback to the simulation clock.")
parser.add_argument("--eval_episodes", type=int, default=0,
                    help="Evaluate this many complete GetUp episodes per environment, then write JSON metrics.")
parser.add_argument("--metrics_path", type=str, default=None, help="Output path for GetUp evaluation JSON.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.path.insert(0, str(_repo_root / "exts/bipedal_locomotion"))
sys.path.insert(0, str(_repo_root / "rsl_rl"))

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import json
import time

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import (
    RslRlPpoAlgorithmMlpCfg, RslRlVecEnvWrapper, export_mlp_as_onnx, export_policy_as_jit,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    cli_args.configure_getup(env_cfg, args_cli)
    if args_cli.eval_episodes:
        if not hasattr(env_cfg, "getup") or args_cli.eval_episodes < 1:
            raise ValueError("--eval_episodes requires a GetUp task and a positive count.")
        env_cfg.getup.curriculum_enabled = False

    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if args_cli.export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    steps = 0
    completed = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    episode_steps = torch.zeros_like(completed)
    records = []
    started = time.monotonic()
    # simulate environment
    while simulation_app.is_running():
        frame_started = time.monotonic()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, dones, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 
            steps += 1
            if args_cli.eval_episodes:
                episode_steps += 1
                # TerminationManager retains this step's flags even after auto-reset.
                success = env.unwrapped.termination_manager.get_term("success")
                accepted = dones.bool() & (completed < args_cli.eval_episodes)
                for idx in accepted.nonzero(as_tuple=False).flatten().tolist():
                    records.append({"env_id": idx, "success": bool(success[idx]),
                                    "duration_s": float(episode_steps[idx] * env.unwrapped.step_dt)})
                completed += accepted.long()
                episode_steps[dones.bool()] = 0
                if bool((completed >= args_cli.eval_episodes).all()):
                    break
            if args_cli.max_steps is not None and steps >= args_cli.max_steps:
                break
            if args_cli.video and not args_cli.eval_episodes and args_cli.max_steps is None and steps >= args_cli.video_length:
                break
        if args_cli.real_time or not args_cli.headless:
            time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - frame_started)))

    if args_cli.eval_episodes:
        successes = [r["duration_s"] for r in records if r["success"]]
        metrics = {"checkpoint": os.path.abspath(resume_path), "seed": env_cfg.seed,
                   "stage": env_cfg.getup.initial_level, "num_envs": env.num_envs,
                   "episodes_per_env": args_cli.eval_episodes, "episodes": len(records),
                   "complete": bool((completed >= args_cli.eval_episodes).all()),
                   "success_rate": len(successes) / len(records) if records else None,
                   "mean_success_time_s": sum(successes) / len(successes) if successes else None,
                   "elapsed_wall_s": time.monotonic() - started, "records": records}
        metrics_path = args_cli.metrics_path or os.path.join(log_dir, f"getup_eval_stage{env_cfg.getup.initial_level}.json")
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as output:
            json.dump(metrics, output, indent=2)
        print(f"[GetUp] Evaluation written to {metrics_path}: {len(successes)}/{len(records)} successes")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
