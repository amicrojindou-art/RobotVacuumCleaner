"""沿边沿墙 RL 策略回放与可视化。

用法：
  # 随机墙面课程（与训练同分布）
  $env:PYTHONPATH="."; python scripts/debug_wallfollow.py --path experiments/vacuum_wf

  # 家居场景（验证泛化：策略只见过随机折线墙，从未见过该户型）
  $env:PYTHONPATH="."; python scripts/debug_wallfollow.py --path experiments/vacuum_wf --home
"""

import os
import sys
import time
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vacuum.vacuum_wf_env import VacuumWFEnv
from scripts.wall_follow_demo import draw_lasers


def run(env, policy, episodes):
    for ep in range(episodes):
        observation = env.reset()
        env.render()
        viewer = env.viewer
        viewer._paused = True

        done = False
        ts, end_ts = 0, 1200
        ep_rewards = []
        track_errs = []

        while (ts < end_ts) and (not done):
            start = time.time()
            with torch.no_grad():
                action = policy.forward(torch.Tensor(observation),
                                        deterministic=True).detach().numpy()
            observation, _, done, info = env.step(action.copy())
            ep_rewards.append(info)

            left = env.laser_left.read()
            if left.hit:
                track_errs.append(abs(left.distance - env.task.TARGET_DIST))

            draw_lasers(env, viewer)
            env.render()

            sim_dt = env.robot.client.sim_dt()
            delay = max(0, env.frame_skip * sim_dt - (time.time() - start))
            time.sleep(delay)
            ts += 1

        print("Episode {} finished after {} timesteps".format(ep, ts))
        if ep_rewards:
            keys = ep_rewards[-1].keys()
            for k in keys:
                print("  {:<10s}: {:+.4f}".format(
                    k, float(np.mean([step[k] for step in ep_rewards]))))
        if track_errs:
            errs = np.array(track_errs)
            print("  贴边误差 |d-1cm|: 平均 {:.1f}mm, 中位 {:.1f}mm".format(
                errs.mean() * 1000, np.median(errs) * 1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='experiments/vacuum_wf', type=str,
                        help="训练输出目录或 actor.pt 路径")
    parser.add_argument("--home", action="store_true",
                        help="在家居场景中回放（验证泛化）")
    parser.add_argument("--episodes", default=5, type=int)
    args = parser.parse_args()

    if os.path.isfile(args.path) and args.path.endswith(".pt"):
        path_to_actor = args.path
    else:
        path_to_actor = os.path.join(args.path, "actor.pt")

    policy = torch.load(path_to_actor)
    policy.eval()

    env = VacuumWFEnv(home_scene=args.home)

    # 观测维度校验：环境观测已从 22 维升级到 34 维（幽灵墙修复轮加入激光
    # 历史帧），旧模型不兼容——给出明确提示而不是维度错误闪退
    expected = None
    if getattr(policy, 'obs_mean', None) is not None:
        expected = int(policy.obs_mean.numel())
    actual = int(env.observation_space.shape[0])
    if expected is not None and expected != actual:
        print("=" * 60)
        print("[错误] 模型与环境观测维度不匹配！")
        print("  actor.pt 期望输入 {} 维，当前环境观测为 {} 维。".format(expected, actual))
        print("  原因：环境观测在 2026-07-07 幽灵墙修复中从 22 维升级到 34 维，")
        print("        修复前训练的旧模型已不兼容（且旧模型是在无碰撞的幽灵墙")
        print("        上训练的，本身无效）。")
        print("  解决：用修复后重新训练的输出目录，例如：")
        print("        python scripts/debug_wallfollow.py --path experiments/vacuum_wf2")
        print("=" * 60)
        sys.exit(1)

    run(env, policy, args.episodes)
    env.close()


if __name__ == '__main__':
    main()
