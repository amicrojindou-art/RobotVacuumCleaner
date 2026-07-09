"""扫地机器人过坎/脱困策略回放与可视化。

用法：
  PYTHONPATH=.:$PYTHONPATH python scripts/debug_vacuum.py --path experiments/vacuum
  # Windows PowerShell:
  $env:PYTHONPATH="."; python scripts/debug_vacuum.py --path experiments/vacuum

可选 --mode 强制锁定场景 (flat/cross/highcenter/against)，便于单独检验某个能力。
"""

import os
import sys
import time
import argparse
import torch
import pickle
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import import_env
from tasks.vacuum_task import VacuumModes


def print_reward(ep_rewards):
    if not ep_rewards:
        return
    mean_rewards = {k: [] for k in ep_rewards[-1].keys()}
    print('*********************************')
    for key in mean_rewards.keys():
        l = [step[key] for step in ep_rewards]
        mean_rewards[key] = sum(l) / len(l)
        print(key, ': ', mean_rewards[key])
    print('*********************************')
    print("mean per step reward: ", sum(mean_rewards.values()))


def draw_markers(env, viewer):
    task = env.task
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    # 目标点（绿色）
    viewer.add_marker(pos=[task.goal_pos[0], task.goal_pos[1], 0.05],
                      size=np.ones(3) * task.TARGET_RADIUS,
                      rgba=np.array([0, 1, 0, 0.35]), type=sphere, label="GOAL")
    # 主障碍中心（红色）
    if task.thr_height > 1e-3:
        viewer.add_marker(pos=task.thr_pos.tolist(),
                          size=np.ones(3) * 0.04,
                          rgba=np.array([1, 0, 0, 1]), type=sphere,
                          label="thr h={:.3f}".format(task.thr_height))


MODE_MAP = {
    'flat': VacuumModes.FLAT,
    'cross': VacuumModes.CROSS,
    'highcenter': VacuumModes.HIGHCENTER,
    'against': VacuumModes.AGAINST,
}


def run(env, policy, force_mode=None):
    observation = env.reset()

    # 可选：强制锁定场景后重置一次
    if force_mode is not None and force_mode in MODE_MAP:
        # 反复 reset 直到抽到目标模式（reset 内部随机选模式）
        for _ in range(100):
            if env.task.mode == MODE_MAP[force_mode]:
                break
            observation = env.reset()
        print("场景模式:", env.task.mode.name)

    env.render()
    viewer = env.viewer
    viewer._paused = True
    done = False
    ts, end_ts = 0, 2000
    ep_rewards = []

    while (ts < end_ts) and (done is False):
        start = time.time()
        with torch.no_grad():
            action = policy.forward(torch.Tensor(observation), deterministic=True).detach().numpy()
        observation, _, done, info = env.step(action.copy())
        ep_rewards.append(info)

        draw_markers(env, viewer)
        env.render()

        sim_dt = env.robot.client.sim_dt()
        delaytime = max(0, env.frame_skip / (1 / sim_dt) - (time.time() - start))
        time.sleep(delaytime)
        ts += 1

    print("Episode finished after {} timesteps".format(ts))
    print("到达目标:" , env.task._reached_goal)
    print_reward(ep_rewards)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='experiments/vacuum', required=False, type=str,
                        help="训练输出目录或 actor.pt 路径")
    parser.add_argument("--mode", default=None, required=False,
                        choices=['flat', 'cross', 'highcenter', 'against'],
                        help="强制锁定场景模式，便于单独检验")
    args = parser.parse_args()

    if os.path.isfile(args.path) and args.path.endswith(".pt"):
        path_to_actor = args.path
        path_to_pkl = os.path.join(os.path.dirname(args.path), "experiment.pkl")
    else:
        path_to_actor = os.path.join(args.path, "actor.pt")
        path_to_pkl = os.path.join(args.path, "experiment.pkl")

    run_args = pickle.load(open(path_to_pkl, "rb"))
    policy = torch.load(path_to_actor)
    policy.eval()
    env = import_env(run_args.env)()

    run(env, policy, force_mode=args.mode)
    print("-----------------------------------------")


if __name__ == '__main__':
    main()
