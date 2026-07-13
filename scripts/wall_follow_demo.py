"""整屋沿边沿墙（wall-following）清扫行为演示。

机器人从外间中央出发，先直行找墙，然后以右侧线激光（与真机一致装在右侧）
反馈把"机身右边缘-墙面"距离保持在 1cm，沿整个屋子的边界逆时针巡边一整圈：
  外间四壁 -> 隔断西面 -> 门洞阳角（三段式包边：直行/原地右转90°/前出找墙，
  5cm 薄隔断的墙端被自然拆成两次连续阳角包边绕行 180°）进入里间 ->
  里间四壁（含积木等障碍绕行）-> 门洞对侧阳角包边出来 -> 继续外间沿墙，循环巡边。
控制逻辑见 envs/vacuum/wall_follower.py（纯规则控制器，不依赖训练模型）。

速度：默认 --speed 2.0（长直段巡航 0.24m/s，为基础速度的 2 倍；阳角包边、
转角保持原速以保证包边几何精度），一圈约 230s；--speed 1.0 回到基础速度。

用法：
  # 带可视化（空格暂停/继续）
  $env:PYTHONPATH="."; python scripts/wall_follow_demo.py

  # 无渲染跑完整一圈并输出贴边精度/整屋覆盖统计
  $env:PYTHONPATH="."; python scripts/wall_follow_demo.py --headless --steps 12000
"""

import os
import sys
import time
import argparse

import numpy as np
import mujoco
import transforms3d as tf3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vacuum.vacuum_env import VacuumEnv
from envs.vacuum.gen_xml import MAX_WHEEL_TORQUE
from envs.vacuum.wall_follower import WallFollower, FollowState
from tasks.vacuum_task import VacuumModes


def reset_flat(env, start_xy=(0.0, 0.0), yaw=np.pi / 2):
    """重置为平地场景（无动态门槛），并把机器人放到指定位姿。"""
    env.reset()
    task = env.task
    task.mode = VacuumModes.FLAT
    task._bury_all_boxes()
    task.thr_pos = np.array([10.0, 0.0, -1.0])
    task.thr_height = 0.0

    qpos = env.interface.get_qpos()
    qvel = np.zeros(env.model.nv)
    adr = env.interface.get_jnt_qposadr_by_name('root')[0]
    qpos[adr + 0], qpos[adr + 1], qpos[adr + 2] = start_xy[0], start_xy[1], 0.05
    qpos[adr + 3:adr + 7] = tf3.euler.euler2quat(0, 0, yaw)
    env.set_state(qpos, qvel)
    task.prev_xy = np.array(start_xy, dtype=np.float64)


def _ray_mat(direction):
    """构造把局部 +z 对到 direction 的旋转矩阵（marker 用）。"""
    z = direction / (np.linalg.norm(direction) + 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x / n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).flatten()


def draw_lasers(env, viewer):
    """把每条激光射线画成细圆柱（绿=命中，灰=无回波），命中点画红球。"""
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    for laser in (env.laser_front, env.laser_right):
        for origin, direction, dist, hit in laser.ray_states():
            mid = origin + direction * dist / 2.0
            rgba = np.array([0.1, 0.9, 0.2, 0.5]) if hit else np.array([0.6, 0.6, 0.6, 0.25])
            viewer.add_marker(pos=mid.tolist(), mat=_ray_mat(direction),
                              size=np.array([0.0015, 0.0015, dist / 2.0]),
                              rgba=rgba, type=cyl, label="")
            if hit:
                viewer.add_marker(pos=(origin + direction * dist).tolist(),
                                  size=np.ones(3) * 0.006,
                                  rgba=np.array([1.0, 0.1, 0.1, 0.9]),
                                  type=sphere, label="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16000,
                        help="控制步数(40Hz)，16000 步约覆盖整屋一圈")
    parser.add_argument("--headless", action="store_true", help="无渲染，只输出统计")
    parser.add_argument("--noise", type=float, default=0.0, help="激光测距噪声std(m)")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="行进速度倍率（沿墙巡航 0.12m/s x 该倍率）")
    args = parser.parse_args()

    env = VacuumEnv()
    env.laser_front.noise_std = args.noise
    env.laser_right.noise_std = args.noise

    # 从房间中央出发，朝北墙（避开门洞方向），控制器自行找墙、贴边
    reset_flat(env, start_xy=(0.0, 0.0), yaw=np.pi / 2)
    follower = WallFollower(dt=env.robot.control_dt, speed_scale=args.speed)

    viewer = None
    if not args.headless:
        env.render()
        viewer = env.viewer
        viewer._paused = True

    follow_errs = []          # FOLLOW 状态下的横向距离误差
    state_time = {s: 0.0 for s in FollowState}
    dt = env.robot.control_dt
    prev_xy = env.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()

    # 整屋覆盖统计：隔断在 x=2.0，里间为 x>2.0；门洞穿越 = 在门洞范围内跨过隔断
    outer_t = inner_t = 0.0
    door_cross = 0
    prev_side = prev_xy[0] > 2.0

    for ts in range(args.steps):
        t0 = time.time()
        front = env.laser_front.read()
        side = env.laser_right.read()
        wheel_vel = env.interface.get_act_joint_velocities()

        # 里程计速度（真机来自轮式里程计+陀螺融合，这里取机身水平速度模长）
        cur_xy = env.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()
        odom_speed = float(np.linalg.norm(cur_xy - prev_xy)) / dt
        prev_xy = cur_xy

        tau = follower.step(front, side, wheel_vel, odom_speed=odom_speed)
        env.robot.step(tau / MAX_WHEEL_TORQUE)   # robot.step 内部 × torque_scale

        state_time[follower.state] += dt
        cur_side = cur_xy[0] > 2.0
        if cur_side:
            inner_t += dt
        else:
            outer_t += dt
        if cur_side != prev_side and abs(cur_xy[1]) < 0.45:
            door_cross += 1
        prev_side = cur_side

        if follower.state == FollowState.FOLLOW and side.hit:
            # steady=True 表示已进入 FOLLOW 超过 4s（排除拐角后的重新收敛段）
            follow_errs.append((side.distance - follower.TARGET_DIST,
                                follower._state_t > 4.0))

        if viewer is not None:
            draw_lasers(env, viewer)
            env.render()
            delay = max(0, env.frame_skip * env.robot.client.sim_dt() - (time.time() - t0))
            time.sleep(delay)

        if ts % 200 == 0:
            pos = env.interface.get_object_xpos_by_name('base', 'OBJ_BODY')
            print("t={:5.1f}s state={:<10s} pos=({:+.2f},{:+.2f}) "
                  "front={:.3f}/{:.2f} right={:.3f}/{:.2f} v={:.2f} w={:+.2f}".format(
                      ts * dt, follower.state.name, pos[0], pos[1],
                      front.distance, front.confidence,
                      side.distance, side.confidence,
                      follower.cmd_v, follower.cmd_w))

    print("\n========== 沿边统计 ==========")
    for s, t in state_time.items():
        print("  {:<11s}: {:6.1f}s".format(s.name, t))
    print("  过坎强推   : {}".format(follower.climb_count))
    print("  脱困次数   : {}".format(follower.escape_count))
    print("========== 整屋覆盖 ==========")
    print("  外间时长   : {:6.1f}s".format(outer_t))
    print("  里间时长   : {:6.1f}s".format(inner_t))
    print("  门洞穿越   : {} 次{}".format(
        door_cross, "（进出里间均成功，整屋沿墙贯通）" if door_cross >= 2 else ""))
    if follow_errs:
        errs = np.abs(np.array([e for e, _ in follow_errs]))
        steady = np.abs(np.array([e for e, s in follow_errs if s]))
        print("FOLLOW 全程 |d-1cm|: 平均 {:.1f}mm, 中位 {:.1f}mm, 最大 {:.1f}mm".format(
            errs.mean() * 1000, np.median(errs) * 1000, errs.max() * 1000))
        if steady.size:
            print("FOLLOW 稳态(>4s) |d-1cm|: 平均 {:.1f}mm, 中位 {:.1f}mm, "
                  "±5mm 内占比 {:.0f}%".format(
                      steady.mean() * 1000, np.median(steady) * 1000,
                      100.0 * (steady < 0.005).mean()))
    env.close()


if __name__ == '__main__':
    main()
