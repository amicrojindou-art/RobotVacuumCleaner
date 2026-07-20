# -*- coding: utf-8 -*-
"""方案A（力矩→轮速适配层）样本 demo：在训练仿真里闭环验证整条大核链路。

背景：vacuum_wf 策略输出【轮力矩】，但大核只能向小核下发【轮速目标】
（g_vels_shm，小核做电机速度闭环）。方案A在大核侧插一个虚拟动力学层：

  policy 力矩 (40Hz)
    └─> VirtualDiffDrive：按训练仿真的名义动力学积分出 (v, ω) → 左右轮速目标
          └─> 速度伺服（本 demo 里是仿真子步 P 控制器，扮演小核电机环）→ 轮子

本 demo 在同一批随机墙课程上对比两种模式：
  direct  —— 力矩直出（训练/评估的原始通路，基线）
  adapter —— 上述方案A完整通路（策略只看观测，不知道底下换了执行方式）

若 adapter 的回报/贴边精度接近 direct，说明"力矩策略 + 轮速接口"在动力学
层面可行，可以推进大核 C++ 实现；差距大则优先调 kp/惯量/阻尼参数，或者
直接走方案B（速度动作重训，见 docs/2026-07-15-plan-b-velocity-action-retrain.md）。

用法（RL312 环境，仓库根目录）：
  $env:PYTHONPATH="."; python scripts/demo_plan_a_adapter.py --episodes 8
  # 追加 --save-obs 会把 direct 模式的观测录成 npz，供转换链对拍复测

实验结论（2026-07-15，vacuum_wf0715，8 集同种子）：
  - k_fb=1.0（编码器反馈全回拉 = "力矩→加速度指令"接口）：return 60.8 vs
    direct 75.5（80%），贴边误差 11.0mm 优于 direct 14.3mm，碰撞 1% vs 4%
    —— 方案A可行，设备端 rl_wf_action 按 k_fb=1 形态实现；
  - k_fb=0.5 → 25.4，k_fb=0（纯开环积分）→ 24.9/-22（质量错灌 350kg 时
    完全失效）：虚拟状态必须锚定实测轮速，纯开环不可用。
"""
import os
import sys
import json
import argparse

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from envs.vacuum.vacuum_wf_env import VacuumWFEnv
from envs.vacuum.robot import VacuumRobot
from envs.vacuum.gen_xml import MAX_WHEEL_TORQUE, WHEEL_RADIUS, WHEEL_Y, CHASSIS_RADIUS

# 观测向量索引（envs/vacuum/vacuum_wf_env.py get_obs，改动需同步）
OBS_SIDE_D, OBS_SIDE_CONF, OBS_CONTACT_DIR = 16, 17, 19
LASER_MAX_RANGE = 0.10
TARGET_DIST = 0.01


class VirtualDiffDrive(object):
    """大核侧虚拟动力学：把策略力矩积分成轮速目标（名义差速底盘模型）。

    真机上这段逻辑将原样进入 rl_wf_action.cpp：状态只有 (v, ω) 两个浮点，
    参数与训练仿真对齐（质量/惯量/轮径/半轮距），阻尼项防零输入漂移。
    """

    def __init__(self, mass, inertia_z, wheel_r=WHEEL_RADIUS, half_track=WHEEL_Y,
                 c_v=0.5, c_w=1.0, v_max=0.6, w_max=6.0):
        self.m = mass
        self.iz = inertia_z
        self.r = wheel_r
        self.half_track = half_track
        self.c_v = c_v          # 线速度黏性阻尼 (1/s)，代偿真实滚阻
        self.c_w = c_w          # 角速度黏性阻尼 (1/s)
        self.v_max = v_max
        self.w_max = w_max
        self.v = 0.0            # 虚拟线速度 (m/s)
        self.w = 0.0            # 虚拟角速度 (rad/s)

    def reset(self):
        self.v = 0.0
        self.w = 0.0

    def sync(self, v_meas, w_meas, k):
        """用实测速度回拉虚拟状态（真机=编码器反馈）。

        没有这一步，顶墙/打滑时虚拟模型自由积分飞到 v_max，setpoint 与实际
        轮速严重脱节（首轮实验伺服误差 8 rad/s，策略行为完全失真）。
        k=1 时退化为"力矩→加速度指令"接口：w_set = w_meas + a·dt，完全贴地。
        """
        self.v += k * (v_meas - self.v)
        self.w += k * (w_meas - self.w)

    def update(self, tau_l, tau_r, dt):
        """输入左右轮力矩 (N·m)，返回左右轮速目标 (rad/s)。"""
        acc = (tau_l + tau_r) / (self.r * self.m) - self.c_v * self.v
        alpha = (tau_r - tau_l) * self.half_track / (self.r * self.iz) - self.c_w * self.w
        self.v = float(np.clip(self.v + acc * dt, -self.v_max, self.v_max))
        self.w = float(np.clip(self.w + alpha * dt, -self.w_max, self.w_max))
        wl = (self.v - self.w * self.half_track) / self.r
        wr = (self.v + self.w * self.half_track) / self.r
        return wl, wr


class ServoAdapterRobot(VacuumRobot):
    """方案A执行链：力矩指令 → 虚拟动力学 → 轮速目标 → 子步速度伺服。

    子步 P 伺服扮演小核电机速度环（真机由小核以更高频闭环完成）。
    prev_action/prev_torque 的记账语义与基类一致，观测与奖励管线无感。
    稳定性约束：kp < J_eff/sim_dt ≈ (m/2)*r²/0.0025 ≈ 0.71，默认 0.5。
    """

    def __init__(self, pdgains, dt, active, client, torque_scale, virt, kp=0.5, k_fb=1.0):
        VacuumRobot.__init__(self, pdgains, dt, active, client, torque_scale)
        self.virt = virt
        self.kp = kp
        self.k_fb = k_fb            # 编码器反馈回拉增益（每控制步）
        self.track_err_sum = 0.0    # 伺服跟踪误差统计 (rad/s)
        self.track_err_n = 0

    def _measured_vw(self):
        wv = np.asarray(self.client.get_act_joint_velocities())[self.actuators]
        rim = wv * self.virt.r                       # 轮缘线速 (m/s)
        v = 0.5 * (rim[0] + rim[1])
        w = (rim[1] - rim[0]) / (2.0 * self.virt.half_track)
        return v, w

    def step(self, action):
        action = np.asarray(action).flatten()
        tau_cmd = np.clip(action * self.torque_scale, self.tau_low, self.tau_high)

        self.last_action = action.copy() if self.prev_action is None else self.prev_action.copy()
        self.last_torque = tau_cmd.copy() if self.prev_torque is None else self.prev_torque.copy()

        if self.k_fb > 0.0:
            v_meas, w_meas = self._measured_vw()
            self.virt.sync(v_meas, w_meas, self.k_fb)
        wl_set, wr_set = self.virt.update(tau_cmd[0], tau_cmd[1], self.control_dt)
        w_set = np.array([wl_set, wr_set])

        for _ in range(self.frame_skip):
            for dof, sign in self._brush_dofs:
                self.client.data.qvel[dof] = sign * self.BRUSH_SPEED
            wv = np.asarray(self.client.get_act_joint_velocities())[self.actuators]
            err = w_set - wv
            servo_tau = np.clip(self.kp * err, self.tau_low, self.tau_high)
            self.client.set_motor_torque(servo_tau)
            self.client.step()
            self.track_err_sum += float(np.abs(err).mean())
            self.track_err_n += 1

        self.prev_action = action.copy()
        self.prev_torque = tau_cmd.copy()
        return tau_cmd


def load_policy(path):
    actor = torch.load(path, map_location="cpu", weights_only=False)
    actor.eval()

    def act(obs):
        with torch.no_grad():
            a = actor.forward(torch.tensor(obs, dtype=torch.float32), deterministic=True)
        return a.numpy().flatten()
    return act


def run_mode(env, policy, mode, episodes, max_steps, seed, collect_obs=None):
    results = []
    np.random.seed(seed)   # 两种模式同种子 → 同一批随机墙课程，公平对比
    torch.manual_seed(seed)
    for ep in range(episodes):
        obs = env.reset()
        if isinstance(env.robot, ServoAdapterRobot):
            env.robot.virt.reset()
            env.robot.track_err_sum, env.robot.track_err_n = 0.0, 0

        ep_ret, steps, contact_steps = 0.0, 0, 0
        side_errs = []
        dist_travelled = 0.0
        prev_xy = env.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()

        for _ in range(max_steps):
            if collect_obs is not None:
                collect_obs.append(np.asarray(obs, dtype=np.float32).copy())
            a = policy(obs)
            obs, rew, done, _ = env.step(a)
            ep_ret += rew
            steps += 1
            if obs[OBS_SIDE_CONF] > 0.9:
                side_errs.append(abs(obs[OBS_SIDE_D] * LASER_MAX_RANGE - TARGET_DIST))
            if obs[OBS_CONTACT_DIR] != 0.0:
                contact_steps += 1
            cur_xy = env.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()
            dist_travelled += float(np.linalg.norm(cur_xy - prev_xy))
            prev_xy = cur_xy
            if done:
                break

        r = {
            "mode": mode, "episode": ep, "return": round(ep_ret, 2), "steps": steps,
            "terminated_early": steps < max_steps,
            "dist_m": round(dist_travelled, 2),
            "side_err_mean_mm": round(1e3 * float(np.mean(side_errs)), 1) if side_errs else None,
            "side_err_median_mm": round(1e3 * float(np.median(side_errs)), 1) if side_errs else None,
            "side_engaged_ratio": round(len(side_errs) / max(steps, 1), 2),
            "contact_ratio": round(contact_steps / max(steps, 1), 2),
        }
        if isinstance(env.robot, ServoAdapterRobot) and env.robot.track_err_n:
            r["servo_track_err_rad_s"] = round(env.robot.track_err_sum / env.robot.track_err_n, 3)
        results.append(r)
        print("  [{}] ep{:02d} ret={:8.2f} steps={:3d}{} dist={:5.2f}m side_err={}mm eng={:.0%} contact={:.0%}{}".format(
            mode, ep, r["return"], steps, "*" if r["terminated_early"] else " ",
            r["dist_m"], r["side_err_mean_mm"], r["side_engaged_ratio"], r["contact_ratio"],
            " servo_err={:.3f}rad/s".format(r["servo_track_err_rad_s"]) if "servo_track_err_rad_s" in r else ""))
    return results


def summarize(results):
    agg = {}
    for mode in sorted(set(r["mode"] for r in results)):
        rs = [r for r in results if r["mode"] == mode]
        errs = [r["side_err_mean_mm"] for r in rs if r["side_err_mean_mm"] is not None]
        agg[mode] = {
            "episodes": len(rs),
            "return_mean": round(float(np.mean([r["return"] for r in rs])), 2),
            "return_std": round(float(np.std([r["return"] for r in rs])), 2),
            "steps_mean": round(float(np.mean([r["steps"] for r in rs])), 1),
            "early_term": sum(r["terminated_early"] for r in rs),
            "side_err_mean_mm": round(float(np.mean(errs)), 1) if errs else None,
            "contact_ratio_mean": round(float(np.mean([r["contact_ratio"] for r in rs])), 2),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/vacuum_wf0715/actor.pt")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kp", type=float, default=0.5, help="子步速度伺服增益 N·m/(rad/s)，须 <0.71")
    ap.add_argument("--k-fb", type=float, default=1.0,
                    help="编码器反馈回拉增益/控制步；1=加速度指令接口，0=纯开环积分")
    ap.add_argument("--c-v", type=float, default=0.5, help="虚拟动力学线速度阻尼 (1/s)")
    ap.add_argument("--c-w", type=float, default=1.0, help="虚拟动力学角速度阻尼 (1/s)")
    ap.add_argument("--iz", type=float, default=None, help="虚拟惯量，默认 0.5*m*R²")
    ap.add_argument("--save-obs", action="store_true", help="录制 direct 模式观测，供转换链对拍")
    ap.add_argument("--out-dir", default=None, help="默认 <checkpoint目录>/plan_a_demo")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "plan_a_demo")
    os.makedirs(out_dir, exist_ok=True)

    policy = load_policy(args.checkpoint)
    env = VacuumWFEnv()
    # 机器人子树质量（base 及以下），不能用 mj_getTotalmass —— 那是含墙体
    # 地形的全场景质量（实测 350kg），会把虚拟动力学灌死
    mass = float(env.model.body_subtreemass[env.model.body('base').id])
    iz = args.iz if args.iz is not None else 0.5 * mass * CHASSIS_RADIUS ** 2
    print("[cfg] mass={:.2f}kg iz={:.4f}kg·m² kp={} c_v={} c_w={}".format(mass, iz, args.kp, args.c_v, args.c_w))

    # -------- 基线：力矩直出（训练原始通路） --------
    print("[run] direct（基线，力矩直出）")
    obs_log = [] if args.save_obs else None
    results = run_mode(env, policy, "direct", args.episodes, args.max_steps, args.seed,
                       collect_obs=obs_log)

    # -------- 方案A：虚拟动力学 + 速度伺服 --------
    print("[run] adapter（方案A：力矩→轮速→伺服）")
    virt = VirtualDiffDrive(mass, iz, c_v=args.c_v, c_w=args.c_w)
    env.robot = ServoAdapterRobot(None, env.robot.control_dt, [0, 1], env.interface,
                                  MAX_WHEEL_TORQUE, virt, kp=args.kp, k_fb=args.k_fb)
    results += run_mode(env, policy, "adapter", args.episodes, args.max_steps, args.seed)

    agg = summarize(results)
    print("\n================ 汇总（direct vs adapter）================")
    for mode, s in agg.items():
        print("  {:<8s} return {:>8.2f} ±{:<7.2f} steps {:>6.1f} 早终止 {}/{} "
              "贴边误差 {}mm 碰撞占比 {:.0%}".format(
                  mode, s["return_mean"], s["return_std"], s["steps_mean"],
                  s["early_term"], s["episodes"],
                  s["side_err_mean_mm"], s["contact_ratio_mean"]))

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "mass": mass, "iz": iz,
                   "episodes": results, "summary": agg}, f, ensure_ascii=False, indent=2)
    print("[ok] results ->", os.path.join(out_dir, "results.json"))

    if obs_log:
        obs_path = os.path.join(out_dir, "rollout_obs.npz")
        np.savez_compressed(obs_path, obs=np.asarray(obs_log, dtype=np.float32))
        print("[ok] obs({}) -> {}".format(len(obs_log), obs_path))


if __name__ == "__main__":
    main()
