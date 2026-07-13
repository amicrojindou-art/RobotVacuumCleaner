"""线激光传感器特性验证（无渲染，直接读传感器）。

验证 gen_xml 中建模的两条关键特性：
  1) 有效测距范围 10cm：量程内测距准确，超量程无回波；
  2) 有效检测高度约 7cm：>=7cm 可靠（confidence=1）、4~6cm 低置信（0<conf<1）、
     2~3cm 盲区（conf=0）。

用法：
  $env:PYTHONPATH="."; python scripts/test_line_laser.py
"""

import os
import sys

import numpy as np
import mujoco
import transforms3d as tf3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vacuum.vacuum_env import VacuumEnv
from envs.vacuum.gen_xml import CHASSIS_RADIUS, LASER_MAX_RANGE


def set_robot_pose(env, x=0.0, y=0.0, yaw=0.0):
    qpos = env.interface.get_qpos()
    qvel = np.zeros(env.model.nv)
    adr = env.interface.get_jnt_qposadr_by_name('root')[0]
    qpos[adr + 0], qpos[adr + 1], qpos[adr + 2] = x, y, 0.05
    qpos[adr + 3:adr + 7] = tf3.euler.euler2quat(0, 0, yaw)
    env.set_state(qpos, qvel)


def place_box(env, name, x, y, height, yaw=0.0):
    """把 mocap 墙块摆到 (x,y)、朝向 yaw，有效高度 height（下沉入地实现）。"""
    model, data = env.model, env.data
    half_z = float(model.geom(name).size[2])
    mid = model.body_mocapid[model.body(name).id]
    data.mocap_pos[mid] = np.array([x, y, height - half_z])
    data.mocap_quat[mid] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    mujoco.mj_forward(env.model, env.data)


def check(desc, cond):
    tag = "PASS" if cond else "FAIL"
    print("[{}] {}".format(tag, desc))
    return cond


def main():
    env = VacuumEnv()
    env.reset()
    env.task._bury_all_boxes()
    # 挪到空旷处，避开家居场景的静态物体
    set_robot_pose(env, x=1.0, y=0.0, yaw=0.0)

    ok = True
    edge_x = 1.0 + CHASSIS_RADIUS   # 机身前缘的世界 x 坐标
    half_t = float(env.model.geom('box01').size[1])   # 墙块半厚

    # ---------- 特性1：测距范围 ----------
    print("== 测距范围（前向激光，20cm 高墙）==")
    for gap, expect_hit in [(0.03, True), (0.06, True), (0.09, True), (0.12, False)]:
        # 墙块转 90 度，厚度方向朝 x，内侧面在 edge_x+gap 处
        place_box(env, 'box01', edge_x + gap + half_t, 0.0, 0.20, yaw=np.pi / 2)
        r = env.laser_front.read()
        if expect_hit:
            ok &= check("间距 {:.0f}cm: 命中且测距 {:.4f}m ≈ {:.2f}m".format(
                gap * 100, r.distance, gap),
                r.hit and abs(r.distance - gap) < 0.005)
        else:
            ok &= check("间距 {:.0f}cm: 超量程无回波 (hit={})".format(gap * 100, r.hit),
                        not r.hit and r.distance == LASER_MAX_RANGE)

    # ---------- 特性2：检测高度 ----------
    print("== 检测高度（前向激光，障碍固定在 5cm 处）==")
    gap = 0.05
    cases = [
        (0.020, "2cm 低矮物（盲区）",   lambda r: (not r.hit) and r.confidence == 0.0),
        (0.030, "3cm 低矮物（盲区）",   lambda r: (not r.hit) and r.confidence == 0.0),
        (0.050, "5cm 物体（低置信）",   lambda r: r.hit and 0.0 < r.confidence < 1.0),
        # 7cm 恰好与最高射线共面（边界），允许最高一条打在棱上丢失
        (0.070, "7cm 物体（边界可靠）", lambda r: r.hit and r.confidence >= 0.75),
        (0.080, "8cm 物体（可靠）",     lambda r: r.hit and r.confidence == 1.0),
        (0.200, "20cm 墙面（可靠）",    lambda r: r.hit and r.confidence == 1.0),
    ]
    for h, desc, pred in cases:
        place_box(env, 'box01', edge_x + gap + half_t, 0.0, h, yaw=np.pi / 2)
        r = env.laser_front.read()
        ok &= check("{}: hit={} conf={:.2f} dist={:.3f}".format(
            desc, r.hit, r.confidence, r.distance), pred(r))

    # ---------- 右侧激光（与真机一致装在右侧）----------
    print("== 右侧激光（右侧 2cm 处 20cm 高墙）==")
    env.task._bury_all_boxes()
    mujoco.mj_forward(env.model, env.data)
    gap = 0.02
    edge_y = 0.0 - CHASSIS_RADIUS
    place_box(env, 'box02', 1.0, edge_y - gap - half_t, 0.20)
    r = env.laser_right.read()
    ok &= check("右侧测距 {:.4f}m ≈ {:.2f}m, conf={:.2f}".format(r.distance, gap, r.confidence),
                r.hit and abs(r.distance - gap) < 0.005 and r.confidence == 1.0)

    print("\n全部通过" if ok else "\n存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
