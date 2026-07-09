"""扫地机器人"过坎 + 脱困"任务。

设计目标：让差速底盘从起点驶向前方目标点，途中需要：
  1) 过坎：驶过不同高度的门槛 / 地板接缝（CROSS 模式）；
  2) 脱困：从"骑坎打滑卡住"（HIGHCENTER）和"顶住高障碍"（AGAINST）等
     被困状态中通过前后/差速摆动脱身，恢复正常行驶。

课程学习：随训练迭代（iteration_count）逐步抬高门槛、加大被困难度。

奖励组成（calc_reward 返回 dict，env 求和）：
  progress   : 朝目标方向的实际位移速度（核心驱动）
  goal_dist  : 到目标距离的势能整形
  cross      : 越过门槛后的持续奖励（仅当有门槛时）
  upright    : 保持机身水平、不翻车（横滚惩罚 > 俯仰）
  heading    : 朝向目标（脱困需要倒车，故权重较低）
  stuck      : 卡住惩罚（命令大力矩但几乎不动）
  energy     : 力矩能耗惩罚
  smooth     : 动作平滑惩罚
  success    : 到达目标的一次性大奖励

终止：翻车 / 机身被弹飞 / 出界 / 到达目标。
"""

import numpy as np
import transforms3d as tf3
from enum import Enum, auto


class VacuumModes(Enum):
    FLAT = auto()        # 平地行驶（基线）
    CROSS = auto()       # 过坎：前方一道门槛
    HIGHCENTER = auto()  # 脱困：骑在凸脊上，轮子减载打滑
    AGAINST = auto()     # 脱困：顶住一段过高障碍，需倒车绕行


class VacuumTask(object):
    # ---- 任务超参数 ----
    GOAL_X = 1.2          # 目标点（世界系，机器人朝 +x 出发）
    V_DES = 0.30          # 期望行进速度 (m/s)
    TARGET_RADIUS = 0.15  # 判定到达目标的半径
    MAX_TAU = 2.0         # 与 XML/robot 的力矩上限保持一致（用于归一化）

    def __init__(self,
                 client=None,
                 dt=0.025,
                 root_body='base',
                 chassis_geom='chassis_geom'):
        self._client = client
        self._control_dt = dt
        self._root_body_name = root_body
        self._chassis_geom_name = chassis_geom

        self.iteration_count = np.inf

        # 由 reset 填充
        self.mode = VacuumModes.FLAT
        self.goal_pos = np.array([self.GOAL_X, 0.0])
        self.thr_pos = np.array([10.0, 0.0, -1.0])  # 主障碍中心（世界系），默认埋地下
        self.thr_height = 0.0
        self.thr_yaw = 0.0
        self.robot_init = (0.0, 0.0, 0.05, 0.0)     # (x, y, z, yaw) 初始位姿

        # 运行时状态
        self.prev_xy = np.zeros(2)
        self.contact_belly = 0.0
        self.contact_lwheel = 0.0
        self.contact_rwheel = 0.0
        self.stuck_frames = 0
        self.is_stuck = 0.0
        self._reached_goal = False
        self._reached_now = 0.0

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _base_xy(self):
        return self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2].copy()

    def _base_pose(self):
        qpos = self._client.get_qpos()
        pos = qpos[0:3].copy()
        quat = qpos[3:7].copy()
        roll, pitch, yaw = tf3.euler.quat2euler(quat)
        R = tf3.quaternions.quat2mat(quat)
        up_z = R[2, 2]
        return pos, (roll, pitch, yaw), up_z

    def _chassis_contact(self):
        """底盘（腹部）是否与任何物体接触 —— 骑坎/卡住的关键信号。"""
        model = self._client.model
        data = self._client.data
        gid = model.geom(self._chassis_geom_name).id
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return 1.0
        return 0.0

    def _to_body_frame(self, world_xy, base_xy, yaw):
        d = np.asarray(world_xy) - np.asarray(base_xy)
        fwd = d[0] * np.cos(yaw) + d[1] * np.sin(yaw)
        left = -d[0] * np.sin(yaw) + d[1] * np.cos(yaw)
        return fwd, left

    # ------------------------------------------------------------------
    # 给 env.get_obs 用的外部状态（目标 + 地形 + 接触），共 9 维
    # ------------------------------------------------------------------
    def get_ext_state(self):
        pos, (roll, pitch, yaw), up_z = self._base_pose()
        base_xy = pos[0:2]
        goal_fwd, goal_left = self._to_body_frame(self.goal_pos, base_xy, yaw)
        thr_fwd, thr_left = self._to_body_frame(self.thr_pos[0:2], base_xy, yaw)
        thr_yaw_rel = self.thr_yaw - yaw
        ext = np.array([
            goal_fwd, goal_left,                 # 目标在机体系的前/左方向
            thr_fwd, thr_left, self.thr_height,  # 主障碍位置 + 高度
            pos[2],                              # 机身离地高度
            self.contact_belly,                  # 腹部接触（骑坎）
            self.contact_lwheel, self.contact_rwheel,  # 左右轮接地
        ])
        return ext

    # ------------------------------------------------------------------
    # 每个控制步推进任务状态
    # ------------------------------------------------------------------
    def step(self):
        # 接触状态
        self.contact_belly = self._chassis_contact()
        self.contact_lwheel = 1.0 if len(self._client.get_lfoot_floor_contacts()) > 0 else 0.0
        self.contact_rwheel = 1.0 if len(self._client.get_rfoot_floor_contacts()) > 0 else 0.0

        # 基于位移的世界系速度（避免 free joint qvel 坐标系歧义）
        cur_xy = self._base_xy()
        self.world_vel_xy = (cur_xy - self.prev_xy) / self._control_dt
        self.prev_xy = cur_xy

        # 到达目标判定
        dist = np.linalg.norm(self.goal_pos - cur_xy)
        self._reached_now = 0.0
        if (not self._reached_goal) and dist < self.TARGET_RADIUS:
            self._reached_goal = True
            self._reached_now = 1.0
        return

    def substep(self):
        pass

    # ------------------------------------------------------------------
    # 奖励
    # ------------------------------------------------------------------
    def calc_reward(self, prev_torque, prev_action, action):
        pos, (roll, pitch, yaw), up_z = self._base_pose()
        base_xy = pos[0:2]

        goal_vec = self.goal_pos - base_xy
        dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (dist + 1e-6)

        # 1) 朝目标的位移速度
        v_toward = float(np.dot(self.world_vel_xy, goal_dir))
        progress = np.clip(v_toward / self.V_DES, -1.0, 1.0)

        # 2) 距离势能整形
        goal_dist_r = np.exp(-dist / 0.8)

        # 3) 越坎奖励：越过门槛 x 后持续给奖（仅有门槛时）
        cross_r = 0.0
        if self.thr_height > 1e-3:
            cross_r = 1.0 if base_xy[0] > self.thr_pos[0] else 0.0

        # 4) 保持水平（横滚惩罚重于俯仰，过坎时允许一定俯仰）
        upright_r = np.exp(-(4.0 * roll * roll + 1.5 * pitch * pitch))

        # 5) 朝向目标（脱困需倒车，权重低）
        heading_err = np.arctan2(goal_dir[1], goal_dir[0]) - yaw
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        heading_r = np.exp(-(heading_err * heading_err) / 0.5)

        # 6) 卡住检测：命令大力矩但几乎不动
        cmd_mag = np.mean(np.abs(prev_torque)) / self.MAX_TAU
        speed = np.linalg.norm(self.world_vel_xy)
        if cmd_mag > 0.5 and speed < 0.02:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
        self.is_stuck = 1.0 if self.stuck_frames >= 4 else 0.0

        # 7) 能耗 / 平滑
        energy = np.mean(np.square(prev_torque)) / (self.MAX_TAU ** 2)
        dact = action - (prev_action if prev_action is not None else action)
        smooth = np.mean(np.square(dact))

        reward = dict(
            progress=0.30 * progress,
            goal_dist=0.15 * goal_dist_r,
            cross=0.15 * cross_r,
            upright=0.12 * upright_r,
            heading=0.06 * heading_r,
            stuck=0.10 * (-self.is_stuck),
            energy=0.04 * (-energy),
            smooth=0.03 * (-smooth),
            success=5.0 * self._reached_now,
        )
        return reward

    # ------------------------------------------------------------------
    # 终止条件
    # ------------------------------------------------------------------
    def done(self):
        pos, (roll, pitch, yaw), up_z = self._base_pose()
        conditions = {
            "flipped": (up_z < 0.4) or (abs(roll) > 1.0) or (abs(pitch) > 1.2),
            "launched": (pos[2] > 0.45) or (pos[2] < -0.05),
            "out_of_bounds": (abs(pos[0]) > 5.0) or (abs(pos[1]) > 5.0),
            "reached_goal": self._reached_goal,
        }
        return True in conditions.values()

    # ------------------------------------------------------------------
    # 重置：选模式、布地形、定初始位姿
    # ------------------------------------------------------------------
    def reset(self, iter_count=0):
        self.iteration_count = iter_count
        frac = np.clip(iter_count / 4000.0, 0.0, 1.0)  # 课程进度 0->1

        self.stuck_frames = 0
        self.is_stuck = 0.0
        self._reached_goal = False
        self._reached_now = 0.0

        # 选择模式（脱困场景占比随训练略增由难度体现，这里用固定概率 + 难度课程）
        self.mode = np.random.choice(
            [VacuumModes.FLAT, VacuumModes.CROSS, VacuumModes.HIGHCENTER, VacuumModes.AGAINST],
            p=[0.15, 0.45, 0.25, 0.15])

        # 默认：目标在前方，障碍埋地下
        self.goal_pos = np.array([self.GOAL_X, 0.0])
        self.thr_pos = np.array([10.0, 0.0, -1.0])
        self.thr_height = 0.0
        self.thr_yaw = 0.0
        start_yaw = np.random.uniform(-0.10, 0.10)

        self._bury_all_boxes()

        if self.mode == VacuumModes.FLAT:
            self.robot_init = (np.random.uniform(-0.05, 0.05),
                               np.random.uniform(-0.05, 0.05), 0.05, start_yaw)

        elif self.mode == VacuumModes.CROSS:
            h = np.random.uniform(0.010, 0.020 + 0.030 * frac)  # 1cm -> 5cm
            thr_x = np.random.uniform(0.45, 0.65)
            self._place_ridge('ridge01', thr_x, height=h)
            self.thr_pos = np.array([thr_x, 0.0, h])
            self.thr_height = h
            self.robot_init = (np.random.uniform(-0.05, 0.05),
                               np.random.uniform(-0.05, 0.05), 0.05, start_yaw)

        elif self.mode == VacuumModes.HIGHCENTER:
            # 凸脊高度略高于离地间隙 (1.5cm)，使腹部压脊、轮子减载打滑。
            # 窄脊(宽10cm)沿行进方向、横向偏置 ±8cm：从中线万向球与驱动轮
            # 之间的空隙下方穿过 —— 腹部骑在脊上、机身微倾、单侧轮打滑，
            # 但另一侧轮仍有抓地，摆动/转向可以脱困
            h = np.random.uniform(0.018, 0.018 + 0.012 * frac)  # ~1.8cm -> 3cm
            side = 1.0 if np.random.rand() < 0.5 else -1.0
            ridge_y = side * 0.08
            self._place_ridge('ridge03', 0.0, height=h, y=ridge_y)
            self.thr_pos = np.array([0.0, ridge_y, h])
            self.thr_height = h
            # 机器人骑在脊上（略抬高，mj_forward/前几步内沉降到位）
            self.robot_init = (np.random.uniform(-0.03, 0.03),
                               np.random.uniform(-0.015, 0.015),
                               0.05 + h, start_yaw)

        elif self.mode == VacuumModes.AGAINST:
            # 顶住一段过高的台阶（窄于通道，可绕行），机器人需倒车 + 转向
            wall_x = np.random.uniform(0.30, 0.40)
            self._place_ridge('ridge02', wall_x, height=0.07)
            self.thr_pos = np.array([wall_x, 0.0, 0.07])
            self.thr_height = 0.07
            self.robot_init = (np.random.uniform(-0.05, 0.0),
                               np.random.uniform(-0.03, 0.03), 0.05, start_yaw)

        # 初始化位移速度基准
        self.prev_xy = np.array(self.robot_init[0:2])
        return

    # ------------------------------------------------------------------
    # 地形操作。门槛/短墙是 mocap body（ridge01/ridge02）：
    # 静态 world geom 的运行时 pos/size 修改对碰撞检测不生效（编译期烘焙），
    # 必须用 mocap 摆放；geom 尺寸固定，"不同高度"靠下沉入地实现。
    # ------------------------------------------------------------------
    def _bury_all_boxes(self):
        model = self._client.model
        data = self._client.data
        for b in range(model.nbody):
            mid = model.body_mocapid[b]
            if mid >= 0:
                data.mocap_pos[mid] = np.array([0.0, 0.0, -2.0])
                data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])

    def _place_ridge(self, name, x, height, y=0.0, yaw=0.0):
        model = self._client.model
        data = self._client.data
        half_z = float(model.geom(name).size[2])
        mid = model.body_mocapid[model.body(name).id]
        data.mocap_pos[mid] = np.array([x, y, height - half_z])  # 下沉调有效高度
        data.mocap_quat[mid] = np.array([np.cos(yaw / 2.0), 0.0, 0.0,
                                         np.sin(yaw / 2.0)])
