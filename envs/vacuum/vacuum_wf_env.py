"""扫地机器人"沿边沿墙清扫"RL 环境。

与 VacuumEnv（过坎/脱困）的区别：
  - 任务换成 WallFollowTask：每个 episode 随机生成折线墙课程；
  - 观测不含任何全局/特权信息（无目标点、无障碍真值坐标），只有
    两个线激光的读数 + 本体感知，policy 因此可泛化到任意户型；
  - XML 默认不带家居场景（空场地 + 随机墙）；home_scene=True 生成带
    家居场景的版本，用于回放验证泛化性。

观测（34 维）：
  机体 14：root 四元数(yaw 置零) 4 + 机体角速度 3 + 机体线速度 3 +
           轮速 2 + 上一步动作 2
  传感  8：前向距离/0.1、前向置信度（纯激光读数，碰撞不注入任何回波——
           历史上三次把碰撞翻译进该通道都摧毁了阳角包边，见任务文件头
           "重要教训"）、右侧距离/0.1、右侧置信度、
           右侧距离变化率(滤波)、机身接触方位(+1左/-1右/0无，真机保险杠
           左右分区)、左轮接地、右轮接地
  历史 12：激光 4 元组（前距/前置信/右距/右置信）在 0.05s/0.1s/0.2s
           前的快照。拐角处是部分可观测问题（激光会短暂丢失目标），
           前馈网络需要短时历史来分辨"刚丢墙"和"一直没墙"。

侧边激光与真机一致装在【右侧】（沿墙时墙在机器人右侧、绕房间逆时针巡边）。
"""

import os
import tempfile
import numpy as np
import transforms3d as tf3

from tasks import wall_follow_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.vacuum import robot as vacuum_robot
from envs.vacuum.sensors import LineLaser
from .gen_xml import builder, MAX_WHEEL_TORQUE, LASER_MAX_RANGE, NUM_BOXES


class VacuumWFEnv(mujoco_env.MujocoEnv):

    def __init__(self, home_scene=False):
        sim_dt = 0.0025
        control_dt = 0.025
        self.home_scene = home_scene

        # 生成并加载模型。文件名带 pid 与场景变体，避免 ray 多进程并发写同一文件
        variant = 'home' if home_scene else 'rand'
        path_to_xml_out = os.path.join(
            tempfile.gettempdir(), 'mjcf-export', 'vacuum_wf',
            'vacuum_wf_{}_{}.xml'.format(variant, os.getpid()))
        builder(path_to_xml_out, scene=home_scene)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        self.actuators = [0, 1]

        self.interface = robot_interface.RobotInterface(
            self.model, self.data,
            rfoot_body_name='right_wheel', lfoot_body_name='left_wheel')

        # 线激光（侧边激光与真机一致装在右侧）
        self.laser_front = LineLaser(self.model, self.data, 'front')
        self.laser_right = LineLaser(self.model, self.data, 'right')

        # 训练墙/门槛（mocap 地形）接触加硬：默认软接触在轮子大力矩顶墙时
        # 会深度穿透，策略会学到"顶进墙里"这类不物理的行为
        terrain = ['box' + repr(i + 1).zfill(2) for i in range(NUM_BOXES)]
        terrain += ['ridge01', 'ridge02', 'ridge03']
        for name in terrain:
            gid = self.model.geom(name).id
            self.model.geom_solref[gid][0:2] = np.array([0.005, 1.0])
            self.model.geom_solimp[gid][0:3] = np.array([0.95, 0.99, 0.001])

        # 任务
        self.task = wall_follow_task.WallFollowTask(
            client=self.interface,
            dt=control_dt,
            root_body='base',
            chassis_geom='chassis_geom',
        )
        self.task.home_scene = home_scene
        self.task.laser_front = self.laser_front
        self.task.laser_right = self.laser_right

        self.robot = vacuum_robot.VacuumRobot(
            pdgains=None, dt=control_dt, active=self.actuators,
            client=self.interface, torque_scale=MAX_WHEEL_TORQUE)

        self.action_space = np.zeros(len(self.actuators))

        # 激光历史帧滞后（控制步数）：0.05s / 0.1s / 0.2s
        self.hist_lags = (2, 4, 8)

        self.robot_state_len = 14
        self.ext_state_len = 8 + 4 * len(self.hist_lags)
        self.base_obs_len = self.robot_state_len + self.ext_state_len
        self.observation_space = np.zeros(self.base_obs_len)

        # 右侧距离变化率（低通滤波），给 policy 提供阻尼信息
        self._prev_side_d = None
        self._dside_f = 0.0
        self._laser_hist = []

        self.reset_model()

    # ------------------------------------------------------------------
    def get_obs(self):
        qpos = np.copy(self.interface.get_qpos())
        roll, pitch = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(roll, pitch, 0)
        lin_local, ang_local = self.interface.get_body_vel('base', frame=1)

        wheel_vel = self.interface.get_act_joint_velocities()
        wheel_vel = [wheel_vel[i] for i in self.actuators]

        prev_action = self.robot.prev_action
        if prev_action is None:
            prev_action = np.zeros(len(self.actuators))
        # 限幅到 ±1：|action|>1 后力矩已被 ctrlrange 饱和（torque_scale=2、
        # ctrlrange ±2），更大的动作值对动力学无意义；不限幅时越界动作会
        # 通过观测正反馈自激（家居回放实测动作跑飞到 ±20，观测严重出分布，
        # 策略输出全频抖动）。
        prev_action = np.clip(prev_action, -1.0, 1.0)

        robot_state = np.concatenate([
            root_orient,        # 4
            ang_local,          # 3
            lin_local,          # 3
            wheel_vel,          # 2
            prev_action,        # 2
        ])

        front = self.laser_front.read()
        side = self.laser_right.read()
        # 激光通道保持纯净：碰撞不注入任何观测。历史教训（0714~0715 三次
        # 重训失败）：把碰撞翻译成前向回波会摧毁阳角包边 —— 紧贴包边必然
        # 偶尔蹭角，蹭角接触方位角随右转迁移进任何触发锥，"正前有墙"触发
        # 左转反射把机器人推离墙。碰撞只走接触方位观测 + 奖励项。
        front_d, front_c = front.distance, front.confidence

        # 右侧距离变化率（限幅 + 低通）
        if self._prev_side_d is None:
            dside = 0.0
        else:
            dside = np.clip((side.distance - self._prev_side_d) / self.robot.control_dt,
                            -0.3, 0.3)
        self._prev_side_d = side.distance
        self._dside_f += 0.35 * (dside - self._dside_f)

        laser_now = np.array([
            front_d / LASER_MAX_RANGE,
            front_c,
            side.distance / LASER_MAX_RANGE,
            side.confidence,
        ])

        # 激光历史帧（不足时用最早一帧补齐）
        self._laser_hist.append(laser_now)
        if len(self._laser_hist) > max(self.hist_lags) + 1:
            self._laser_hist.pop(0)
        hist = []
        for lag in self.hist_lags:
            idx = max(0, len(self._laser_hist) - 1 - lag)
            hist.append(self._laser_hist[idx])

        ext_state = np.concatenate([
            laser_now,
            np.array([
                self._dside_f / 0.3,
                self.task.contact_dir,   # 机身接触方位 +1左/-1右/0无
                self.task.contact_lwheel,
                self.task.contact_rwheel,
            ]),
        ] + hist)

        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.base_obs_len,), \
            "obs dim {} != {}".format(state.shape, self.base_obs_len)
        return state.flatten()

    # ------------------------------------------------------------------
    def step(self, a):
        self.robot.step(a)
        self.task.step()
        # prev_torque = 本步实际施加的力矩（能耗项）；last_action = 真正上一步的
        # 动作（平滑项）。不能用 prev_action —— robot.step 已把它覆盖成本步动作。
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.last_action, a)
        total_reward = sum([float(i) for i in rewards.values()])
        done = self.task.done()
        obs = self.get_obs()
        return obs, total_reward, done, rewards

    # ------------------------------------------------------------------
    def reset_model(self):
        self.task.reset(iter_count=self.robot.iteration_count)

        # 激光噪声由任务课程决定
        self.laser_front.noise_std = self.task.sensor_noise
        self.laser_right.noise_std = self.task.sensor_noise

        c = 0.01
        init_qpos = np.array(self.robot.init_qpos_, dtype=np.float64)
        init_qvel = np.array(self.robot.init_qvel_, dtype=np.float64)
        init_qpos += np.random.uniform(low=-c, high=c, size=self.model.nq)
        init_qvel += np.random.uniform(low=-c, high=c, size=self.model.nv)

        x, y, z, yaw = self.task.robot_init
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        init_qpos[root_adr + 0] = x
        init_qpos[root_adr + 1] = y
        init_qpos[root_adr + 2] = z
        init_qpos[root_adr + 3:root_adr + 7] = tf3.euler.euler2quat(0, 0, yaw)

        self.robot.prev_action = None
        self.robot.prev_torque = None
        self.robot.last_action = None
        self.robot.last_torque = None
        self._prev_side_d = None
        self._dside_f = 0.0
        self._laser_hist = []

        self.set_state(init_qpos, init_qvel)
        self.task.prev_xy = self.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()
        return self.get_obs()
