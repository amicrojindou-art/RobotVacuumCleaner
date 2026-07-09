import os
import tempfile
import numpy as np
import transforms3d as tf3

from tasks import vacuum_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.vacuum import robot as vacuum_robot
from envs.vacuum.sensors import LineLaser
from .gen_xml import builder, MAX_WHEEL_TORQUE


class VacuumEnv(mujoco_env.MujocoEnv):
    """扫地机器人过坎/脱困仿真环境（差速圆盘底盘，轮力矩控制）。

    与人形 env 的对应关系：
      - get_obs(): 机体状态(14) + 任务外部状态(9) = 23 维观测
      - step():    robot.step(力矩) -> task.step() -> calc_reward -> done
      - reset_model(): 由 task.reset() 选模式/布地形/定初始位姿，再写入仿真
    """

    def __init__(self):
        sim_dt = 0.0025
        control_dt = 0.025

        # 生成并加载模型（跨平台临时目录）。每次都重建，保证 gen_xml.py 的
        # 修改立即生效，避免加载到过期的缓存 XML。
        path_to_xml_out = os.path.join(tempfile.gettempdir(), 'mjcf-export', 'vacuum', 'vacuum.xml')
        builder(path_to_xml_out)
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        # 两个驱动轮电机
        self.actuators = [0, 1]

        # 机器人接口：把左右轮当作"左右脚"，从而复用 *foot* 接触/GRF 接口
        self.interface = robot_interface.RobotInterface(
            self.model, self.data,
            rfoot_body_name='right_wheel', lfoot_body_name='left_wheel')

        # 线激光传感器（沿边沿墙用）：正前方 + 左侧各一个。
        # 仅作为传感器接口暴露，不进入 RL 观测（观测维度保持 23，兼容已训练模型）。
        self.laser_front = LineLaser(self.model, self.data, 'front')
        self.laser_left = LineLaser(self.model, self.data, 'left')

        # 任务
        self.task = vacuum_task.VacuumTask(
            client=self.interface,
            dt=control_dt,
            root_body='base',
            chassis_geom='chassis_geom',
        )

        # 机器人（pdgains 这里不用，传占位）
        self.robot = vacuum_robot.VacuumRobot(
            pdgains=None, dt=control_dt, active=self.actuators,
            client=self.interface, torque_scale=MAX_WHEEL_TORQUE)

        # 动作空间：2 个轮力矩
        self.action_space = np.zeros(len(self.actuators))

        # 观测空间：机体 14 + 外部 9 = 23
        self.robot_state_len = 14
        self.ext_state_len = 9
        self.base_obs_len = self.robot_state_len + self.ext_state_len
        self.observation_space = np.zeros(self.base_obs_len)

        # 不使用左右镜像（run_experiment 会因缺少 mirrored_obs 优雅回退）

        self.reset_model()

    # ------------------------------------------------------------------
    def get_obs(self):
        qpos = np.copy(self.interface.get_qpos())

        # 机身朝向：保留横滚/俯仰，偏航置零（与人形一致，避免学到绝对朝向）
        roll, pitch = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(roll, pitch, 0)

        # 机体系线速度 / 角速度（mj_objectVelocity, frame=1 局部坐标）
        lin_local, ang_local = self.interface.get_body_vel('base', frame=1)

        # 驱动轮转速
        wheel_vel = self.interface.get_act_joint_velocities()
        wheel_vel = [wheel_vel[i] for i in self.actuators]

        # 上一步动作（缩放前）
        prev_action = self.robot.prev_action
        if prev_action is None:
            prev_action = np.zeros(len(self.actuators))

        robot_state = np.concatenate([
            root_orient,        # 4
            ang_local,          # 3
            lin_local,          # 3
            wheel_vel,          # 2
            prev_action,        # 2
        ])

        ext_state = self.task.get_ext_state()  # 9

        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.base_obs_len,), \
            "obs dim {} != {}".format(state.shape, self.base_obs_len)
        return state.flatten()

    # ------------------------------------------------------------------
    def step(self, a):
        applied_action = self.robot.step(a)

        self.task.step()
        # last_action = 真正上一步的动作（平滑项）。prev_action 在 robot.step 里
        # 已被覆盖成本步动作，用它会让 dact 恒为 0（平滑惩罚失效）。
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.last_action, a)
        total_reward = sum([float(i) for i in rewards.values()])

        done = self.task.done()
        obs = self.get_obs()
        return obs, total_reward, done, rewards

    # ------------------------------------------------------------------
    def reset_model(self):
        # 先让任务决定模式 / 地形 / 初始位姿
        self.task.reset(iter_count=self.robot.iteration_count)

        # 基础初始状态 + 轻微噪声
        c = 0.01
        init_qpos = np.array(self.robot.init_qpos_, dtype=np.float64)
        init_qvel = np.array(self.robot.init_qvel_, dtype=np.float64)
        init_qpos += np.random.uniform(low=-c, high=c, size=self.model.nq)
        init_qvel += np.random.uniform(low=-c, high=c, size=self.model.nv)

        # 写入任务给定的初始位姿
        x, y, z, yaw = self.task.robot_init
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        init_qpos[root_adr + 0] = x
        init_qpos[root_adr + 1] = y
        init_qpos[root_adr + 2] = z
        init_qpos[root_adr + 3:root_adr + 7] = tf3.euler.euler2quat(0, 0, yaw)

        # 复位上一步缓存
        self.robot.prev_action = None
        self.robot.prev_torque = None
        self.robot.last_action = None
        self.robot.last_torque = None

        self.set_state(init_qpos, init_qvel)

        # 用真实落点刷新任务的位移基准
        self.task.prev_xy = self.interface.get_object_xpos_by_name('base', 'OBJ_BODY')[0:2].copy()

        return self.get_obs()
