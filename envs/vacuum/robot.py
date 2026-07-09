import numpy as np


class VacuumRobot:
    """差速驱动扫地机器人。

    与人形 JVRC 类不同：人形用"关节位置 PD（动作=名义姿态偏移）"控制；
    扫地机器人是轮式底盘，动作 = 两个驱动轮的力矩，直接施加到电机执行器，
    更贴近真实电机，过坎/脱困时的打滑、堵转行为更真实。

    动作约定：
      action ∈ R^2，分别对应 (左轮, 右轮)。
      经 torque_scale 缩放并裁剪到执行器 ctrlrange 后作为电机力矩。

    边刷：模型中的 sbrush_l / sbrush_r 两个 hinge 关节由这里以恒定转速
    运动学驱动（每个仿真子步直接写 qvel），清扫时持续旋转、向机身中线
    扫拢（左刷顺时针、右刷逆时针，俯视）。不占用执行器，不影响动作维度。
    """

    BRUSH_SPEED = 12.0   # 边刷转速 rad/s（约 115 rpm）

    def __init__(self, pdgains, dt, active, client, torque_scale=2.0):
        self.client = client
        self.control_dt = dt
        self.actuators = active            # [0, 1] -> 左右轮电机
        self.torque_scale = torque_scale

        # 执行器力矩上限（来自 XML ctrlrange）
        low, high = self.client.get_actuator_ctrl_range()
        self.tau_low = np.asarray(low)
        self.tau_high = np.asarray(high)

        # 初始 qpos / qvel（base free joint 7 + 2 轮）
        self.init_qpos_ = [0] * self.client.nq()
        self.init_qvel_ = [0] * self.client.nv()
        # base 初始位姿：原点、单位四元数，高度由环境在 reset 时覆盖
        self.init_qpos_[0:7] = [0, 0, 0.05, 1, 0, 0, 0]

        self.prev_action = None            # 最近一次动作（缩放前）。step 结束后 = 本步动作，供观测用
        self.prev_torque = None            # 最近一次实际施加力矩。step 结束后 = 本步力矩，供能耗项用
        # 真正"上一控制步"的动作/力矩：在 step() 覆盖 prev_* 之前抓取。
        # 历史 bug：奖励里的平滑项直接用 prev_action，而 env.step 是先 robot.step
        # （已把 prev_action 覆盖成本步动作）再 calc_reward，导致 dact = a - a ≡ 0，
        # 平滑惩罚恒为 0 —— 沿墙"机身一直摇晃"的直接原因之一。
        self.last_action = None
        self.last_torque = None
        self.iteration_count = np.inf      # 由 PPO 注入，用于课程学习

        # 边刷关节（旧模型没有这两个关节时优雅跳过）
        self._brush_dofs = []
        for name, sign in (("sbrush_l", -1.0), ("sbrush_r", 1.0)):
            try:
                dof = int(self.client.model.joint(name).dofadr[0])
                self._brush_dofs.append((dof, sign))
            except Exception:
                pass

        # frame skip
        if (np.around(self.control_dt % self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt / self.client.sim_dt())

        self.num_actions = len(self.actuators)

    def step(self, action):
        action = np.asarray(action).flatten()

        # 缩放为力矩并裁剪到执行器范围
        torque = np.clip(action * self.torque_scale, self.tau_low, self.tau_high)

        # 先留存上一控制步的动作/力矩（回合首步没有上一步，用本步代替 -> dact=0）
        self.last_action = action.copy() if self.prev_action is None else self.prev_action.copy()
        self.last_torque = torque.copy() if self.prev_torque is None else self.prev_torque.copy()

        self.do_simulation(torque, self.frame_skip)

        self.prev_action = action.copy()
        self.prev_torque = torque.copy()
        return torque

    def do_simulation(self, torque, n_frames):
        for _ in range(n_frames):
            # 边刷恒速旋转（运动学驱动，直接写关节速度）
            for dof, sign in self._brush_dofs:
                self.client.data.qvel[dof] = sign * self.BRUSH_SPEED
            self.client.set_motor_torque(torque)
            self.client.step()
