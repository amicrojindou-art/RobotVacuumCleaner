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


class VacuumVelocityRobot(VacuumRobot):
    """方案B：动作 = 左右轮速目标系数 ∈ [-1,1]，×V_MAX_WHEEL 得轮速目标 (rad/s)。

    真机大核对小核只有【轮速目标】通道（g_vels_shm：左右轮 mm/s，小核做电机
    速度闭环），没有力矩/PWM 接口。让训练动作语义与真机接口天然一致：动作 =
    轮速目标，仿真内嵌一个模拟小核电机速度环的低层 P 伺服 + 斜率限幅。
    sim2real 差距从"整条力矩动力学链"缩小为"速度环响应差异"，后者用域随机化覆盖。

    与真机几何绑定（2026-07-21 对齐）：
      V_MAX_WHEEL = 真机 300mm/s ÷ 轮半径 0.037250m ≈ 8.05 rad/s
      KP 稳定界   = (m/2)·r²/sim_dt = (2.9/2)·0.03725²/0.0025 ≈ 0.805，KP=0.5 安全
    """

    V_MAX_WHEEL = 8.05      # rad/s；×轮半径0.037250 = 0.30m/s = 真机 MAX_SPEED 300mm/s
    KP_SERVO    = 0.5       # N·m/(rad/s)，模拟小核速度环 P 增益（名义值）
    SLEW_RATE   = 40.0      # rad/s² 轮速目标斜率限幅（≈真机 g_vels_shm accels 语义）

    # 域随机化范围（reset 时采样，覆盖真机速度环不确定性）
    KP_RAND     = (0.6, 1.4)   # 伺服增益缩放；上限按稳定界 clip
    KP_CAP      = 0.7          # kp_eff 硬上限（< 稳定界 0.805）
    VMAX_RAND   = (0.95, 1.05) # 轮径磨损/打滑标定误差
    DELAY_MAX   = 2            # 指令延迟最大控制步数（SPI-RPC 下发+小核执行，实测10~30ms）
    WVEL_NOISE  = 0.1          # 轮速测量噪声 σ (rad/s)，编码器量化

    def __init__(self, pdgains, dt, active, client, torque_scale=2.0):
        super().__init__(pdgains, dt, active, client, torque_scale)
        # 伺服 / 域随机化运行时状态
        self._kp_eff = self.KP_SERVO
        self._vmax_eff = self.V_MAX_WHEEL
        self._delay = 0
        self._wvel_noise = 0.0       # 本回合轮速噪声 σ（随课程展开）
        self._cmd_fifo = []          # 指令延迟 FIFO（存轮速目标 rad/s）
        self._w_set_prev = np.zeros(len(active))  # 斜率限幅用（上一步已发目标）

    def randomize(self, frac=1.0):
        """reset 时由 env 调用，采样本回合的伺服/延迟/量程随机化。

        域随机化【课程】(2026-07-21 修复)：幅度随 frac 0->1 渐进展开。frac=0 时
        全部取名义值（无随机化），让策略先在干净任务上学会基础沿边；frac=1 时
        才拉满不确定性。与 task 的 sensor_noise/friction 课程同理。上一版
        (vacuum_wf_vel0721) 从第0轮就满随机化，策略被噪声淹没学不出稳定映射、
        ep_len 从514跌到310 —— 本修复的直接对象。
        """
        f = float(np.clip(frac, 0.0, 1.0))
        # kp/vmax 范围从 [1,1] 渐开到全范围
        kp_lo = 1.0 - (1.0 - self.KP_RAND[0]) * f
        kp_hi = 1.0 + (self.KP_RAND[1] - 1.0) * f
        self._kp_eff = float(np.clip(self.KP_SERVO * np.random.uniform(kp_lo, kp_hi),
                                     0.0, self.KP_CAP))
        vmax_lo = 1.0 - (1.0 - self.VMAX_RAND[0]) * f
        vmax_hi = 1.0 + (self.VMAX_RAND[1] - 1.0) * f
        self._vmax_eff = self.V_MAX_WHEEL * float(np.random.uniform(vmax_lo, vmax_hi))
        # 延迟上限、轮速噪声随 frac 增长
        self._delay = int(np.random.randint(0, int(round(self.DELAY_MAX * f)) + 1))
        self._wvel_noise = self.WVEL_NOISE * f
        self._cmd_fifo = []
        self._w_set_prev = np.zeros(len(self.actuators))

    def step(self, action):
        # 1) 动作 clip 到 ±1（Gaussian 采样会越界；速度版必须显式 clip，
        #    否则 V_MAX 失去意义。力矩版靠 ctrlrange 天然限幅，这里没有）
        action = np.clip(np.asarray(action).flatten(), -1.0, 1.0)
        w_set_raw = action * self._vmax_eff

        # 2) 斜率限幅（真机小核有加速度限制；训练内置使策略学会平滑指令）
        dmax = self.SLEW_RATE * self.control_dt
        w_set = np.clip(w_set_raw, self._w_set_prev - dmax, self._w_set_prev + dmax)
        self._w_set_prev = w_set.copy()

        # 3) 指令延迟（FIFO）：本步目标入队，取延迟 self._delay 步前的目标执行
        self._cmd_fifo.append(w_set.copy())
        w_active = self._cmd_fifo[0] if len(self._cmd_fifo) > self._delay \
            else np.zeros(len(self.actuators))
        if len(self._cmd_fifo) > self._delay:
            self._cmd_fifo.pop(0)

        # 4) 记账（与基类一致）：last_* = 真正上一控制步，供奖励平滑/能耗项
        self.last_action = action.copy() if self.prev_action is None else self.prev_action.copy()
        self.last_torque = (self.prev_torque.copy() if self.prev_torque is not None
                            else np.zeros(len(self.actuators)))

        # 5) 内嵌 P 伺服子步循环：每子步按实测轮速算力矩（模拟小核速度环）。
        #    堵转/顶墙时伺服饱和到 ±MAX_TAU，物理行为与真机速度环深度饱和一致，
        #    "顶推过坎"能力仍可学到（只是通过速度指令表达）。
        tau_accum = np.zeros(len(self.actuators))
        for _ in range(self.frame_skip):
            for dof, sign in self._brush_dofs:
                self.client.data.qvel[dof] = sign * self.BRUSH_SPEED
            wv = np.asarray(self.client.get_act_joint_velocities())[self.actuators]
            wv_meas = wv + np.random.normal(0.0, self._wvel_noise, size=wv.shape)
            tau = np.clip(self._kp_eff * (w_active - wv_meas), self.tau_low, self.tau_high)
            self.client.set_motor_torque(tau)
            self.client.step()
            tau_accum += tau

        self.prev_action = action.copy()
        self.prev_torque = tau_accum / self.frame_skip   # 本控制步伺服力矩均值（能耗项用）
        return self.prev_torque
