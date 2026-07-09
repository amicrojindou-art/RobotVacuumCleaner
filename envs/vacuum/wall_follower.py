"""沿边沿墙（edge/wall-following）运动控制逻辑。

传感器输入是两个线激光（envs.vacuum.sensors.LineLaser 的读数）：
  left  : 左侧线激光 —— 沿墙时保持"机身左边缘-墙面"横向距离 = TARGET_DIST(1cm)
  front : 正前方线激光 —— 前向障碍检测与内角转向触发
可选里程计速度 odom_speed（真机由轮式里程计+陀螺/光流融合得到），用于检测
"激光盲区障碍"（2~3cm 低矮物）导致的卡住并触发脱困。

控制结构（经典两层）：
  上层  状态机 + 横向距离 PD      ->  期望 (v, w)（机体前向线速度 / 偏航角速度）
  下层  差速运动学分解 + 轮速 PI  ->  左右轮力矩（与 RL 动作同一力矩接口）

状态机（墙始终在机器人左侧，绕房间顺时针巡边）：
  SEEK         直行找墙。前方出现墙 -> TURN_RIGHT；左侧先贴到墙 -> FOLLOW
  TURN_RIGHT   原地右转（内角/初始对齐）。等前方畅通后，跟踪左侧距离的运行
               最小值；距离越过最小值回升 = 机身已转过与新墙平行的姿态，退出
               到 FOLLOW（避免斜看到墙就提前退出、以进墙姿态楔死）
  FOLLOW       沿墙行进，PD 把左侧距离收敛到 1cm。前方出现墙 -> TURN_RIGHT；
               左侧丢墙（外角/门洞） -> WRAP_LEFT
  WRAP_LEFT    外角包边：先直行让机身后半越过凸角，再左转圆弧绕角；
               左侧重新看到墙 -> FOLLOW；途中前方出现墙 -> TURN_RIGHT
  ESCAPE_BACK  脱困一段：检测到"有速度指令但里程计几乎不动"（多为激光盲区的
  ESCAPE_TURN  低矮障碍顶住底盘/轮子打滑）时，先倒车，再右转固定角度换向，
               然后回 SEEK 重新找墙（真机对盲区障碍同样只能靠脱困策略）
"""

from enum import Enum

import numpy as np

from envs.vacuum.gen_xml import WHEEL_RADIUS, WHEEL_Y, MAX_WHEEL_TORQUE


class FollowState(Enum):
    SEEK = 1
    TURN_RIGHT = 2
    FOLLOW = 3
    WRAP_LEFT = 4
    ESCAPE_BACK = 5
    ESCAPE_TURN = 6


class WheelSpeedPI(object):
    """单轮转速 PI 控制器：期望轮速 -> 电机力矩（带积分限幅抗饱和）。"""

    def __init__(self, kp=0.25, ki=1.5, i_max=0.8, tau_max=MAX_WHEEL_TORQUE):
        self.kp = kp
        self.ki = ki
        self.i_max = i_max
        self.tau_max = tau_max
        self.integ = 0.0

    def reset(self):
        self.integ = 0.0

    def step(self, omega_des, omega, dt):
        err = omega_des - omega
        self.integ = float(np.clip(self.integ + self.ki * err * dt,
                                   -self.i_max, self.i_max))
        return float(np.clip(self.kp * err + self.integ,
                             -self.tau_max, self.tau_max))


class WallFollower(object):
    # ---- 行为参数（单位：米 / 米每秒 / 弧度每秒 / 秒）----
    TARGET_DIST  = 0.01    # 沿墙目标横向距离：机身左边缘距墙 1cm
    FRONT_TURN   = 0.055   # 前方距离小于此值 -> 触发内角右转
    FRONT_CLEAR  = 0.085   # 前方距离大于此值（或无回波）视为畅通
    LEFT_ACQUIRE = 0.09    # 左侧距离小于此值视为"已捕获墙面"

    CRUISE_V = 0.15        # 找墙直行速度
    FOLLOW_V = 0.12        # 沿墙巡航速度
    TURN_W   = -1.4        # 原地右转角速度
    WRAP_V   = 0.08        # 外角包边线速度
    # 包边圆弧半径 ~= 机器人半径 + 目标间距，使机身绕凸角外缘扫过
    WRAP_W   = WRAP_V / (0.175 + TARGET_DIST)
    WRAP_STRAIGHT_DIST = 0.20  # 丢墙后先直行的距离（让机身后半越过凸角）

    # 横向距离 PD。沿墙闭环近似 e'' + v*KD*e' + v*KP*e = 0（v 为巡航速度），
    # 取 KP=8/KD=16，在 v=0.12 时 wn~1.0 rad/s、阻尼比~1（临界阻尼，无超调不打转）
    KP = 8.0
    KD = 16.0
    DERR_ALPHA = 0.35      # 距离微分的低通滤波系数（0~1，越小越平滑）
    W_MAX = 0.8            # 沿墙时的转向角速度限幅

    LOST_FRAMES = 3        # 左侧连续丢墙帧数阈值（抗单帧抖动）
    MIN_DWELL = 0.15       # 状态最短驻留时间 (s)，抗切换抖动

    # 脱困：有前进指令但里程计几乎不动（激光盲区障碍顶住/打滑）
    STALL_CMD_V   = 0.03   # 指令速度阈值
    STALL_SPEED   = 0.015  # 里程计速度阈值
    STALL_TIME    = 1.5    # 持续时间 (s)
    ESCAPE_BACK_T = 1.2    # 倒车时长 (s)
    ESCAPE_TURN_T = 0.75   # 右转时长 (s)，约 60 度，换向避开盲区障碍

    def __init__(self, dt):
        self.dt = dt
        self.pi_left = WheelSpeedPI()
        self.pi_right = WheelSpeedPI()
        self.reset()

    def reset(self):
        self.state = FollowState.SEEK
        self._state_t = 0.0
        self._prev_err = None
        self._derr_f = 0.0
        self._lost = 0
        self._wrap_dist = 0.0
        self._min_left = np.inf
        self._front_cleared = False
        self._stall_t = 0.0
        self.escape_count = 0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.pi_left.reset()
        self.pi_right.reset()

    # ------------------------------------------------------------------
    def step(self, front, left, wheel_vel, odom_speed=None):
        """计算一个控制步的轮力矩。

        front/left : LaserReading
        wheel_vel  : (左轮, 右轮) 角速度 rad/s
        odom_speed : 里程计水平速度模长 m/s（None 则关闭脱困检测）
        返回 (2,) 左右轮力矩。
        """
        v, w = self._plan(front, left, odom_speed)
        self.cmd_v, self.cmd_w = v, w

        # 差速运动学：v = r*(wl+wr)/2, w = r*(wr-wl)/(2*WHEEL_Y)
        omega_l = (v - w * WHEEL_Y) / WHEEL_RADIUS
        omega_r = (v + w * WHEEL_Y) / WHEEL_RADIUS

        tau_l = self.pi_left.step(omega_l, wheel_vel[0], self.dt)
        tau_r = self.pi_right.step(omega_r, wheel_vel[1], self.dt)
        return np.array([tau_l, tau_r])

    # ------------------------------------------------------------------
    def _switch(self, new_state):
        self.state = new_state
        self._state_t = 0.0
        self._prev_err = None
        self._derr_f = 0.0
        self._wrap_dist = 0.0
        self._min_left = np.inf
        self._front_cleared = False
        self._stall_t = 0.0

    def _check_stall(self, odom_speed):
        """前进指令下里程计几乎不动 -> 撞上激光盲区障碍或打滑。"""
        if odom_speed is None:
            return False
        moving_cmd = abs(self.cmd_v) > self.STALL_CMD_V
        if moving_cmd and odom_speed < self.STALL_SPEED:
            self._stall_t += self.dt
        else:
            self._stall_t = 0.0
        return self._stall_t >= self.STALL_TIME

    def _plan(self, front, left, odom_speed):
        self._state_t += self.dt
        dwelled = self._state_t >= self.MIN_DWELL

        front_blocked = front.hit and front.distance < self.FRONT_TURN
        front_clear = (not front.hit) or front.distance > self.FRONT_CLEAR
        left_seen = left.hit and left.distance < self.LEFT_ACQUIRE

        self._lost = 0 if left.hit else self._lost + 1

        # ---- 脱困检测（对所有前进状态生效）----
        if self.state in (FollowState.SEEK, FollowState.FOLLOW, FollowState.WRAP_LEFT):
            if self._check_stall(odom_speed):
                self.escape_count += 1
                self._switch(FollowState.ESCAPE_BACK)

        # ---- 状态转移 ----
        if self.state == FollowState.SEEK:
            if front_blocked:
                self._switch(FollowState.TURN_RIGHT)
            elif left_seen:
                self._switch(FollowState.FOLLOW)

        elif self.state == FollowState.TURN_RIGHT:
            # 先等前方畅通（机头扫过旧墙/内角另一面），再开始跟踪左侧距离的
            # 运行最小值：右转中左侧距离先减小（左轴逐渐垂直于新墙）、过平行
            # 姿态后回升。以"越过最小值回升"为对齐完成的判据。
            if front_clear:
                self._front_cleared = True
                if left.hit:
                    self._min_left = min(self._min_left, left.distance)
            passed_parallel = (self._front_cleared and left.hit and
                               self._min_left < self.LEFT_ACQUIRE and
                               left.distance > self._min_left + 0.002)
            if dwelled and front_clear and left_seen and passed_parallel:
                self._switch(FollowState.FOLLOW)

        elif self.state == FollowState.FOLLOW:
            if front_blocked:
                self._switch(FollowState.TURN_RIGHT)
            elif dwelled and self._lost >= self.LOST_FRAMES:
                self._switch(FollowState.WRAP_LEFT)

        elif self.state == FollowState.WRAP_LEFT:
            if front_blocked:
                self._switch(FollowState.TURN_RIGHT)
            elif left_seen:
                self._switch(FollowState.FOLLOW)

        elif self.state == FollowState.ESCAPE_BACK:
            if self._state_t >= self.ESCAPE_BACK_T:
                self._switch(FollowState.ESCAPE_TURN)

        elif self.state == FollowState.ESCAPE_TURN:
            if self._state_t >= self.ESCAPE_TURN_T:
                self._switch(FollowState.SEEK)

        # ---- 各状态的 (v, w) ----
        if self.state == FollowState.SEEK:
            return self.CRUISE_V, 0.0

        if self.state == FollowState.TURN_RIGHT:
            return 0.0, self.TURN_W

        if self.state == FollowState.ESCAPE_BACK:
            return -0.08, 0.0

        if self.state == FollowState.ESCAPE_TURN:
            return 0.0, self.TURN_W

        if self.state == FollowState.WRAP_LEFT:
            # 阶段1：直行，让机身（激光在机身中部）后半越过凸角
            self._wrap_dist += self.WRAP_V * self.dt
            if self._wrap_dist < self.WRAP_STRAIGHT_DIST:
                return self.WRAP_V, 0.0
            # 阶段2：小半径左转圆弧包边
            return self.WRAP_V, self.WRAP_W

        # FOLLOW：横向距离 PD（误差>0 离墙偏远 -> w>0 左转贴墙）。
        # 微分项经限幅 + 低通，抑制接触/求解抖动引起的测距毛刺
        err = left.distance - self.TARGET_DIST
        derr = 0.0 if self._prev_err is None else (err - self._prev_err) / self.dt
        derr = float(np.clip(derr, -0.3, 0.3))
        self._derr_f += self.DERR_ALPHA * (derr - self._derr_f)
        self._prev_err = err
        # 过近侧（err<0，可能已接近擦墙）用更大比例增益尽快让开
        kp = self.KP if err >= 0.0 else 2.0 * self.KP
        w = float(np.clip(kp * err + self.KD * self._derr_f,
                          -self.W_MAX, self.W_MAX))

        # 转向越急越减速；巡航速度基本恒定（PD 阻尼分析假设 v 恒定），
        # 仅在前方已很近（即将触发内角转向）时预减速
        v = self.FOLLOW_V * (1.0 - 0.5 * min(abs(w) / self.W_MAX, 1.0))
        if front.hit and front.distance < 0.07:
            v = min(v, 0.05)
        return v, w
