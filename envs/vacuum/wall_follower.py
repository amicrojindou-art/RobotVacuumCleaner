"""沿边沿墙（edge/wall-following）运动控制逻辑。

传感器输入是两个线激光（envs.vacuum.sensors.LineLaser 的读数）：
  side  : 右侧线激光（与真机一致装在右侧）——
          沿墙时保持"机身右边缘-墙面"横向距离 = TARGET_DIST(1cm)
  front : 正前方线激光 —— 前向障碍检测与内角转向触发
可选里程计速度 odom_speed（真机由轮式里程计+陀螺/光流融合得到），用于检测
"激光盲区障碍"（2~3cm 低矮物）导致的卡住并触发脱困。

控制结构（经典两层）：
  上层  状态机 + 横向距离 PD      ->  期望 (v, w)（机体前向线速度 / 偏航角速度）
  下层  差速运动学分解 + 轮速 PI  ->  左右轮力矩（与 RL 动作同一力矩接口）

状态机（墙始终在机器人右侧，绕房间逆时针巡边）：
  SEEK         直行找墙。前方出现墙 -> TURN_LEFT；右侧先贴到墙 -> FOLLOW
  TURN_LEFT    原地左转（内角/初始对齐）。等前方畅通后，跟踪右侧距离的运行
               最小值；距离越过最小值回升 = 机身已转过与新墙平行的姿态，退出
               到 FOLLOW（避免斜看到墙就提前退出、以进墙姿态楔死）
  FOLLOW       沿墙行进，PD 把右侧距离收敛到 1cm。前方出现墙 -> TURN_LEFT；
               右侧丢墙（外角/阳角/门洞） -> WRAP_RIGHT
  WRAP_RIGHT   阳角包边，三段式（直行 -> 原地右转90° -> 前出重找墙）：
               1) 先直行 WRAP_STRAIGHT_DIST，让机身整体越过凸角；
               2) 原地右转 90°（偏航角由轮式里程计积分判定）；
               3) 直行前出，右侧重新看到墙 -> FOLLOW；前出超过
                  WRAP_RESEEK_DIST 仍未见墙（大于90°的阳角）-> 回到 2) 再转。
               这样薄隔断的墙端（如门洞阳角，需绕行180°）会被拆成
               "旧墙面 -> 墙端面 -> 新墙面"两次连续包边自然通过，机器人
               从门洞进入另一房间沿墙、绕完后再从对侧阳角包出来；且过门槛
               条时是垂直直行通过（最利于爬坎的姿态）。
               途中前方出现墙 -> TURN_LEFT；原地右转超时（被卡）-> 脱困
  ESCAPE_BACK  脱困一段：检测到"有速度指令但实际几乎不动"（多为激光盲区的
  ESCAPE_TURN  低矮障碍顶住底盘/轮子打滑）时，先倒车，再左转固定角度换向，
               然后回 SEEK 重新找墙（真机对盲区障碍同样只能靠脱困策略）

过坎强推（叠加在状态机之上，见 _apply_climb）：前进类状态被顶住 0.4s 时，
先给一段大速度指令让轮速 PI 饱和到最大力矩直推——12mm 门槛条/毯沿这类低于
激光最低射线的"可爬小坎"（恒速指令推不过、全力矩可过）直接碾过去，状态机
上下文不丢；强推仍不动才进入倒车脱困。
"""

from enum import Enum

import numpy as np

from envs.vacuum.gen_xml import WHEEL_RADIUS, WHEEL_Y, MAX_WHEEL_TORQUE


class FollowState(Enum):
    SEEK = 1
    TURN_LEFT = 2
    FOLLOW = 3
    WRAP_RIGHT = 4
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
    TARGET_DIST  = 0.01    # 沿墙目标横向距离：机身右边缘距墙 1cm
    FRONT_TURN   = 0.055   # 前方距离小于此值 -> 触发内角左转
    FRONT_CLEAR  = 0.085   # 前方距离大于此值（或无回波）视为畅通
    SIDE_ACQUIRE = 0.09    # 右侧距离小于此值视为"已捕获墙面"

    CRUISE_V = 0.15        # 找墙直行速度
    FOLLOW_V = 0.12        # 沿墙巡航速度
    TURN_W   = 1.4         # 原地左转角速度（内角转向：墙在右侧 -> 向左让开）
    APPROACH_V = 0.12      # 前方见障后的接近限速（不随 speed_scale 缩放）

    # 阳角包边（三段式：直行 -> 原地右转90° -> 前出重找墙）
    WRAP_V       = 0.12    # 包边直行速度（垂直过门槛条需要足够动量）
    WRAP_PIVOT_W = -1.4    # 原地右转角速度
    # 直行距离 = 机身半径 + 目标间距 + 余量：转过90°后右边缘距墙端面约3.5cm，
    # 在激光量程内且不擦角，交给 FOLLOW 的 PD 收敛回 1cm
    WRAP_STRAIGHT_DIST = 0.21
    WRAP_RESEEK_DIST   = 0.30              # 前出找墙最大距离，超过 -> 再右转90°
    WRAP_PIVOT_YAW     = np.deg2rad(88.0)  # 原地右转目标角（留2°给动量过冲）
    WRAP_EARLY_YAW     = np.deg2rad(60.0)  # 转过此角后右侧见墙可提前退出
    WRAP_PIVOT_TIMEOUT = 2.5               # 原地右转超时 (s)，被卡 -> 脱困

    # 横向距离 PD。沿墙闭环近似 e'' + v*KD*e' + v*KP*e = 0（v 为巡航速度），
    # 取 KP=8/KD=16，在 v=0.12 时 wn~1.0 rad/s、阻尼比~1（临界阻尼，无超调不打转）
    KP = 8.0
    KD = 16.0
    DERR_ALPHA = 0.35      # 距离微分的低通滤波系数（0~1，越小越平滑）
    W_MAX = 0.8            # 沿墙时的转向角速度限幅

    LOST_FRAMES = 3        # 右侧连续丢墙帧数阈值（抗单帧抖动）
    MIN_DWELL = 0.15       # 状态最短驻留时间 (s)，抗切换抖动

    # 过坎强推：前进被顶住（多为激光盲区的低矮门槛/毯沿顶住万向球）时，
    # 先给一段大速度指令让轮速 PI 饱和到最大力矩直推过坎（实测 12mm 门槛
    # 条恒速 0.12m/s 推不过、全力矩可过），推不动再进入倒车脱困
    CLIMB_TRIG_T = 0.4     # 停滞该时长即触发强推 (s)
    CLIMB_V      = 0.5     # 强推速度指令 (m/s)，远超巡航 -> 轮力矩饱和
    CLIMB_T      = 1.2     # 单次强推最长时长 (s)
    CLIMB_REARM  = 0.05    # 里程计速度高于此值视为已脱离，重新武装强推
    CLIMB_EXIT   = 0.12    # 强推中速度恢复到此值即提前结束（已越过坎）

    # 脱困：有前进指令但实际几乎不动/远慢于指令（盲区障碍顶住、打滑、
    # 顶着门槛侧蹭都算）。停滞阈值 = max(下限, min(0.3*指令速度, 上限))
    STALL_CMD_V   = 0.03   # 指令速度阈值
    STALL_SPEED   = 0.015  # 停滞判定速度下限 (m/s)
    STALL_RATIO   = 0.3    # 实际速度低于指令速度的该比例也视为停滞
    STALL_CAP     = 0.05   # 停滞判定速度上限（强推 0.5 指令下不误判爬坎蠕动）
    STALL_TIME    = 1.8    # 持续时间 (s)：覆盖强推触发 0.4 + 强推 1.2 + 余量
    ESCAPE_BACK_T = 1.2    # 倒车时长 (s)
    ESCAPE_TURN_T = 0.75   # 左转时长 (s)，约 60 度，换向避开盲区障碍

    def __init__(self, dt, speed_scale=1.0):
        """dt: 控制周期 (s)；speed_scale: 行进速度缩放倍数。

        speed_scale 只缩放长直段速度（找墙/沿墙巡航）和沿墙转向限幅
        （保持最大曲率不变）；阳角包边全程（直行/原地转/前出）、几何距离、
        内角/前方触发距离保持原速原值——包边段只有 0.2~0.3m，提速收益极小，
        高速反而会因横向漂移错过薄墙端面（仅 5cm 宽）的重捕窗口。
        """
        self.dt = dt
        self.CRUISE_V = self.CRUISE_V * speed_scale
        self.FOLLOW_V = self.FOLLOW_V * speed_scale
        self.W_MAX = self.W_MAX * speed_scale
        self.pi_left = WheelSpeedPI()
        self.pi_right = WheelSpeedPI()
        self.reset()

    def reset(self):
        self.state = FollowState.SEEK
        self._state_t = 0.0
        self._prev_err = None
        self._derr_f = 0.0
        self._lost = 0
        self._wrap_phase = 0
        self._wrap_dist = 0.0
        self._wrap_cycles = 0
        self._pivot_yaw = 0.0
        self._phase_t = 0.0
        self._min_side = np.inf
        self._front_cleared = False
        self._stall_t = 0.0
        self._boost_t = 0.0
        self._boost_armed = True
        self.escape_count = 0
        self.climb_count = 0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.pi_left.reset()
        self.pi_right.reset()

    # ------------------------------------------------------------------
    def step(self, front, side, wheel_vel, odom_speed=None):
        """计算一个控制步的轮力矩。

        front/side : LaserReading（side 为右侧激光）
        wheel_vel  : (左轮, 右轮) 角速度 rad/s
        odom_speed : 里程计水平速度模长 m/s（None 则关闭脱困检测）
        返回 (2,) 左右轮力矩。
        """
        # 轮式里程计偏航角速度（差速运动学反解），供包边原地右转的角度积分
        yaw_rate = WHEEL_RADIUS * (wheel_vel[1] - wheel_vel[0]) / (2.0 * WHEEL_Y)
        v, w = self._plan(front, side, odom_speed, yaw_rate)
        # 前方已看到障碍时限速接近（高速巡航下留出制动距离，避免撞墙后才转向）
        if front.hit and front.distance < self.FRONT_CLEAR and v > self.APPROACH_V:
            v = self.APPROACH_V
        v, w = self._apply_climb(v, w, odom_speed, front)
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
        self._wrap_phase = 0
        self._wrap_dist = 0.0
        self._wrap_cycles = 0
        self._pivot_yaw = 0.0
        self._phase_t = 0.0
        self._min_side = np.inf
        self._front_cleared = False
        self._stall_t = 0.0

    def _stall_thresh(self):
        """停滞判定速度阈值：随指令速度自适应，带上下限。"""
        return max(self.STALL_SPEED,
                   min(self.STALL_RATIO * abs(self.cmd_v), self.STALL_CAP))

    def _apply_climb(self, v, w, odom_speed, front):
        """过坎强推：前进被盲区低坎顶住时，饱和力矩直推一段（不改变状态机）。

        只在前进直行类场景生效（SEEK / FOLLOW / WRAP 的直行阶段），且前方
        激光必须无近距回波（有回波说明顶住的是可见墙面/家具，强推只会硬顶，
        交给转向/脱困处理）；每个停滞事件只强推一次，推过（里程计恢复）后
        重新武装，推不过由脱困接管。
        """
        if odom_speed is None:
            return v, w
        front_open = (not front.hit) or front.distance > self.FRONT_CLEAR
        boostable = (front_open and
                     (self.state in (FollowState.SEEK, FollowState.FOLLOW) or
                      (self.state == FollowState.WRAP_RIGHT and self._wrap_phase != 1)))
        if odom_speed > self.CLIMB_REARM:
            self._boost_armed = True
        if self._boost_t > 0.0:
            if not boostable:
                self._boost_t = 0.0    # 状态已切换（转向/脱困），取消强推
            elif (odom_speed > self.CLIMB_EXIT and
                  self._boost_t < self.CLIMB_T - 0.2):
                self._boost_t = 0.0    # 速度已恢复 -> 已越坎，提前结束
            else:
                self._boost_t -= self.dt
                return self.CLIMB_V, 0.0
        elif (boostable and self._boost_armed and v > self.STALL_CMD_V and
              self._stall_t >= self.CLIMB_TRIG_T):
            self._boost_armed = False
            self._boost_t = self.CLIMB_T
            self.climb_count += 1
            return self.CLIMB_V, 0.0
        return v, w

    def _check_stall(self, odom_speed):
        """前进指令下实际速度远低于指令 -> 顶上盲区障碍/打滑/顶坎侧蹭。"""
        if odom_speed is None:
            return False
        moving_cmd = abs(self.cmd_v) > self.STALL_CMD_V
        if moving_cmd and odom_speed < self._stall_thresh():
            self._stall_t += self.dt
        else:
            self._stall_t = 0.0
        return self._stall_t >= self.STALL_TIME

    def _plan(self, front, side, odom_speed, yaw_rate=0.0):
        self._state_t += self.dt
        dwelled = self._state_t >= self.MIN_DWELL

        front_blocked = front.hit and front.distance < self.FRONT_TURN
        front_clear = (not front.hit) or front.distance > self.FRONT_CLEAR
        side_seen = side.hit and side.distance < self.SIDE_ACQUIRE

        self._lost = 0 if side.hit else self._lost + 1

        # ---- 脱困检测（对所有前进状态生效）----
        if self.state in (FollowState.SEEK, FollowState.FOLLOW, FollowState.WRAP_RIGHT):
            if self._check_stall(odom_speed):
                self.escape_count += 1
                self._switch(FollowState.ESCAPE_BACK)

        # ---- 状态转移 ----
        if self.state == FollowState.SEEK:
            if front_blocked:
                self._switch(FollowState.TURN_LEFT)
            elif side_seen:
                self._switch(FollowState.FOLLOW)

        elif self.state == FollowState.TURN_LEFT:
            # 先等前方畅通（机头扫过旧墙/内角另一面），再开始跟踪右侧距离的
            # 运行最小值：左转中右侧距离先减小（右轴逐渐垂直于新墙）、过平行
            # 姿态后回升。以"越过最小值回升"为对齐完成的判据。
            if front_clear:
                self._front_cleared = True
                if side.hit:
                    self._min_side = min(self._min_side, side.distance)
            passed_parallel = (self._front_cleared and side.hit and
                               self._min_side < self.SIDE_ACQUIRE and
                               side.distance > self._min_side + 0.002)
            if dwelled and front_clear and side_seen and passed_parallel:
                self._switch(FollowState.FOLLOW)

        elif self.state == FollowState.FOLLOW:
            if front_blocked:
                self._switch(FollowState.TURN_LEFT)
            elif dwelled and self._lost >= self.LOST_FRAMES:
                self._switch(FollowState.WRAP_RIGHT)

        elif self.state == FollowState.WRAP_RIGHT:
            self._phase_t += self.dt
            if self._wrap_phase == 1:
                # 原地右转 90°：积分轮式里程计偏航角（右转 yaw_rate<0，取负累加）
                self._pivot_yaw += -yaw_rate * self.dt
                if self._phase_t >= self.WRAP_PIVOT_TIMEOUT:
                    # 原地转都转不动 -> 被卡（骑坎/顶住），走脱困
                    self.escape_count += 1
                    self._switch(FollowState.ESCAPE_BACK)
                elif side_seen and self._pivot_yaw >= self.WRAP_EARLY_YAW:
                    self._switch(FollowState.FOLLOW)
                elif self._pivot_yaw >= self.WRAP_PIVOT_YAW:
                    self._wrap_phase = 2      # 转完 -> 前出重找墙
                    self._wrap_dist = 0.0
                    self._phase_t = 0.0
            else:
                # 直行阶段（0=越过凸角，2=转过90°后前出找新墙）
                adv = self.cmd_v if odom_speed is None else odom_speed
                self._wrap_dist += max(adv, 0.0) * self.dt
                if front_blocked:
                    self._switch(FollowState.TURN_LEFT)
                elif side_seen:
                    self._switch(FollowState.FOLLOW)
                else:
                    cap = (self.WRAP_STRAIGHT_DIST if self._wrap_phase == 0
                           else self.WRAP_RESEEK_DIST)
                    if self._wrap_dist >= cap:
                        if self._wrap_phase == 2 and self._wrap_cycles >= 1:
                            # 追加右转一轮后仍找不到墙 -> 周围是开阔区，
                            # 放弃包边回 SEEK 直行找墙（防止原地绕方框死循环）
                            self._switch(FollowState.SEEK)
                        else:
                            # 直行到位 / 前出未见墙（>90°阳角）-> 原地右转90°
                            if self._wrap_phase == 2:
                                self._wrap_cycles += 1
                            self._wrap_phase = 1
                            self._pivot_yaw = 0.0
                            self._phase_t = 0.0

        elif self.state == FollowState.ESCAPE_BACK:
            if self._state_t >= self.ESCAPE_BACK_T:
                self._switch(FollowState.ESCAPE_TURN)

        elif self.state == FollowState.ESCAPE_TURN:
            if self._state_t >= self.ESCAPE_TURN_T:
                self._switch(FollowState.SEEK)

        # ---- 各状态的 (v, w) ----
        if self.state == FollowState.SEEK:
            return self.CRUISE_V, 0.0

        if self.state == FollowState.TURN_LEFT:
            return 0.0, self.TURN_W

        if self.state == FollowState.ESCAPE_BACK:
            return -0.08, 0.0

        if self.state == FollowState.ESCAPE_TURN:
            return 0.0, self.TURN_W

        if self.state == FollowState.WRAP_RIGHT:
            if self._wrap_phase == 1:
                return 0.0, self.WRAP_PIVOT_W    # 原地右转
            return self.WRAP_V, 0.0              # 直行（越角 / 前出找墙）

        # FOLLOW：横向距离 PD（墙在右侧，误差>0 离墙偏远 -> w<0 右转贴墙）。
        # 微分项经限幅 + 低通，抑制接触/求解抖动引起的测距毛刺
        if not side.hit:
            # 右侧无回波（丢墙待确认）：保持直行，不能拿"无回波=量程"的
            # 假距离喂 PD 猛打右转——高速下会在包边接管前把姿态转歪
            return self.FOLLOW_V, 0.0
        err = side.distance - self.TARGET_DIST
        derr = 0.0 if self._prev_err is None else (err - self._prev_err) / self.dt
        derr = float(np.clip(derr, -0.3, 0.3))
        self._derr_f += self.DERR_ALPHA * (derr - self._derr_f)
        self._prev_err = err
        # 过近侧（err<0，可能已接近擦墙）用更大比例增益尽快让开
        kp = self.KP if err >= 0.0 else 2.0 * self.KP
        w = float(np.clip(-(kp * err + self.KD * self._derr_f),
                          -self.W_MAX, self.W_MAX))

        # 转向越急越减速；巡航速度基本恒定（PD 阻尼分析假设 v 恒定），
        # 仅在前方已很近（即将触发内角转向）时预减速
        v = self.FOLLOW_V * (1.0 - 0.5 * min(abs(w) / self.W_MAX, 1.0))
        if front.hit and front.distance < 0.07:
            v = min(v, 0.05)
        return v, w
