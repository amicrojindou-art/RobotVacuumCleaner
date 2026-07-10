"""扫地机器人"沿边沿墙清扫"RL 任务。

目标：只依靠两个线激光（左侧 + 正前方）与本体感知，学会以 1cm 横向距离
贴墙行进，并泛化到任意墙形（内角/外角/斜墙/不同高度的墙沿）。

泛化性来源（每个 episode 重新随机）：
  1) 随机墙面课程：用 box01~06 摆一条随机折线墙（长度/拐角方向与角度/
     墙高随机），墙高既有 >=7cm 的可靠段、也有 4.5~7cm 的低置信段，
     迫使策略利用激光 confidence；
  2) 初始位姿随机：大部分回合出生在墙边 2~9cm；一部分回合出生在离墙
     12~45cm 处（"找墙"课程——先接近、捕获墙面再贴边，这是策略能在
     陌生户型里从任意位置启动的关键）。找墙回合里又有 50% 是"正对墙面"
     出生（head_on），必须靠前向激光把速度降到 0、停在墙前 ~1cm 再转向对齐，
     而不是撞上去；其余找墙回合中另有一部分"全盲出生"（任意朝向、更远
     距离，激光量程内可能长时间无任何回波——对应家居场景任意位置任意
     朝向启动），期望行为是不打转、径直巡航直到捕获墙面；
  3) 传感器噪声：激光测距高斯噪声随课程升到 ~2mm；
  4) 地面摩擦随机化。

奖励设计要点（防局部最优 / 防三类坏行为）：
  - progress（沿墙推进）用贴边质量门控：|d-1cm| 越差，速度分越少。
    否则"斜压在墙上滑行"能白拿满速度分，成为强局部最优；速度分本身是
    在 V_DES 处封顶的"帐篷"形——超速线性扣回、2×V_DES 归零、更快为负：
    旧 clip 形超速零代价，策略实际巡航到 2×V_DES，10cm 量程下反应时间
    减半，贴边振荡/丢墙/撞墙全被放大；
  - 蹭墙（机身接触）重罚，且持续接触 1.5s 直接终止 episode——
    把"贴墙推着走/顶进墙角"这条捷径彻底封死；
  - approach（前向安全 / "距墙 1cm 停住"）：由前向距离定一条前向限速——1cm 处
    限速 0、9cm 及以外满速，只罚超过限速的部分。强制"接近墙面把速度降到 0、
    停在墙前 ~1cm 再转向"，而不是先撞墙再沿墙（家居场景先直冲撞墙的根因是
    前向激光在训练里从没被用于刹车）；
  - 阳角（外角）包边窗口：左墙"刚"丢 + 前方无障碍时，鼓励先直行越过墙角
    再左转找墙，避免急着左勾把外露墙角勾进机身左前缘（"卡进机身"）；
  - wobble（直行段抖动抑制）：贴边良好且前方无障碍时惩罚偏航角速度，
    压住"沿墙一直摇晃"的极限环；用 track + 前方无障碍双重门控，不误伤拐弯；
  - 缺口桥接（门洞/断墙）：连续墙上留出一段两端共线、中间无墙的缺口，
    机器人抵达时左侧激光会丢墙。此刻若前方仍有共线墙可续贴（真值判定），
    就【保持直行匀速穿过缺口】、几乎不罚丢墙，并压住偏航（别拐进缺口）；
    到对侧自然重新贴墙。与"外角包边"用真值区分：外角前方没有共线墙，
    仍走原来的包边逻辑。没有这一项，策略把缺口误当外角/丢墙 —— 中途减速、
    原地转向、重新找墙，沿边作业被打断。缺口宽度上限压在 0.35m：必须明显
    小于家居门洞宽度（0.6~0.9m），门洞才会被判成"外角拐入"而不是被直行
    桥接跳过；
  - 找墙巡航（seek）防打转：激光全盲时（首次找墙、或捕获后丢墙超过包边
    宽限的"丢墙重找"——两者同治）给径直巡航更高的速度分、并惩罚偏航
    角速度（盲区专用的更高饱和上限，保证高速打转仍有梯度可下降）。
    10cm 量程下原地旋转毫无侦察价值，家居场景"出生打转 / 丢墙后打转
    甩出"的根因就是盲区里"转圈"与"直行"奖励几乎无差、且旧防打转罚
    在 0.6rad/s 就饱和（实测策略以 2.5~3rad/s 打转，罚被速度分淹没）；
  - 墙端 U 型包边（门洞侧柱）课程：部分段间过渡改为"下一段贴同一堵墙的
    背面反向延伸、端面对齐"，机器人到墙端后需连做两个左外角（合计≈180°）
    绕过墙端、再沿背面继续贴边 —— 对应家居场景"沿隔断走到门洞侧柱、
    拐进里屋继续沿墙"。此前课程只有 55°~115° 单拐角，策略在门洞侧柱处
    不会包边、原地打转甩出。

终止：翻车 / 弹飞 / 出界 / 持续蹭墙 / 丢墙超时（首次捕获前有更长宽限；
      正在桥接缺口时不按丢墙终止）。
"""

import numpy as np
import transforms3d as tf3


class WallFollowTask(object):
    # ---- 任务超参数 ----
    TARGET_DIST = 0.01       # 贴边目标：机身左边缘距墙 1cm
    V_DES = 0.15             # 期望沿墙速度 (m/s)
    MAX_TAU = 2.0            # 力矩上限（归一化用，与 XML 一致）
    FRONT_SAFE_DIST = 0.09   # 正前方开始减速的距离（≈激光量程；防撞墙）
    FRONT_STOP_GAP  = 0.01   # 期望停墙间距：正前方 1cm 处前向速度应降到 0
    YAW_RATE_SCALE = 0.6     # 抖动抑制归一化尺度 (rad/s)，由实测 yaw_rate 分布标定
    # 阳角包边宽限窗口 (s)：左侧激光装在机体横向中心线(x=0)上，丢墙那一刻墙角
    # 正好在机身正侧方，要把墙角甩到机身后方需再前进约一个底盘半径
    # （0.175m / V_DES 0.15 ≈ 1.17s）。窗口太短会逼策略提前左勾、勾住墙角。
    WRAP_GRACE_TIME = 1.2
    # 墙体缺口（门洞/断墙）课程：连续墙上留一段两端共线、中间无墙的缺口，
    # 训练"直行桥接穿过"。宽度随课程展开；太宽会与外角越来越难区分（同一段
    # 激光观测下"缺口"与"外角"前 gap 距离内不可分），故上限保守。
    GAP_MIN = 0.12           # 缺口最小宽度 (m)
    # 缺口最大宽度基线，随课程升到 +0.15 -> 0.35m。上限必须明显小于家居
    # 门洞宽度（0.6~0.9m）：桥接课程教的是"这个宽度以内直行跨过"，若上限
    # 逼近门洞宽度，策略会把门洞当缺口直行桥接、永远不拐进里屋。
    GAP_MAX_BASE = 0.20
    GAP_PROB = 0.35          # 每个共线段边界插入缺口的概率
    U_TURN_PROB = 0.25       # 段间过渡为"墙端 U 型包边"（门洞侧柱课程）的概率
    GAP_BRIDGE_LAT = 0.35    # 判定"正在桥接该缺口"的最大横向偏离 (m)
    GAP_BRIDGE_HEAD = 0.6    # 判定桥接的最小航向·墙向余弦（须朝墙向直行）
    WALL_HALF_T = 0.08       # 训练墙半厚度（厚墙 + 硬接触，防止大力顶穿）
    LOST_TERM_TIME = 4.0     # 捕获过墙后，连续丢墙超时终止 (s)
    ACQUIRE_GRACE = 8.0      # 首次捕获前的找墙宽限 (s)
    SCRAPE_TERM_TIME = 1.5   # 机身持续接触超时终止 (s)
    CURRICULUM_ITRS = 1500.0

    def __init__(self,
                 client=None,
                 dt=0.025,
                 root_body='base',
                 chassis_geom='chassis_geom',
                 num_boxes=6):
        self._client = client
        self._control_dt = dt
        self._root_body_name = root_body
        self._chassis_geom_name = chassis_geom
        self._num_boxes = num_boxes

        self.home_scene = False   # True：家居场景回放（不摆随机墙，验证泛化）
        self.laser_front = None   # 由 env 注入 LineLaser 实例
        self.laser_left = None

        self.iteration_count = np.inf
        self.robot_init = (0.0, 0.0, 0.05, 0.0)
        self.sensor_noise = 0.0
        self.course = []          # [(center_xy, len, height, yaw)] 供可视化
        self.gaps = []            # [(near_jamb_xy, dir_unit, width)] 墙体缺口（真值）
        self.bridging = False     # 当前是否正在直行桥接某个缺口

        # 运行时状态
        self.prev_xy = np.zeros(2)
        self.world_vel_xy = np.zeros(2)
        self.yaw_rate = 0.0
        self.front_read = None
        self.left_read = None
        self.lost_time = 0.0
        self.contact_time = 0.0
        self.acquired = False
        self.contact_belly = 0.0
        self.contact_lwheel = 0.0
        self.contact_rwheel = 0.0

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
        return pos, (roll, pitch, yaw), R[2, 2]

    def _gap_bridging(self, pos_xy, yaw):
        """机器人是否正沿某个墙体缺口"直行桥接"（前方有共线墙可续贴）。

        用课程真值判定，区别于"外角"：外角处前方【没有】共线墙，返回 False，
        走原来的包边逻辑；缺口处两端墙共线，机器人在缺口跨度内、贴着原墙线、
        朝墙向直行时返回 True。判据全部满足才算桥接：
          - 航向与墙向同向（cos > GAP_BRIDGE_HEAD）—— 是在直行穿越、不是拐进缺口；
          - 沿墙向坐标落在缺口跨度内（含两端小裕量）—— 缺口内且对侧墙就在前方；
          - 横向仍贴在原墙线一侧、偏离不超过 GAP_BRIDGE_LAT —— 没有跑偏离墙。
        """
        if not self.gaps:
            return False
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        for gp, d, g in self.gaps:
            rel = np.asarray(pos_xy) - gp
            along = float(rel.dot(d))
            normal = np.array([-d[1], d[0]])          # 指向远离机器人的一侧
            lateral = float(rel.dot(normal))          # 机器人在 -normal 侧 -> 负
            if (float(fwd.dot(d)) > self.GAP_BRIDGE_HEAD
                    and -0.10 <= along <= g + 0.12
                    and -self.GAP_BRIDGE_LAT <= lateral <= 0.02):
                return True
        return False

    def _chassis_contact(self):
        """机身壳体与任何物体接触 —— 蹭墙/顶死的信号。"""
        model = self._client.model
        data = self._client.data
        gid = model.geom(self._chassis_geom_name).id
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # 每个控制步推进任务状态（env.step 在 robot.step 之后调用）
    # ------------------------------------------------------------------
    def step(self):
        self.contact_belly = self._chassis_contact()
        self.contact_lwheel = 1.0 if len(self._client.get_lfoot_floor_contacts()) > 0 else 0.0
        self.contact_rwheel = 1.0 if len(self._client.get_rfoot_floor_contacts()) > 0 else 0.0

        cur_xy = self._base_xy()
        self.world_vel_xy = (cur_xy - self.prev_xy) / self._control_dt
        self.prev_xy = cur_xy

        # 缺口桥接判定（真值几何）：随机墙课程才有缺口；家居回放 gaps 为空。
        # calc_reward 与 done 都用它，故在这里算一次并缓存。
        _, (_, _, yaw_b), _ = self._base_pose()
        self.bridging = self._gap_bridging(cur_xy, yaw_b)

        # 机体偏航角速度（局部系 z），用于沿墙抖动抑制。
        # 注：也试过用偏航角"加速度"度量摇晃，实测被接触/求解噪声淹没
        # （规则控制器 |yaw_accel| 均值已达 8.6 rad/s²，注入振荡后仅 1.1 倍，
        # 无判别力）；|yaw_rate| 在直行门控下才是有效判别量（注入振荡 1.5~2.0 倍）。
        _, ang_local = self._client.get_body_vel(self._root_body_name, frame=1)
        self.yaw_rate = float(ang_local[2])

        self.front_read = self.laser_front.read()
        self.left_read = self.laser_left.read()

        if self.left_read.hit:
            self.lost_time = 0.0
            self.acquired = True
        else:
            self.lost_time += self._control_dt

        if self.contact_belly > 0.5:
            self.contact_time += self._control_dt
        else:
            self.contact_time = 0.0
        return

    def substep(self):
        pass

    # ------------------------------------------------------------------
    # 奖励
    # ------------------------------------------------------------------
    def calc_reward(self, prev_torque, prev_action, action):
        pos, (roll, pitch, yaw), up_z = self._base_pose()
        left = self.left_read
        front = self.front_read
        front_clear = (front is None) or (not front.hit)

        # 1) 贴边：左侧距离 -> 1cm（指数整形，8mm 尺度）
        if left.hit:
            err = abs(left.distance - self.TARGET_DIST)
            track = np.exp(-err / 0.008)
            lost = 0.0
        else:
            track = 0.0
            lost = 1.0

        # 2) 沿墙推进：机体前向速度，用贴边质量门控。
        #    贴得准（track->1）才能拿满速度分；斜压在墙上滑行（track~0.3）
        #    只能拿 ~一半，再叠加蹭墙重罚后净收益为负，封死局部最优。
        fwd_vel = float(self.world_vel_xy[0] * np.cos(yaw) +
                        self.world_vel_xy[1] * np.sin(yaw))
        # 速度分改为在 V_DES 处封顶的"帐篷"形：fwd=V_DES 拿满分、超速线性扣回、
        # 2×V_DES 归零、更快为负。旧 clip 形在 V_DES 饱和、超速零代价，实测
        # 策略沿墙巡航到 2×V_DES（0.24~0.34m/s）：10cm 量程激光只剩 <0.3s
        # 反应时间，贴边振荡/丢墙/撞墙全被放大，也让"高速螺旋找墙"净收益为正。
        # fwd<=V_DES 一侧与旧公式完全一致，不影响刹车/慢速接近的既有梯度。
        progress = np.clip(min(fwd_vel, 2.0 * self.V_DES - fwd_vel) / self.V_DES,
                           -1.0, 1.0)
        seek_blind = False    # 盲区巡航找墙状态（首次找墙 / 丢墙重找），联动 wobble
        if left.hit:
            progress *= (0.25 + 0.75 * track)
        elif self.bridging:
            # 墙体缺口（门洞/断墙）：前方有共线墙可续贴 —— 鼓励【全速直行桥接】、
            # 几乎不罚丢墙，让机器人匀速穿过缺口到对侧重新贴墙，而不是中途减速、
            # 原地转向、重新找墙。与外角包边用真值区分（外角前方没有共线墙）。
            progress *= 1.0
            lost *= 0.15
        elif not self.acquired:
            # 找墙巡航（seek）：本回合还从未捕获过墙、激光全盲。给径直巡航
            # 明显高于旧值 0.2 的速度分，配合下方 wobble 的 seek 门控惩罚
            # 原地打转 —— 家居场景出生后原地转圈的根因是盲区里"转圈"与
            # "直行"的奖励几乎无差（转圈不吃任何罚、直行只多 0.2 档速度分）。
            progress *= 0.5
            seek_blind = True
        elif self.lost_time < self.WRAP_GRACE_TIME and front_clear:
            # 阳角（外角）包边窗口：左墙"刚"丢 + 正前方无障碍 —— 典型外角信号。
            # 此时鼓励【先直行越过墙角】、并减轻丢墙焦虑，避免策略急着左勾
            # 把外露的墙角勾进机身左前缘（"卡进机身"）。越过墙角后再转左找墙。
            progress *= 0.6
            lost *= 0.3
        else:
            # 丢墙重找：捕获过墙、但丢墙已超过包边宽限 —— 与首次找墙（seek）
            # 同治：径直巡航找墙 + 防打转。旧版此分支 progress×0.2 且 wobble
            # 门控为 0，丢墙后打转零代价 —— 家居场景"沿墙一段后丢墙 → 以
            # 3~4rad/s 高速打转甩出"的直接来源（训练课程里丢墙段短、分布内
            # 总能很快重捕获，把这个洞掩盖了）。
            progress *= 0.5
            seek_blind = True

        # 3) 前向安全（"距墙 1cm 停住"）：把它写成一条【由前向距离决定的前向限速】——
        #    正前方障碍在 FRONT_STOP_GAP(1cm) 处允许前向速度=0、在 FRONT_SAFE_DIST(9cm)
        #    及以外允许满速 V_DES，中间线性。只惩罚"超过该限速的部分"(overspeed)：
        #    慢速小心接近 / 内角减速都在限速内、不被误伤，唯有"冲着墙加速"才吃罚，
        #    梯度直接指向"贴到 1cm 前把速度降到 0"。置信度加权：矮墙只挡部分射线、
        #    看不真切时罚得轻。没有这一项前向激光对策略就是无用输入，家居场景里会
        #    先直冲撞墙再沿墙。
        if front is not None and front.hit:
            span = self.FRONT_SAFE_DIST - self.FRONT_STOP_GAP
            v_cap = self.V_DES * np.clip((front.distance - self.FRONT_STOP_GAP) / span,
                                         0.0, 1.0)
            overspeed = max(fwd_vel - v_cap, 0.0) / self.V_DES
            approach = front.confidence * overspeed
        else:
            approach = 0.0

        # 4) 保持水平
        upright = np.exp(-(4.0 * roll * roll + 1.5 * pitch * pitch))

        # 5) 蹭墙重罚（贴边 1cm 但机身不能接触；持续接触还会被终止）
        scrape = self.contact_belly

        # 6) 直行段抖动抑制：贴边良好（track->1）且前方无障碍时，惩罚机体偏航
        #    角速度，压住"沿墙一直摇晃"的极限环。拐角需要转向，用 track 与
        #    前方无障碍双重门控 —— 只在本该直行时才罚，避免误伤正常拐弯。
        #    尺度 YAW_RATE_SCALE 由实测分布标定（见 CHANGELOG）：规则控制器
        #    只被罚掉 track 收益的 ~7%，摇晃版本被罚 2.8 倍，且远低于丢墙罚
        #    0.05/步 —— 策略不会为躲这项而放弃贴墙。
        #    桥接缺口时左侧无读数（track=0），但仍要压住偏航防止拐进缺口，
        #    故此时用满门控（steady=1）惩罚偏航，逼机器人直行穿过。
        if not front_clear:
            steady, wob_cap = 0.0, 1.0
        elif left.hit:
            steady, wob_cap = track, 1.0
        elif self.bridging:
            steady, wob_cap = 1.0, 1.0
        elif seek_blind:
            # 盲区巡航（首次找墙 / 丢墙重找）防打转：全盲时旋转对 10cm 量程
            # 的激光毫无侦察价值，惩罚偏航角速度，让"径直巡航"成为唯一高收益
            # 行为。门控 0.6：轻微弧线巡航（|yaw_rate|~0.25rad/s）罚 ~0.01/步、
            # 几乎无感。饱和上限放宽到 4.0 —— 旧上限 1.0 在 0.6rad/s 就饱和：
            # 实测重训后的策略以 2.5~3rad/s 螺旋打转，罚封顶只有 -0.06/步、
            # 被螺旋推进的 progress +0.13 净淹没，且饱和区对"转多快"零梯度，
            # 防打转形同虚设。现在 1.2rad/s 才饱和、打转封顶罚 -0.24/步，
            # 配合帐篷速度分（超速为负），"直行巡航"严格占优。
            # 正对墙刹停后的原地转向对齐不受影响 —— 那时 front 有回波，
            # 已被第一个分支放开。
            steady, wob_cap = 0.6, 4.0
        else:
            steady, wob_cap = 0.0, 1.0
        wobble = steady * min((self.yaw_rate / self.YAW_RATE_SCALE) ** 2, wob_cap)

        # 7) 能耗 / 平滑
        energy = np.mean(np.square(prev_torque)) / (self.MAX_TAU ** 2)
        dact = action - (prev_action if prev_action is not None else action)
        smooth = np.mean(np.square(dact))

        reward = dict(
            track=0.28 * track,
            progress=0.26 * progress,
            upright=0.05 * upright,
            lost=0.05 * (-lost),
            scrape=0.20 * (-scrape),
            approach=0.22 * (-approach),
            wobble=0.10 * (-wobble),
            energy=0.04 * (-energy),
            smooth=0.06 * (-smooth),
        )
        return reward

    # ------------------------------------------------------------------
    # 终止条件
    # ------------------------------------------------------------------
    def done(self):
        pos, (roll, pitch, yaw), up_z = self._base_pose()
        lost_limit = self.ACQUIRE_GRACE if not self.acquired else self.LOST_TERM_TIME
        if self.home_scene:
            # 家居回放宽限 ×4（原 ×2）：从基站到隔断的全盲直行 ~2.4m，按
            # V_DES 0.15m/s 要 ~16s，×2（16s）会在即将到墙前误杀回合。
            lost_limit *= 4.0
        conditions = {
            "flipped": (up_z < 0.4) or (abs(roll) > 1.0) or (abs(pitch) > 1.2),
            "launched": (pos[2] > 0.45) or (pos[2] < -0.05),
            "out_of_bounds": (abs(pos[0]) > 6.0) or (abs(pos[1]) > 6.0),
            "lost_wall": (self.lost_time > lost_limit) and (not self.bridging),
            "scraping": self.contact_time > self.SCRAPE_TERM_TIME,
        }
        return True in conditions.values()

    # ------------------------------------------------------------------
    # 重置：随机墙面课程 + 初始位姿 + 域随机化
    # ------------------------------------------------------------------
    def reset(self, iter_count=0):
        self.iteration_count = iter_count
        frac = float(np.clip(iter_count / self.CURRICULUM_ITRS, 0.0, 1.0))

        self.lost_time = 0.0
        self.contact_time = 0.0
        self.acquired = False
        self.world_vel_xy = np.zeros(2)
        self.yaw_rate = 0.0
        self.front_read = None
        self.left_read = None
        self.gaps = []
        self.bridging = False

        # 传感器噪声课程：0 -> 2mm
        self.sensor_noise = 0.002 * frac

        # 地面摩擦随机化（后期开启）
        if frac > 0.3:
            self._client.model.geom('floor').friction[0] = np.random.uniform(0.8, 1.25)

        if self.home_scene:
            self._bury_all_boxes()
            self.course = []
            # 出生在回充基站正前方、紧贴 8mm 底板前缘（底板前伸到 x=-0.835，
            # 机身后缘留 2cm：x = -0.835 + 0.175 + 0.02）。
            # 不能骑在底板上出生：底板宽 26cm < 轮距 30cm，机器人在底板上时
            # 前/后万向球压板把机身架起、两驱动轮悬空失去抓地 —— 实测满力矩
            # 前进/原地转 2s 位移与偏航都是 0（高位架空死局），这正是"出生后
            # 困在基站附近几秒、然后暴力甩出"的根因。
            # 朝向 +x 背离基站（基站靠西墙）驶出，±0.15rad 微扰覆盖真机出站
            # 的朝向误差；出生即刻就能满抓地直行找墙。
            self.robot_init = (-0.64, 0.70, 0.05, np.random.uniform(-0.15, 0.15))
        else:
            self._build_random_course(frac)

        self.prev_xy = np.array(self.robot_init[0:2])
        return

    def _build_random_course(self, frac):
        """用 mocap 墙块摆一条随机折线墙：机器人从原点朝 +x 出发，墙在左侧。

        墙块尺寸固定（gen_xml.WALL_BOX_SPECS，三种长度各两块），段长由所选
        墙块决定；不同"有效高度"通过把墙块下沉进地面实现。
        """
        self._bury_all_boxes()
        self.course = []

        n_seg = 1 + np.random.randint(0, 2 + int(round(3 * frac)))   # 1~2 -> 1~5
        n_seg = min(n_seg, self._num_boxes)

        # 出生模式：大部分贴墙起步；一部分远离墙起步（找墙课程）。
        # 远离墙起步里再分出"正对墙面"的头对头模式：机器人朝墙（+y）出发，
        # 必须用前向激光减速并转向对齐后再沿墙，专门训练"接近墙面先刹车"
        # ——这是家居场景"直接撞墙再沿墙"的针对性纠正。
        far_spawn = (frac > 0.1) and (np.random.rand() < 0.35)
        head_on = far_spawn and (np.random.rand() < 0.50)
        # 全盲出生占比 0.40 -> 0.60（约 10.5% 的回合）：家居场景从基站出发
        # 就是全盲状态，且丢墙重找（新奖励分支）复用同一巡航技能，加大采样。
        blind = far_spawn and (not head_on) and (np.random.rand() < 0.60)
        if far_spawn:
            if head_on:
                # yaw≈+90° 正对左侧墙面（更宽的偏航，覆盖更多斜向接近）。必须靠
                # 前向激光把速度降到 0、停在墙前 ~1cm 再转向对齐，而不是撞上去。
                # d0 从 8cm 起：前向激光量程 10cm，8~10cm 出生的回合一睁眼就看到墙、
                # 必须立刻刹车（最强的刹车梯度）；更远的先巡航、进 10cm 再刹。
                d0 = np.random.uniform(0.08, 0.35)
                start_yaw = np.pi / 2.0 + np.random.uniform(-0.6, 0.5)
            elif blind:
                # 全盲找墙（家居场景出生分布）：任意朝向、更远距离，激光量程内
                # 长时间无任何回波。期望行为：不打转、径直巡航（seek 速度分 +
                # 防打转罚），直到某个方向撞进 10cm 量程再刹车/贴边。朝向背离
                # 墙的回合会被 ACQUIRE_GRACE 超时截断 —— 全盲下本就不存在
                # "更聪明"的策略，学到"睁眼一片黑就直走"即达标。
                # 上限 0.60 -> 1.00m：家居从基站到隔断要全盲直行 ~2.4m，训练
                # 里必须见过"长时间全盲仍坚持直行"的时段（V_DES 0.15m/s 下
                # 1m ≈ 7s，逼近 8s 找墙宽限，更远的朝向差回合自然被截断）。
                d0 = np.random.uniform(0.20, 1.00)
                start_yaw = np.random.uniform(-np.pi, np.pi)
            else:
                d0 = np.random.uniform(0.12, 0.45)
                start_yaw = np.random.uniform(-0.5, 0.5)
        else:
            d0 = np.random.uniform(0.02, 0.04 + 0.05 * frac)
            start_yaw = np.random.uniform(-1.0, 1.0) * (0.10 + 0.25 * frac)

        # 墙块顺序：首段用长块（保证出生点在墙侧旁），其余随机
        model = self._client.model
        names = ['box' + repr(i + 1).zfill(2) for i in range(self._num_boxes)]
        first = names.pop(np.random.randint(2))     # box01/box02 是长块
        np.random.shuffle(names)
        order = [first] + names

        # 缺口课程：热身后（frac>0.15）才在共线段之间插缺口，宽度随课程展开
        # 到 0.35m 封顶（见 GAP_MAX_BASE 注释：必须与门洞宽度拉开差距）。
        gaps_on = frac > 0.15
        gap_max = self.GAP_MAX_BASE + 0.15 * frac
        # 墙端 U 型包边课程：拐角热身后（frac>0.25）开启。
        uturn_on = frac > 0.25

        p = np.array([-0.4, 0.175 + d0])    # 墙内侧面折线起点（机器人左侧）
        d = np.array([1.0, 0.0])            # 墙走向
        prev_gap = False                    # 上一段边界是否为缺口（共线，不回收）
        for i in range(n_seg):
            name = order[i]
            hx, ht, hz = model.geom(name).size    # 半长 / 半厚 / 半高
            # 拐角处向回收 10cm，让相邻墙段在角上搭接、不留缝；缺口后是共线续墙，
            # 不能回收（否则会把缺口吃掉甚至叠上），按缺口真实起点摆放
            seg_start = p if (i == 0 or prev_gap) else p - d * 0.10
            L = 2.0 * hx

            # 折返回起点附近的墙段会压到机器人出生点，从这段起截断课程
            if i > 0 and self._segment_near_origin(seg_start, d, L):
                break

            # 有效墙高：60% 可靠高度（>=10cm），后期 40% 低置信高度（4.5~7cm）
            if frac < 0.3 or np.random.rand() < 0.6:
                h = np.random.uniform(0.10, 2.0 * hz)
            else:
                h = np.random.uniform(0.045, 0.07)

            mid = seg_start + d * (L / 2.0)
            normal = np.array([-d[1], d[0]])          # 指向远离机器人一侧
            center = mid + normal * ht
            seg_yaw = float(np.arctan2(d[1], d[0]))
            self._mocap_place(name, [center[0], center[1], h - hz], seg_yaw)
            self.course.append((center.copy(), L, h, seg_yaw))

            # ---- 段间过渡：缺口（共线续墙）/ 墙端 U 型包边 / 拐角，三选一 ----
            p = seg_start + d * L
            prev_gap = False
            if gaps_on and i < n_seg - 1 and np.random.rand() < self.GAP_PROB:
                # 墙体缺口：保持墙向不变，跨过一段无墙缺口后续墙。记录真值供奖励。
                g = float(np.random.uniform(self.GAP_MIN, gap_max))
                self.gaps.append((p.copy(), d.copy(), g))
                p = p + d * g
                prev_gap = True
            elif uturn_on and i < n_seg - 1 and np.random.rand() < self.U_TURN_PROB:
                # 墙端 U 型包边（门洞侧柱课程）：下一段贴着同一堵墙的【背面】
                # 反向延伸、端面与本段对齐（跨过 2×半厚），机器人到墙端后需
                # 连做两个左外角（合计≈180°）绕过墙端、再沿背面继续贴边 ——
                # 正是家居场景"沿隔断走到门洞侧柱、拐进里屋"的动作。
                # 训练墙厚 16cm、家居隔断 5cm，包边窗口逻辑一致，可泛化。
                p = p + normal * (2.0 * ht)
                d = -d
                prev_gap = True   # 端面已对齐，下一段不做拐角回收（同缺口逻辑）
            else:
                # 拐角：前期固定 90°，后期 55°~115°；负角=内角（墙折向机器人前方）
                ang = np.pi / 2 if frac < 0.3 else np.radians(np.random.uniform(55.0, 115.0))
                if np.random.rand() < 0.5:
                    ang = -ang
                c, s = np.cos(ang), np.sin(ang)
                d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

        self.robot_init = (0.0, 0.0, 0.05, start_yaw)

    def _segment_near_origin(self, p, d, L, safe_r=0.30):
        """墙段（线段 p -> p+d*L）到机器人出生点(原点)的距离是否过近。"""
        t = float(np.clip(-p.dot(d), 0.0, L))
        closest = p + d * t
        return float(np.linalg.norm(closest)) < safe_r + self.WALL_HALF_T

    # ------------------------------------------------------------------
    # mocap 地形操作
    # ------------------------------------------------------------------
    def _bury_all_boxes(self):
        """把所有 mocap 地形（墙块+门槛）埋回地下。"""
        model = self._client.model
        data = self._client.data
        for b in range(model.nbody):
            mid = model.body_mocapid[b]
            if mid >= 0:
                data.mocap_pos[mid] = np.array([0.0, 0.0, -2.0])
                data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])

    def _mocap_place(self, name, pos3, yaw=0.0):
        model = self._client.model
        data = self._client.data
        mid = model.body_mocapid[model.body(name).id]
        data.mocap_pos[mid] = np.asarray(pos3, dtype=np.float64)
        data.mocap_quat[mid] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
