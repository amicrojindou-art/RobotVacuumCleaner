"""扫地机器人"沿边沿墙清扫"RL 任务。

目标：只依靠两个线激光（右侧 + 正前方）与本体感知，学会以 1cm 横向距离
贴墙行进，并泛化到任意墙形（内角/外角/斜墙/不同高度的墙沿）。
侧边激光与真机一致装在【右侧】：沿墙时墙在机器人右侧，绕房间逆时针巡边。

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
  - 阳角（外角）包边窗口：右墙"刚"丢 + 前方无障碍时，鼓励先直行越过墙角
    再右转找墙，避免急着右勾把外露墙角勾进机身右前缘（"卡进机身"）；
  - 阳角【虚拟贴边分】（0715_1 修复，照搬缺口桥接的成功做法）：包边圆弧
    途中侧激光必然无回波，track=0 —— "紧贴墙角转弯"全程零收入，唯一的
    回报是包完之后的贴墙收入，信号太远太稀。0714~0715_1 四次从头重训全部
    学成"胆小浅弧"（丢墙后右转方向对、但 2s 只转 ~13°，够不到 10cm 量程，
    终被丢墙超时终止 —— 回合平均 ~350 步就死、总回报塌掉的直接原因；
    0713_1 能学会属于幸运收敛）。修复：训练时阳角顶点是课程真值
    （self.wrap_corners），侧激光丢墙且在顶点 0.45m 内时，按贴墙同一公式
    对"机身中心到顶点距离 = 半径+1cm"发虚拟贴边分 —— 紧贴包边圆弧
    全程拿 track，与贴直墙连续，"紧贴转弯"有了直接正梯度。包边中转向
    免罚（wobble steady=0，含超过 1.2s 宽限的慢包边 —— 旧版宽限一过，
    包边圆弧的 1.5rad/s 转向会掉进 seek 防打转罚里挨 -0.24/步）。
    家居回放无课程真值，该分支自动关闭（同缺口桥接）；
  - 阳角速成课程（corner drill，30% 回合）：首段用短墙（0.9m）且强制
    第一个过渡为拐角、70% 是阳角 —— 出生 ~2s 内就进入包边练习，
    从课程第 0 迭代开始密集采样（此前包边样本只能等课程随机撒）；
  - wobble（直行段抖动抑制）：贴边良好且前方无障碍时惩罚偏航角速度，
    压住"沿墙一直摇晃"的极限环；用 track + 前方无障碍双重门控，不误伤拐弯；
  - 缺口桥接（门洞/断墙）：连续墙上留出一段两端共线、中间无墙的缺口，
    机器人抵达时右侧激光会丢墙。此刻若前方仍有共线墙可续贴（真值判定），
    就【保持直行匀速穿过缺口】、几乎不罚丢墙，并压住偏航（别拐进缺口）；
    到对侧自然重新贴墙。与"外角包边"用真值区分：外角前方没有共线墙，
    仍走原来的包边逻辑。没有这一项，策略把缺口误当外角/丢墙 —— 中途减速、
    原地转向、重新找墙，沿边作业被打断。缺口宽度上限压在 0.35m：必须明显
    小于家居门洞宽度（0.6~0.9m），门洞才会被判成"外角拐入"而不是被直行
    桥接跳过；
  - 贴边分"在干活才发"（防缺口往复 / 内角驻留两类白拿 track 的局部最优）：
    0713 版实测暴露同族两个坑 —— ① 缺口处：一越过近端激光丢墙，track 从
    ~0.28 突然掉 0（奖励悬崖），而满幅倒车半步就能重新拿回激光回波，倒车段
    track(+0.28) 与倒车 progress 罚(-0.26) 几乎相抵、净亏≈0，"前进丢墙→
    倒车找回→再前进"的往复极限环收益反超直行桥接，策略在缺口边缘困死；
    ② 内角处：刹停后驻留（fwd≈0、不转向）每步白拿 track+upright≈0.33 且
    零支出（wobble 被 front.hit 门控关掉、approach 不动不罚），9/9 个抵达
    内角的回合全部停死在角落、毫无转向动机。三手修复：
      a) 桥接时用课程真值把原墙线"延长"进缺口，按贴墙同一公式发
         【虚拟贴边分】—— track 与 progress 门控跨缺口完全连续，
         直行穿越与贴墙直行等值，奖励悬崖消失；
      b) 倒车（fwd<0）按比例扣发贴边分、-0.06m/s 起全扣 —— 掐掉往复
         循环唯一的收入来源；墙端/外角/U 型包边处的同构往复循环一并封死；
      c) 驻留不发贴边分：track 乘运动门控 max(前进/0.5·V_DES, |yaw_rate|/0.5)
         —— 正常巡航与内角原地转向都拿满、停死拿 0，"转向离开角落"
         成为内角处唯一的高收益动作。内角刹停转向的微小倒冲(|fwd|<0.03)
         只扣一半以内，不误伤；
  - 找墙巡航（seek）防打转：激光全盲时（首次找墙、或捕获后丢墙超过包边
    宽限的"丢墙重找"——两者同治）给径直巡航更高的速度分、并惩罚偏航
    角速度（盲区专用的更高饱和上限，保证高速打转仍有梯度可下降）。
    10cm 量程下原地旋转毫无侦察价值，家居场景"出生打转 / 丢墙后打转
    甩出"的根因就是盲区里"转圈"与"直行"奖励几乎无差、且旧防打转罚
    在 0.6rad/s 就饱和（实测策略以 2.5~3rad/s 打转，罚被速度分淹没）；
  - 墙端 U 型包边（门洞侧柱）课程：部分段间过渡改为"下一段贴同一堵墙的
    背面反向延伸、端面对齐"，机器人到墙端后需连做两个右外角（合计≈180°）
    绕过墙端、再沿背面继续贴边 —— 对应家居场景"沿隔断走到门洞侧柱、
    拐进里屋继续沿墙"。此前课程只有 55°~115° 单拐角，策略在门洞侧柱处
    不会包边、原地打转甩出。
  - 浅凸起（壁柱/薄墙端面/基站座体）：贴边巡航时前向激光射线距墙 18.5cm，
    凸出量小于它的障碍接触前【双激光零预警】（几何盲区，训练墙厚 16cm 的
    U 型端面也在内），只能靠碰撞感知。处置 = 【纯奖励层】：
      a) press 顶推罚：正面锥（±65°）内接触 + 还在前进 -> 罚（保险杠的
         "刹车老师"，不碰激光通道）；
      b) 接触期转向减税：wobble 在机身接触时打 3 折（折扣上限 0.175 <
         scrape 0.20，"故意碰墙换转向自由"必然亏），教"撞上就转身脱离"；
      c) 观测里的接触方位（+1左/-1右）告诉策略撞的是哪边肩；
      d) 凸块课程（PROT_PROB=0.15）供练习。
    ——【重要教训，禁止再犯】：不要把碰撞注入前向激光观测通道。三次
    重训（0714/0714_1/0715）先后试过 "碰撞期豁免 wobble 0.6s"、"接触期
    注入 1cm 虚拟回波（±90° 触发）"、"虚拟墙记忆（±65° 锥触发）"，
    全部失败，且第三次证明失败与触发锥宽窄无关：紧贴包边必然偶尔蹭到
    墙角，而【蹭角接触的方位角会随右转包边自然迁移进任何触发锥】
    （墙角点在机体系里从 -90° 向 0° 移动），触发后观测变"正前 1cm 有墙"，
    训练成熟的 head_on/内角反射（刹停+左转）恰好把机器人推离墙 ——
    "紧贴阳角包边"与"碰撞=正面有墙"在原理上不相容。定量证据（0715 回放）：
    新策略普通回合阳角包边 0/27；冻结 0713_1 在注入环境 24/40、关掉注入
    后恢复 27/33（回合长 431->561）。碰撞信息只允许走"接触方位观测 +
    奖励项"，前向激光通道必须保持纯净。

终止：翻车 / 弹飞 / 出界 / 持续蹭墙 / 丢墙超时（首次捕获前有更长宽限；
      正在桥接缺口时不按丢墙终止）。
"""

import numpy as np
import transforms3d as tf3


class WallFollowTask(object):
    # ---- 任务超参数 ----
    TARGET_DIST = 0.01       # 贴边目标：机身右边缘距墙 1cm
    # 期望沿墙/巡航速度 (m/s)。帐篷速度分的峰值就是训练出的巡航速度：
    # 0.15 版重训后实测偏慢（用户要求保持 0.24~0.34 巡航带），提到 0.28 ——
    # 带宽 0.24~0.32 内速度分 >=0.86，峰值居中；超过 0.34 开始明显扣分、
    # 2×V_DES=0.56 归零。代价是 10cm 量程下反应时间更短（~0.33s），贴边
    # 振荡与拐角冲过量会比 0.15 版大，由 wobble/scrape/approach 项约束。
    V_DES = 0.28
    MAX_TAU = 2.0            # 力矩上限（归一化用，与 XML 一致）
    FRONT_SAFE_DIST = 0.09   # 正前方开始减速的距离（≈激光量程；防撞墙）
    FRONT_STOP_GAP  = 0.01   # 期望停墙间距：正前方 1cm 处前向速度应降到 0
    YAW_RATE_SCALE = 0.6     # 抖动抑制归一化尺度 (rad/s)，由实测 yaw_rate 分布标定
    # 贴墙直行分支的专用抖动尺度/上限：用户要求沿墙【完全不能摇摆】。
    # 0708 版重训后贴墙残留 |yaw_rate|≈0.6 的左右摇摆（spin%~30）——旧参数
    # （尺度 0.6、上限 1.0）下该振荡每步只罚 0.10×track，相对 track+progress
    # 收益 ~0.54 太弱，策略停在"摇着走"的极限环上。现贴墙分支尺度收紧到
    # 0.35 rad/s、饱和上限放宽到 2.5：0.6rad/s 摇摆罚 -0.25/步（决定性劣势），
    # 而贴墙微调（|yaw_rate|<=0.15，直墙跟踪的真实需求）只罚 <0.02/步、
    # 不受影响。拐角不受影响：内角时 front 有回波（门控=0）、外角时侧墙
    # 已丢（走包边分支）——本尺度只作用于"贴边良好的直行段"。
    TRACK_YAW_SCALE = 0.35
    TRACK_WOB_CAP = 2.5
    # 阳角包边宽限窗口 (s)：右侧激光装在机体横向中心线(x=0)上，丢墙那一刻墙角
    # 正好在机身正侧方，要把墙角甩到机身后方需再前进约一个底盘半径
    # （0.175m，V_DES=0.28 下 ≈0.63s；窗口 1.2s 留 ~2 倍余量，覆盖拐角处
    # 减速的情况）。窗口太短会逼策略提前右勾、勾住墙角；窗口只是"不重罚"
    # 的上限，不强迫直行满窗口。
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
    CHASSIS_RADIUS = 0.175   # 底盘半径 (m)，与 gen_xml 一致；桥接虚拟贴边分用
    # 浅凸起（壁柱/薄墙端面/基站座体等，凸出墙面 < 18.5cm）课程与碰撞转向：
    # 贴边巡航时前向激光射线距墙 = 底盘半径 0.175 + 贴边 0.01 = 18.5cm，
    # 凸出量小于它的障碍从射线外侧掠过、接触前【双激光零预警】（侧激光在
    # 横向中心 x=0，首触点在右前肩 x≈+0.12，首触时侧激光还在凸块上游）——
    # 只能靠碰撞感知。0713_1 实测撞凸块后来回碰撞 28~56 次、绕过要 7~11s。
    # 凸块课程密度压到 0.15（0714/0715 曾用 0.25~0.35：必撞回合太密，
    # 碰撞卫生与包边行为都被带坏），且推迟到 frac>0.6 才开启（0715_1 教训：
    # 凸块回合里"果断右转贴回墙"会随机撞上盲区凸块吃罚，包边技能在成形期
    # 就被毒害 —— 对照：唯一没有凸块课程的 0713_1 包边 30/39，四个带凸块
    # 课程的从头重训全部 <10/35。先固化包边，再学撞凸块的处置）
    PROT_PROB = 0.15         # 每条课程放一个盲区障碍的概率（frac>0.6 开启）
                             # 2026-07-17 置 0：在 0715_2 基础上精修包边
                             # （_3 定案凸块毒化包边），凸块处置留待单独攻关。
                             # 2026-07-20 恢复 0.15：_2c 精修完成、包边/阳角
                             # 已固化（实测流畅），按既定路线开启盲区障碍攻关；
                             # 并新增"薄墙拦路窄缝"变体（见 _build_random_course），
                             # 覆盖 _2c 实测空洞：正前被薄墙堵死 + 前激光穿缝 +
                             # 侧墙仍在 -> 双转弯反射都不触发、原地往复极限环。
    # 阳角包边虚拟贴边分：搜索包边顶点的半径（包边圆弧半径 0.185 + 裕量）
    WRAP_CORNER_RANGE = 0.45
    # 阳角速成课程占比：首段短墙 + 强制首过渡为拐角（70% 阳角）
    CORNER_DRILL_PROB = 0.30
    PROT_JUT_MIN = 0.02      # 凸出量下限 (m)
    PROT_JUT_MAX = 0.12      # 凸出量上限 (m，随课程渐进），= ridge02 横向半宽 ×2
    # 保险杠正面锥（正前 ±65°）：接触点方位角在锥内才算"正面顶推"，
    # 只用于 press 奖励项（顶着障碍还前进 -> 罚），【不注入任何观测】。
    # 几何依据（贴边 1cm、底盘半径 0.175）：凸块首触方位角 =
    # asin((0.185-jut)/0.175) —— 凸出 12cm ≈ ±22°、6cm ≈ ±46°、3cm ≈ ±62°；
    # 平墙侧蹭 / 阳角包边蹭角 ≈ ±90°，不算顶推（只吃 scrape）。
    BUMP_CONE_HALF_DEG = 65.0
    # 接触期 wobble 折扣：机身接触时转向税打 3 折。全额豁免会开"故意碰墙
    # 换转向自由"的洞（0714 教训）：wobble 最大 0.10×2.5=0.25 > scrape 0.20；
    # 打 3 折后最多省 0.175 < 0.20，碰墙必然净亏，同时"撞上后转身脱离"
    # 的必要转向不再被全额压制（0713_1 来回碰撞的根因之一）。
    CONTACT_WOB_RELIEF = 0.7
    # 倒车贴边分门控：fwd>=0 全额发放、<= -REV_TRACK_SCALE 全扣（线性过渡）。
    # 见文件头"防往复极限环"注释；0.06m/s 取"内角刹停微倒冲(~0.03)只扣一半、
    # 有意倒车(>=0.1) 必然全扣"之间。
    REV_TRACK_SCALE = 0.06
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
        self.laser_right = None   # 侧边激光（与真机一致装在右侧）

        self.iteration_count = np.inf
        self.robot_init = (0.0, 0.0, 0.05, 0.0)
        self.sensor_noise = 0.0
        self.course = []          # [(center_xy, len, height, yaw)] 供可视化
        self.gaps = []            # [(near_jamb_xy, dir_unit, width)] 墙体缺口（真值）
        self.bridging = False     # 当前是否正在直行桥接某个缺口
        self.bridge_lat_err = 0.0 # 桥接时机身右缘到"墙线延长线"1cm 目标的偏差 (m)
        self.wrap_corners = []    # [xy] 阳角顶点（课程真值），包边虚拟贴边分用
        self.wrap_err = None      # 本步包边贴边偏差 |到顶点距离-(半径+1cm)|；无=None

        # 运行时状态
        self.prev_xy = np.zeros(2)
        self.world_vel_xy = np.zeros(2)
        self.yaw_rate = 0.0
        self.front_read = None
        self.side_read = None
        self.lost_time = 0.0
        self.contact_time = 0.0
        self.acquired = False
        self.contact_belly = 0.0
        self.contact_dir = 0.0    # 接触方位 +1左/-1右/0无（观测用）
        self.contact_front = 0.0  # 接触是否在正前 ±65° 保险杠锥内（虚拟墙触发用）
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
            normal = np.array([d[1], -d[0]])          # 指向远离机器人的一侧（墙在右）
            lateral = float(rel.dot(normal))          # 机器人在 -normal 侧 -> 负
            if (float(fwd.dot(d)) > self.GAP_BRIDGE_HEAD
                    and -0.10 <= along <= g + 0.12
                    and -self.GAP_BRIDGE_LAT <= lateral <= 0.02):
                # 虚拟贴边误差：缺口内激光无回波，用真值把原墙内侧面延长进
                # 缺口充当"墙"。-lateral = 机身中心到墙线距离，目标 = 底盘
                # 半径 + 1cm 贴边目标 —— calc_reward 按贴墙同一公式发 track，
                # 让奖励跨缺口连续（否则缺口边缘是奖励悬崖，见文件头注释）。
                self.bridge_lat_err = abs(-lateral - (self.CHASSIS_RADIUS + self.TARGET_DIST))
                return True
        return False

    def _chassis_contact(self):
        """机身壳体与任何物体接触 —— 蹭墙/顶死的信号。

        返回 (是否接触, 接触方位, 是否正面)：
          方位 = 接触点均值在机体系下的横向符号，+1 左 / -1 右（墙侧）/ 0 无；
          正面 = 接触点均值方位角在正前 ±BUMP_CONE_HALF_DEG(65°) 锥内
                 （保险杠正面顶推区）。浅凸起（凸出 < 18.5cm）处于双激光
        盲区、只能靠碰撞感知：方位进观测告诉策略"撞的是哪边肩"，正面标志
        只用于 press 奖励项（顶着还前进 -> 罚），【绝不注入激光观测】——
        见文件头"重要教训"。贴边巡航的常态侧蹭 / 阳角包边蹭角（≈±90°）
        不算顶推，只按 scrape 处理。
        """
        model = self._client.model
        data = self._client.data
        gid = model.geom(self._chassis_geom_name).id
        pts = []
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                pts.append(np.array(c.pos[0:2]))
        if not pts:
            return 0.0, 0.0, 0.0
        pos, (_, _, yaw), _ = self._base_pose()
        rel = np.mean(pts, axis=0) - pos[0:2]
        lat = -np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]   # 机体系 y 分量
        lon = np.cos(yaw) * rel[0] + np.sin(yaw) * rel[1]    # 机体系 x 分量
        # 方位角在正前 ±65° 锥内 <=> lon > |lat| / tan(65°)
        in_cone = lon > abs(lat) / np.tan(np.radians(self.BUMP_CONE_HALF_DEG))
        return 1.0, (1.0 if lat > 0.0 else -1.0), (1.0 if in_cone else 0.0)

    # ------------------------------------------------------------------
    # 每个控制步推进任务状态（env.step 在 robot.step 之后调用）
    # ------------------------------------------------------------------
    def step(self):
        self.contact_belly, self.contact_dir, self.contact_front = self._chassis_contact()
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

        # 激光通道保持纯净：碰撞不注入任何观测（见文件头"重要教训"），
        # 碰撞只通过 接触方位观测 + press/scrape/wobble折扣 三个奖励口生效。
        self.front_read = self.laser_front.read()
        self.side_read = self.laser_right.read()

        # 阳角包边虚拟贴边（课程真值，见文件头）：侧激光丢墙且在某个阳角
        # 顶点 WRAP_CORNER_RANGE 内 -> 计算与"包边圆弧"（到顶点距离 =
        # 半径+1cm）的偏差，calc_reward 按贴墙同一公式发虚拟 track。
        # 家居回放 wrap_corners 为空，恒为 None。
        self.wrap_err = None
        if (not self.side_read.hit) and self.wrap_corners:
            dmin = min(float(np.linalg.norm(cur_xy - c)) for c in self.wrap_corners)
            if dmin < self.WRAP_CORNER_RANGE:
                self.wrap_err = abs(dmin - (self.CHASSIS_RADIUS + self.TARGET_DIST))

        if self.side_read.hit:
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
        side = self.side_read
        front = self.front_read
        front_clear = (front is None) or (not front.hit)

        # 机体前向速度（贴边分的倒车门控与速度分都要用）
        fwd_vel = float(self.world_vel_xy[0] * np.cos(yaw) +
                        self.world_vel_xy[1] * np.sin(yaw))

        # 1) 贴边：右侧距离 -> 1cm（指数整形，8mm 尺度）。
        #    桥接缺口时激光无回波，用课程真值墙线的延长线充当"墙"、按同一
        #    公式发【虚拟贴边分】—— track 跨缺口连续，"直行穿越"与"贴墙
        #    直行"等值，消除缺口边缘的奖励悬崖（防往复极限环，见文件头）。
        if side.hit:
            err = abs(side.distance - self.TARGET_DIST)
            track_raw = np.exp(-err / 0.008)
            lost = 0.0
        elif self.bridging:
            track_raw = np.exp(-self.bridge_lat_err / 0.008)
            lost = 1.0
        elif self.wrap_err is not None:
            # 阳角包边虚拟贴边分（课程真值顶点，见文件头）：紧贴包边圆弧
            # （到顶点距离 = 半径+1cm）全程拿 track，与贴直墙连续 ——
            # "紧贴转弯"有直接正梯度，胆小浅弧/直行飘走拿不到。
            track_raw = np.exp(-self.wrap_err / 0.008)
            lost = 1.0
        else:
            track_raw = 0.0
            lost = 1.0
        # 贴边分必须"在干活"才发（防两类白拿 track 的局部最优，见文件头）：
        #   - 前进：0.5·V_DES 拿满 —— 正常巡航/贴墙直行不受影响；
        #   - 原地转向：0.5rad/s 拿满 —— 内角刹停后的原地转向照发全额
        #     （转向正是内角处的任务本身，不能罚）；
        #   - 驻留（fwd≈0 且不转）→ 0：0713 版实测 9/9 个抵达内角的回合
        #     全部停死在角落（5~16s），驻留每步白拿 track+upright≈0.33、
        #     零支出（wobble 被 front.hit 门控关掉、approach 不动不罚），
        #     策略毫无转向动机 —— 内角"直接暂停"的根因；
        #   - 倒车（<=-REV_TRACK_SCALE）强制归零（转着倒也不行）——
        #     缺口/墙端"倒车找回激光回波"往复极限环的收入来源。
        # 注意 progress 门控与 wobble 门控仍用未扣发的 track_raw：倒车/驻留
        # 时贴边质量本身没变差，速度罚不能因门控变小而被稀释。
        move_gate = max(np.clip(fwd_vel / (0.5 * self.V_DES), 0.0, 1.0),
                        np.clip(abs(self.yaw_rate) / 0.5, 0.0, 1.0))
        rev_gate = np.clip((fwd_vel + self.REV_TRACK_SCALE) / self.REV_TRACK_SCALE,
                           0.0, 1.0)
        track = track_raw * float(move_gate * rev_gate)

        # 2) 沿墙推进：机体前向速度，用贴边质量门控。
        #    贴得准（track->1）才能拿满速度分；斜压在墙上滑行（track~0.3）
        #    只能拿 ~一半，再叠加蹭墙重罚后净收益为负，封死局部最优。
        # 速度分改为在 V_DES 处封顶的"帐篷"形：fwd=V_DES 拿满分、超速线性扣回、
        # 2×V_DES 归零、更快为负。旧 clip 形在 V_DES 饱和、超速零代价，实测
        # 策略巡航到 2× 旧V_DES 也不吃任何罚 —— 速度彻底失控，且让"高速螺旋
        # 找墙"净收益为正。帐篷峰值 = 训练出的巡航速度（现 0.28，见 V_DES 注释）。
        # fwd<=V_DES 一侧与旧公式完全一致，不影响刹车/慢速接近的既有梯度。
        progress = np.clip(min(fwd_vel, 2.0 * self.V_DES - fwd_vel) / self.V_DES,
                           -1.0, 1.0)
        seek_blind = False    # 盲区巡航找墙状态（首次找墙 / 丢墙重找），联动 wobble
        if side.hit:
            progress *= (0.25 + 0.75 * track_raw)
        elif self.bridging:
            # 墙体缺口（门洞/断墙）：前方有共线墙可续贴 —— 鼓励【全速直行桥接】、
            # 几乎不罚丢墙，让机器人匀速穿过缺口到对侧重新贴墙，而不是中途减速、
            # 原地转向、重新找墙。与外角包边用真值区分（外角前方没有共线墙）。
            # progress 门控与贴墙同式（用虚拟 track_raw）：跨缺口的 track 与
            # progress 都连续，且拐进缺口内侧（偏离墙线）拿不满速度分。
            progress *= (0.25 + 0.75 * track_raw)
            lost *= 0.15
        elif not self.acquired:
            # 找墙巡航（seek）：本回合还从未捕获过墙、激光全盲。给径直巡航
            # 明显高于旧值 0.2 的速度分，配合下方 wobble 的 seek 门控惩罚
            # 原地打转 —— 家居场景出生后原地转圈的根因是盲区里"转圈"与
            # "直行"的奖励几乎无差（转圈不吃任何罚、直行只多 0.2 档速度分）。
            progress *= 0.5
            seek_blind = True
        elif self.wrap_err is not None and front_clear:
            # 阳角包边（有课程真值顶点）：progress 门控与贴墙同式（用虚拟
            # track_raw），紧贴圆弧才拿满速度分；丢墙罚同包边窗口减轻。
            progress *= (0.25 + 0.75 * track_raw)
            lost *= 0.3
        elif self.lost_time < self.WRAP_GRACE_TIME and front_clear:
            # 阳角（外角）包边窗口（无真值顶点时的兜底，如刚出包边搜索半径）：
            # 鼓励【先直行越过墙角】、并减轻丢墙焦虑，避免策略急着右勾
            # 把外露的墙角勾进机身右前缘（"卡进机身"）。越过墙角后再转右找墙。
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

        # 5b) press 顶推罚（保险杠"刹车老师"，纯奖励层）：正面锥（±65°）内
        #     接触、侧激光仍读墙（贴边巡航中撞上盲区凸起的特征状态）且还在
        #     前进 -> 罚前向速度。盲区凸起撞上后唯一正确的第一反应是"停止
        #     前推"，此前该教学信号缺失（approach 由前向激光触发，盲区碰撞
        #     时前向无回波）。侧蹭/包边蹭角（≈±90°）不罚；包边中蹭角时侧
        #     激光已丢墙，side.hit 门控保证不误伤（0715_1 教训：包边期任何
        #     额外惩罚都会把策略推向"胆小浅弧"）；倒车脱离不罚（max(fwd,0)）。
        press = (self.contact_belly * self.contact_front *
                 (1.0 if side.hit else 0.0) * max(fwd_vel, 0.0) / self.V_DES)

        # 6) 直行段抖动抑制：贴边良好（track->1）且前方无障碍时，惩罚机体偏航
        #    角速度，压住"沿墙一直摇晃"的极限环。拐角需要转向，用 track 与
        #    前方无障碍双重门控 —— 只在本该直行时才罚，避免误伤正常拐弯。
        #    贴墙分支用收紧的专用尺度 TRACK_YAW_SCALE + 更高上限 TRACK_WOB_CAP
        #    （沿墙完全不能摇摆是硬要求，见常量注释）；直墙跟踪的微调
        #    |yaw_rate|<=0.15rad/s 每步只罚 <0.02，不误伤。
        #    桥接缺口时右侧无读数（track=0），但仍要压住偏航防止拐进缺口，
        #    故此时用满门控（steady=1）惩罚偏航，逼机器人直行穿过。
        if not front_clear:
            steady, yaw_scale, wob_cap = 0.0, self.YAW_RATE_SCALE, 1.0
        elif side.hit:
            steady, yaw_scale, wob_cap = track_raw, self.TRACK_YAW_SCALE, self.TRACK_WOB_CAP
        elif self.bridging:
            steady, yaw_scale, wob_cap = 1.0, self.YAW_RATE_SCALE, 1.0
        elif self.wrap_err is not None and self.acquired:
            # 阳角包边中转向免罚（含超过 1.2s 宽限的慢包边）：包边圆弧本身
            # 就需要 ~1.5rad/s 的持续右转，旧版宽限一过就掉进 seek 防打转罚
            # （-0.24/步）—— 慢包边被罚成"不敢包"。真值顶点在 + 已捕获过墙
            # 才免（全盲找墙阶段在墙端附近仍吃 seek 防打转罚），不开洞。
            steady, yaw_scale, wob_cap = 0.0, self.YAW_RATE_SCALE, 1.0
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
            steady, yaw_scale, wob_cap = 0.6, self.YAW_RATE_SCALE, 4.0
        else:
            steady, yaw_scale, wob_cap = 0.0, self.YAW_RATE_SCALE, 1.0
        # 接触期转向减税（打 3 折）：盲区凸起撞上后需要转身脱离，全额转向
        # 税会把"转身"压成"来回小碎步蹭"（0713_1 来回碰撞根因之一）；
        # 折扣上限 0.25×0.7=0.175 < scrape 0.20，"故意碰墙换转向自由"必亏。
        steady *= (1.0 - self.CONTACT_WOB_RELIEF * self.contact_belly)
        wobble = steady * min((self.yaw_rate / yaw_scale) ** 2, wob_cap)

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
            press=0.12 * (-press),
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
            # 家居回放宽限 ×4：现出生已贴墙（不再有长距离全盲找墙段），
            # 保留放宽是为门洞/家具轮廓等大跨度丢墙段留余量，避免回放中
            # 在包边/穿门洞途中误杀回合。
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
        self.contact_dir = 0.0
        self.contact_front = 0.0
        self.acquired = False
        self.world_vel_xy = np.zeros(2)
        self.yaw_rate = 0.0
        self.front_read = None
        self.side_read = None
        self.gaps = []
        self.bridging = False
        self.bridge_lat_err = 0.0
        self.wrap_corners = []
        self.wrap_err = None

        # 传感器噪声课程：0 -> 2mm
        self.sensor_noise = 0.002 * frac

        # 地面摩擦随机化（后期开启）
        if frac > 0.3:
            self._client.model.geom('floor').friction[0] = np.random.uniform(0.8, 1.25)

        if self.home_scene:
            self._bury_all_boxes()
            self.course = []
            # 出生即贴墙（2026-07-14 按用户要求降难度）：不再从房间中央/基站
            # 前找墙，直接出生在【西墙边、沿墙姿态】—— 右侧激光贴墙 1~3cm、
            # 朝向 -y（沿墙向，墙在右侧、绕房间逆时针），出生瞬间就处于
            # 沿边状态，验证的就是纯"沿边+过角"能力。
            # 位置：西墙内侧面 x=-1.175，机身中心 x = -1.175 + 0.175 + d0；
            # y 取基站（y∈[0.57,0.83]）南侧 0.20~0.45，与基站/净化器
            # （y∈[-1.55,-1.35]）都不重叠，出发先沿西墙直行 ~1.6m。
            d0 = np.random.uniform(0.01, 0.03)
            self.robot_init = (-1.175 + 0.175 + d0,
                               np.random.uniform(0.20, 0.45),
                               0.05,
                               -np.pi / 2.0 + np.random.uniform(-0.08, 0.08))
        else:
            self._build_random_course(frac)

        self.prev_xy = np.array(self.robot_init[0:2])
        return

    def _build_random_course(self, frac):
        """用 mocap 墙块摆一条随机折线墙：机器人从原点朝 +x 出发，墙在右侧。

        墙块尺寸固定（gen_xml.WALL_BOX_SPECS，三种长度各两块），段长由所选
        墙块决定；不同"有效高度"通过把墙块下沉进地面实现。
        """
        self._bury_all_boxes()
        self.course = []
        self.wrap_corners = []

        n_seg = 1 + np.random.randint(0, 2 + int(round(3 * frac)))   # 1~2 -> 1~5
        n_seg = min(n_seg, self._num_boxes)

        # 阳角速成课程（corner drill）：首段用短墙（box05/06，0.9m）且强制
        # 第一个过渡为拐角、70% 为阳角 —— 出生 ~2s 内即进入包边练习。
        # 0714~0715_1 的包边样本只能等课程随机撒（首段长墙 1.6m，且过渡还
        # 可能是缺口/U 型），密度不足以在毒化梯度下学出"果断紧贴包边"。
        drill = np.random.rand() < self.CORNER_DRILL_PROB
        if drill:
            n_seg = max(n_seg, 2)

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
                # yaw≈-90° 正对右侧墙面（更宽的偏航，覆盖更多斜向接近）。必须靠
                # 前向激光把速度降到 0、停在墙前 ~1cm 再转向对齐，而不是撞上去。
                # d0 从 8cm 起：前向激光量程 10cm，8~10cm 出生的回合一睁眼就看到墙、
                # 必须立刻刹车（最强的刹车梯度）；更远的先巡航、进 10cm 再刹。
                d0 = np.random.uniform(0.08, 0.35)
                start_yaw = -np.pi / 2.0 + np.random.uniform(-0.5, 0.6)
            elif blind:
                # 全盲找墙（家居场景出生分布）：任意朝向、更远距离，激光量程内
                # 长时间无任何回波。期望行为：不打转、径直巡航（seek 速度分 +
                # 防打转罚），直到某个方向撞进 10cm 量程再刹车/贴边。朝向背离
                # 墙的回合会被 ACQUIRE_GRACE 超时截断 —— 全盲下本就不存在
                # "更聪明"的策略，学到"睁眼一片黑就直走"即达标。
                # 上限 0.60 -> 1.00m：家居从基站到隔断要全盲直行 ~2.4m，训练
                # 里必须见过"长时间全盲仍坚持直行"的时段（V_DES 0.28m/s 下
                # 1m ≈ 3.6s；朝向背墙的回合仍由 8s 找墙宽限截断）。
                d0 = np.random.uniform(0.20, 1.00)
                start_yaw = np.random.uniform(-np.pi, np.pi)
            else:
                d0 = np.random.uniform(0.12, 0.45)
                start_yaw = np.random.uniform(-0.5, 0.5)
        else:
            d0 = np.random.uniform(0.02, 0.04 + 0.05 * frac)
            start_yaw = np.random.uniform(-1.0, 1.0) * (0.10 + 0.25 * frac)

        # 墙块顺序：首段用长块（保证出生点在墙侧旁）；drill 回合首段改用
        # 短块（box05/06，0.9m），出生后 ~0.5m 就到墙端拐角
        model = self._client.model
        names = ['box' + repr(i + 1).zfill(2) for i in range(self._num_boxes)]
        first = names.pop(np.random.randint(4, 6) if drill else np.random.randint(2))
        np.random.shuffle(names)
        order = [first] + names

        # 缺口课程：热身后（frac>0.15）才在共线段之间插缺口，宽度随课程展开
        # 到 0.35m 封顶（见 GAP_MAX_BASE 注释：必须与门洞宽度拉开差距）。
        gaps_on = frac > 0.15
        gap_max = self.GAP_MAX_BASE + 0.15 * frac
        # 墙端 U 型包边课程：拐角热身后（frac>0.25）开启。
        uturn_on = frac > 0.25
        # 浅凸起课程（frac>0.6 才开启，每条课程最多一个）：凸出墙面 2~12cm
        # 的短凸块（壁柱/薄墙端面/基站座体），处于双激光盲区、只能靠碰撞感知。
        # 必须等包边技能固化后再引入 —— 见 PROT_PROB 注释（0715_1 教训）。
        prot_on = frac > 0.6
        prot_placed = False

        p = np.array([-0.4, -(0.175 + d0)])  # 墙内侧面折线起点（机器人右侧）
        d = np.array([1.0, 0.0])             # 墙走向
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
            normal = np.array([d[1], -d[0]])          # 指向远离机器人一侧（墙在右）
            center = mid + normal * ht
            seg_yaw = float(np.arctan2(d[1], d[0]))
            self._mocap_place(name, [center[0], center[1], h - hz], seg_yaw)
            self.course.append((center.copy(), L, h, seg_yaw))

            # ---- 盲区障碍（frac>0.6，每条课程最多一个，两变体各半）。共性：
            #      前激光零预警、只能靠碰撞感知；正确处置同构 —— 接触后向左
            #      让出、待侧激光捕获障碍面，其余复用既有沿边/包边技能。
            #      离段两端 >=0.55m（半长 0.30 + 拐角裕量 0.25），离出生点 >0.6m。 ----
            if prot_on and not prot_placed and np.random.rand() < self.PROT_PROB:
                s_lo, s_hi = 0.55, L - 0.55
                if s_hi > s_lo:
                    s_c = float(np.random.uniform(s_lo, s_hi))
                    if np.random.rand() < 0.5:
                        # 变体 A —— 沿墙浅凸块（壁柱/门框边/基站座体）：ridge02
                        # 转 90°（60cm 沿墙 × 12cm 垂直墙），大部分沉进墙体、只
                        # 凸出 jut。凸出量随课程渐进：先学"轻碰浅凸起就绕"再加深。
                        # 10cm 高 -> 四条激光射线全命中，绕行时凸块面可正常贴边。
                        jut_hi = self.PROT_JUT_MIN + (self.PROT_JUT_MAX - self.PROT_JUT_MIN) * frac
                        jut = float(np.random.uniform(self.PROT_JUT_MIN, jut_hi))
                        p_center = seg_start + d * s_c + normal * (0.06 - jut)
                        if float(np.linalg.norm(p_center)) > 0.60:
                            self._mocap_place('ridge02',
                                              [p_center[0], p_center[1], 0.05],
                                              seg_yaw + np.pi / 2.0)
                            self.course.append((p_center.copy(), 0.60, 0.10, seg_yaw))
                            prot_placed = True
                    else:
                        # 变体 B —— 薄墙拦路窄缝（0719 _2c 回放实测空洞）：ridge02
                        # 长轴垂直于墙横在巡航正前方，内端与所沿墙面留缝 g。
                        # 缝宽推导（贴边 1cm、底盘半径 0.175）：前激光射线距墙
                        # 0.185，g 下限 0.22 > 0.185 + 贴边摆动裕量 -> 射线必然
                        # 穿缝（纯盲）；机身横跨距墙 0.01~0.36m，g 上限 0.32 ->
                        # 与机身至少 4cm 重叠（必然接触，且缝窄于机身钻不过去）。
                        # 侧激光此时仍照着原墙 —— 阴角（前激光缩短）/阳角（侧激光
                        # 丢墙）两套反射都不触发，唯一线索是机身接触，逼策略学出
                        # "正前盲堵 -> 左转贴拦路墙续边"而不是原地往复。拦路墙
                        # 外端两个阳角顶点记入包边真值（同 U 型墙端逻辑），绕过
                        # 端头后沿背面回到原墙、阴角续边，全程复用既有技能。
                        g = float(np.random.uniform(0.22, 0.32))
                        b_center = seg_start + d * s_c - normal * (g + 0.30)
                        if float(np.linalg.norm(b_center)) > 0.60:
                            self._mocap_place('ridge02',
                                              [b_center[0], b_center[1], 0.05],
                                              seg_yaw)
                            self.course.append((b_center.copy(), 0.60, 0.10,
                                                seg_yaw + np.pi / 2.0))
                            tip = seg_start + d * s_c - normal * (g + 0.60)
                            self.wrap_corners.append((tip - d * 0.06).copy())
                            self.wrap_corners.append((tip + d * 0.06).copy())
                            prot_placed = True

            # ---- 段间过渡：缺口（共线续墙）/ 墙端 U 型包边 / 拐角，三选一。
            #      drill 回合的首个过渡强制为拐角（70% 阳角）。 ----
            p = seg_start + d * L
            prev_gap = False
            force_corner = drill and i == 0
            if (not force_corner) and gaps_on and i < n_seg - 1 \
                    and np.random.rand() < self.GAP_PROB:
                # 墙体缺口：保持墙向不变，跨过一段无墙缺口后续墙。记录真值供奖励。
                g = float(np.random.uniform(self.GAP_MIN, gap_max))
                self.gaps.append((p.copy(), d.copy(), g))
                p = p + d * g
                prev_gap = True
            elif (not force_corner) and uturn_on and i < n_seg - 1 \
                    and np.random.rand() < self.U_TURN_PROB:
                # 墙端 U 型包边（门洞侧柱课程）：下一段贴着同一堵墙的【背面】
                # 反向延伸、端面与本段对齐（跨过 2×半厚），机器人到墙端后需
                # 连做两个右外角（合计≈180°）绕过墙端、再沿背面继续贴边 ——
                # 正是家居场景"沿隔断走到门洞侧柱、拐进里屋"的动作。
                # 训练墙厚 16cm、家居隔断 5cm，包边窗口逻辑一致，可泛化。
                # 墙端两个阳角顶点都记入包边真值（内侧面端点 + 背面端点）。
                self.wrap_corners.append(p.copy())
                self.wrap_corners.append((p + normal * (2.0 * ht)).copy())
                p = p + normal * (2.0 * ht)
                d = -d
                prev_gap = True   # 端面已对齐，下一段不做拐角回收（同缺口逻辑）
            else:
                # 拐角：前期固定 90°，后期 55°~115°；墙在右侧时正角（逆时针）
                # =内角（墙折向机器人前方）、负角=外角。drill 首过渡 70% 阳角。
                ang = np.pi / 2 if frac < 0.3 else np.radians(np.random.uniform(55.0, 115.0))
                outer_p = 0.7 if force_corner else 0.5
                if np.random.rand() < outer_p:
                    ang = -ang          # 阳角（外角）
                    # 阳角顶点 = 两段内侧面折线的交点 p（拐角回收只回收墙块
                    # 摆放起点、不改内侧面所在直线），记入包边真值
                    self.wrap_corners.append(p.copy())
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
