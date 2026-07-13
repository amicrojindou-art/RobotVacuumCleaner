"""扫地机器人（差速圆盘底盘）MuJoCo 模型生成器。

与人形项目的 envs/jvrc/gen_xml.py 作用一致：在环境首次创建时把一个完整的
MJCF 模型写到磁盘，再由 MujocoEnv 加载。区别在于扫地机器人结构简单（纯几何体，
无需 mesh），所以这里直接拼接 XML 字符串，无需 dm_control。

外形按真机（米家扫拖机器人，Φ350mm、总高 90mm）建模：
  碰撞/动力学部分（参与物理）：
    - floor            : 地面平面
    - box01 ~ box06    : 通用 box geom，由任务在 reset 时摆成"门槛/凸台/障碍"，
                         未使用的默认埋到地下 z=-1（与 stepping_task 的做法一致）。
    - chassis_geom     : 圆柱壳体（半径 0.175m），底面离地 GROUND_CLEARANCE，
                         顶面在 TOTAL_HEIGHT=0.09m —— 与真机外廓一致。
                         >= GROUND_CLEARANCE 的坎会顶住底盘 → "高底盘骑坎"脱困场景。
    - left/right_wheel : 两个驱动轮（hinge，绕 y 轴），由 motor 执行器施加力矩。
    - caster_f/b       : 前后两个低摩擦万向支撑球（简化的万向轮）。
  纯视觉部分（contype=0 conaffinity=0 mass=0，不影响物理）：
    - 白色上盖 / 银色控制旋钮与按键 / 收起状态的升降雷达盖
    - 前向黑色传感器窗（ToF/摄像头）
    - 底部：橙色主刷 + 刷仓、灰色拖布滚筒、左右两把伸出机身的三叶边刷、
      下视悬崖传感器、回充触点、轮毂盖

家居场景（参考实验室测试场地照片，scene=True 时生成，纯静态 world geom）：
    - 白色矮围墙围出 4.4m x 3.6m 场地，中间一道带门洞的隔断分出里外两间
      （门洞无门槛，平地直通）；
    - 地面：外间白色亮面砖、里间浅米色亮面砖（纯视觉贴片），外间一块 16mm 短毛
      地毯（物理凸台，进出地毯即真实"上毯沿"）；
    - 家具/障碍：木边柜、布艺沙发、白色净化器立柱、回充基座+底板、
      彩色泡沫积木块。
    - 起点 (0,0) -> 目标 (1.2,0) 的中央走廊保持无静态障碍，四种任务模式
      （FLAT/CROSS/HIGHCENTER/AGAINST）的动态地形摆放区域不与家具重叠。

可调几何参数集中在下方常量，改尺寸只需改这里。
"""

import os

# ---- 可调几何/物理参数（单位：米 / 千克 / 牛·米）----
CHASSIS_RADIUS   = 0.175   # 底盘半径（直径 0.35m，与真机 Φ350mm 一致）
TOTAL_HEIGHT     = 0.090   # 整机高度（雷达下降后 90mm，与真机一致）
GROUND_CLEARANCE = 0.015   # 底盘底面离地间隙（决定多高的坎会"骑"住底盘）
WHEEL_RADIUS     = 0.035   # 驱动轮半径
WHEEL_HALF_W     = 0.012   # 驱动轮半宽
WHEEL_Y          = 0.150   # 轮子左右偏移（半轮距）
CASTER_RADIUS    = 0.020   # 万向支撑球半径
CASTER_X         = 0.140   # 万向球前后偏移
CHASSIS_MASS     = 2.50    # 底盘质量（整机约 3kg）
WHEEL_MASS       = 0.20    # 单个驱动轮质量
CASTER_MASS      = 0.05    # 单个万向球质量
MAX_WHEEL_TORQUE = 2.0     # 执行器力矩上限（ctrlrange）

# ---- 训练用可移动地形（mocap body）----
# 关键工程约束：MuJoCo 在编译期把"静态(world) geom"的位置/尺寸烘焙进碰撞
# 加速结构，运行时修改 model.geom_pos/size 对【碰撞检测不生效】——激光
# (mj_ray) 能看到新位置，物理上却是可以穿过的"幽灵"（已实测确认）。
# 因此所有需要在 reset 时动态摆放的地形都挂在 mocap body 上：
#   - 位置/朝向用 data.mocap_pos / mocap_quat 运行时设置，碰撞正确跟随；
#   - geom 尺寸固定不变（palette），"不同有效高度"通过把箱体下沉进地面实现
#     （mocap body 与 world 同属一个 weld，与地面之间不会产生接触）。
WALL_BOX_SPECS = (   # 沿边沿墙训练墙：(名字, 半长, 半厚, 半高)
    ("box01", 0.80, 0.08, 0.125),
    ("box02", 0.80, 0.08, 0.125),
    ("box03", 0.60, 0.08, 0.125),
    ("box04", 0.60, 0.08, 0.125),
    ("box05", 0.45, 0.08, 0.125),
    ("box06", 0.45, 0.08, 0.125),
)
RIDGE_BOX_SPECS = (  # 过坎/脱困任务：门槛长脊 + AGAINST 短墙 + 骑坎窄脊
    ("ridge01", 0.06, 1.00, 0.05),
    ("ridge02", 0.06, 0.30, 0.05),
    # 骑坎脊：沿 x 方向、宽 10cm——恰好能从中线万向球(|y|<0.022)与
    # 驱动轮(|y|>0.138)之间的空隙下方穿过，让腹部压脊、轮子减载
    ("ridge03", 1.00, 0.05, 0.05),
)
NUM_BOXES = len(WALL_BOX_SPECS)

# ---- 线激光传感器（沿边沿墙用，正前方 + 右侧各一个；与真机一致装在右侧）----
# 真机的"线激光"是一条竖直展开的激光条纹，这里用同一水平位置、不同离地高度的
# 一组 rangefinder 射线近似，从而体现两条关键特性：
#   1) 有效测距范围 10cm：超出量程视为无回波；
#   2) 有效检测高度约 7cm：射线分布在离地 3.5~7cm ——
#      高度 >=7cm 的墙面/家具挡住全部射线（可靠检测）；
#      2~3cm 的低矮物体低于最下一条射线，完全检测不到（盲区）；
#      4~6cm 的物体只挡住部分射线（低置信区）。
LASER_MAX_RANGE   = 0.10                            # 有效测距范围 10cm
LASER_RAY_HEIGHTS = (0.035, 0.047, 0.058, 0.070)    # 各射线离地高度（世界系）

# 底盘中心高度：使轮底刚好触地
BASE_Z    = WHEEL_RADIUS + GROUND_CLEARANCE          # 0.05（root body 原点离地高度）
WHEEL_LZ  = WHEEL_RADIUS - BASE_Z                    # 轮轴相对底盘原点的 z（局部坐标）
CASTER_LZ = CASTER_RADIUS - BASE_Z                   # 万向球球心相对底盘原点的 z

# 壳体圆柱：底面在 GROUND_CLEARANCE、顶面在 TOTAL_HEIGHT（世界系）
CHASSIS_HALF_H = (TOTAL_HEIGHT - GROUND_CLEARANCE) / 2.0            # 0.0375
CHASSIS_LZ     = GROUND_CLEARANCE + CHASSIS_HALF_H - BASE_Z         # 0.0025
CHASSIS_TOP_LZ = CHASSIS_LZ + CHASSIS_HALF_H                        # 壳体顶面（局部 z）
CHASSIS_BOT_LZ = CHASSIS_LZ - CHASSIS_HALF_H                        # 壳体底面（局部 z）


def _box_geoms():
    """动态地形：每个 box 是一个 mocap body（默认埋到地下 z=-2）。

    任务在 reset 时通过 data.mocap_pos/quat 摆放；geom 尺寸固定（见
    WALL_BOX_SPECS / RIDGE_BOX_SPECS 注释），有效高度靠下沉入地调节。
    """
    bodies = []
    for name, hx, hy, hz in WALL_BOX_SPECS + RIDGE_BOX_SPECS:
        bodies.append(
            '    <body name="{n}" mocap="true" pos="0 0 -2">\n'
            '      <geom name="{n}" type="box" size="{hx} {hy} {hz}" material="threshold" '
            'contype="1" conaffinity="1" condim="3" friction="1 0.05 0.05"/>\n'
            '    </body>'.format(n=name, hx=hx, hy=hy, hz=hz)
        )
    return "\n".join(bodies)


def _side_brushes():
    """左右两把三叶边刷：小毂 + 3 根过中心的刷条 = 6 条刷臂。

    真机边刷伸出机身边缘，这里刷条端点半径约 0.19m > 壳体 0.175m，同样外伸。
    每把边刷是挂在底盘上的独立 body，带绕 z 轴的 hinge 关节，由 VacuumRobot
    在仿真中以恒定转速驱动（清扫时旋转、向机身中线扫拢：左刷顺时针/右刷逆时针）。
    刷子不参与碰撞（contype=0），只给毂一个极小质量满足关节体的惯量要求，
    对底盘动力学影响可忽略；关节数 +2 不改变执行器数量与动作/观测维度。
    """
    quats = ["1 0 0 0", "0.8660254 0 0 0.5", "0.5 0 0 0.8660254"]  # 绕 z 转 0/60/120°
    hub_z = round(0.0085 - BASE_Z, 4)
    parts = []
    for side, y in (("l", 0.100), ("r", -0.100)):
        parts.append('      <body name="sbrush_{s}" pos="0.112 {y} {z}">'.format(
            s=side, y=y, z=hub_z))
        parts.append('        <joint name="sbrush_{s}" type="hinge" axis="0 0 1" '
                     'damping="0" armature="1e-05" frictionloss="0"/>'.format(s=side))
        parts.append('        <geom name="sbrush_hub_{s}" type="cylinder" size="0.010 0.003" '
                     'contype="0" conaffinity="0" group="1" mass="0.005" '
                     'material="dark"/>'.format(s=side))
        for i, q in enumerate(quats):
            parts.append('        <geom class="visual" name="sbrush_{s}{i}" type="box" '
                         'size="0.042 0.0035 0.0012" pos="0 0 -0.002" quat="{q}" '
                         'rgba="0.15 0.15 0.17 1"/>'.format(s=side, i=i, q=q))
        parts.append('      </body>')
    return "\n".join(parts)


def _laser_sites():
    """两个线激光的发射 site（挂在 base body 上，随机身运动）。

    MuJoCo rangefinder 沿 site 的 +z 轴发射：前向激光 z->+x，右侧激光 z->-y
    （真机侧边传感器装在右侧，沿墙时墙在机器人右侧）。
    site 放在壳体边缘（半径 CHASSIS_RADIUS 处），读数即"机身边缘到物体"的距离。
    rangefinder 自动排除 site 所在 body（base）的所有 geom，不会打到自身壳体/边刷。
    """
    specs = (
        ("front", CHASSIS_RADIUS, 0.0, "0.7071068 0 0.7071068 0"),   # 绕y转90°: z->+x
        ("right", 0.0, -CHASSIS_RADIUS, "0.7071068 0.7071068 0 0"),  # 绕x转+90°: z->-y
    )
    s = []
    for tag, px, py, quat in specs:
        for i, hz in enumerate(LASER_RAY_HEIGHTS):
            s.append('      <site name="laser_{t}_{i}" pos="{x} {y} {z}" quat="{q}" '
                     'size="0.003" rgba="1 0.2 0.2 0.8"/>'.format(
                         t=tag, i=i, x=px, y=py, z=round(hz - BASE_Z, 4), q=quat))
    return "\n".join(s)


def _laser_sensors():
    """rangefinder 传感器定义。cutoff 把读数截断在有效量程 10cm；无回波返回负值。"""
    s = []
    for tag in ("front", "right"):
        for i in range(len(LASER_RAY_HEIGHTS)):
            s.append('    <rangefinder name="rf_{t}_{i}" site="laser_{t}_{i}" '
                     'cutoff="{r}"/>'.format(t=tag, i=i, r=LASER_MAX_RANGE))
    return "\n".join(s)


def _visual_geoms():
    """壳体上的外观细节（纯视觉，不参与碰撞、不带质量）。"""
    top = round(CHASSIS_TOP_LZ, 4)     # 0.040
    bot = round(CHASSIS_BOT_LZ, 4)     # -0.035
    g = []

    # ---- 顶面 ----
    # 上盖面板（略小半径的浅色盖板，形成一圈接缝）
    g.append('      <geom class="visual" name="top_lid" type="cylinder" size="0.171 0.0012" '
             'pos="0 0 {z}" material="lid_white"/>'.format(z=round(top + 0.0012, 4)))
    # 银色控制旋钮 + 电源/回充两个按键（偏前方，参考真机顶视照片）
    g.append('      <geom class="visual" name="dial" type="cylinder" size="0.030 0.0018" '
             'pos="0.095 0 {z}" material="silver"/>'.format(z=round(top + 0.0042, 4)))
    g.append('      <geom class="visual" name="btn_power" type="cylinder" size="0.0055 0.0008" '
             'pos="0.104 0 {z}" material="dark"/>'.format(z=round(top + 0.0068, 4)))
    g.append('      <geom class="visual" name="btn_home" type="cylinder" size="0.0055 0.0008" '
             'pos="0.086 0 {z}" material="dark"/>'.format(z=round(top + 0.0068, 4)))
    # 升降雷达盖（收起状态，接近与上盖平齐）
    g.append('      <geom class="visual" name="lidar_cap" type="cylinder" size="0.041 0.0015" '
             'pos="-0.050 0 {z}" material="dark"/>'.format(z=round(top + 0.0039, 4)))

    # ---- 侧面 ----
    # 前向传感器窗（ToF/摄像头黑窗）
    g.append('      <geom class="visual" name="front_window" type="box" size="0.005 0.048 0.010" '
             'pos="0.172 0 0.008" material="dark"/>')
    # 碰撞缓冲条接缝（一圈略深色的下部环带）
    g.append('      <geom class="visual" name="bumper_band" type="cylinder" size="0.1758 0.010" '
             'pos="0 0 -0.021" material="bumper_white"/>')

    # ---- 底部 ----
    # 主刷仓框 + 橙色主刷（胶毛混合滚刷，轴向 y）
    g.append('      <geom class="visual" name="brush_frame" type="box" size="0.026 0.090 0.005" '
             'pos="0.058 0 -0.0305" material="dark"/>')
    g.append('      <geom class="visual" name="main_brush" type="cylinder" size="0.0155 0.080" '
             'pos="0.058 0 -0.033" quat="0.7071068 0.7071068 0 0" material="brush_orange"/>')
    # 拖布滚筒（灰色绒面，位于机身后部）
    g.append('      <geom class="visual" name="mop_roller" type="cylinder" size="0.017 0.082" '
             'pos="-0.100 0 -0.032" quat="0.7071068 0.7071068 0 0" material="mop_gray"/>')
    # 下视悬崖传感器 x4（前缘两个、左右各一）
    for i, (sx, sy) in enumerate([(0.150, 0.052), (0.150, -0.052), (0.0, 0.162), (0.0, -0.162)]):
        g.append('      <geom class="visual" name="cliff{i}" type="cylinder" size="0.008 0.0008" '
                 'pos="{x} {y} {z}" material="dark"/>'.format(
                     i=i, x=sx, y=sy, z=round(bot - 0.0004, 4)))
    # 回充触点 x2（后部，银色）
    for s, sy in (("l", 0.032), ("r", -0.032)):
        g.append('      <geom class="visual" name="charge_{s}" type="box" size="0.014 0.007 0.0008" '
                 'pos="-0.148 {y} {z}" material="silver"/>'.format(s=s, y=sy, z=round(bot - 0.0004, 4)))

    g.append(_side_brushes())
    return "\n".join(g)


def _scene_geoms():
    """家居测试场景（全部是挂在 worldbody 下的静态 geom）。

    坐标约定：机器人从 (0,0) 朝 +x 出发，目标在 (1.2, 0)。
    场地内框：x ∈ [-1.175, 3.175]，y ∈ [-1.775, 1.775]；
    隔断在 x=2.0，门洞 y ∈ [-0.3, 0.4]（目标点在隔断之前的外间）。
    中央走廊 x ∈ [-0.3, 1.7]、|y| < 0.9 无静态障碍，任务动态地形不受影响。
    """
    g = []

    # ---- 地面分区（纯视觉：外间白亮面砖 / 里间浅米色亮面砖）----
    g.append('    <geom class="visual" name="zone_white" type="box" size="1.59 1.78 0.0008" '
             'pos="0.41 0 0.0008" material="tile_white"/>')
    g.append('    <geom class="visual" name="zone_inner" type="box" size="0.59 1.78 0.0008" '
             'pos="2.59 0 0.0008" material="tile_beige"/>')

    # ---- 白色矮围墙（20cm 高，5cm 厚）----
    for name, x, y, hx, hy in [
        ("wall_w", -1.2, 0.0, 0.025, 1.85),
        ("wall_e",  3.2, 0.0, 0.025, 1.85),
        ("wall_s",  1.0, -1.8, 2.225, 0.025),
        ("wall_n",  1.0,  1.8, 2.225, 0.025),
        # 带门洞的隔断（x=2.0，门洞 y -0.3 ~ 0.4）
        ("part_s",  2.0, -1.05, 0.025, 0.75),
        ("part_n",  2.0,  1.10, 0.025, 0.70),
    ]:
        g.append('    <geom name="{n}" type="box" size="{hx} {hy} 0.10" pos="{x} {y} 0.10" '
                 'material="wall_white"/>'.format(n=name, x=x, y=y, hx=hx, hy=hy))

    # ---- 短毛地毯（12mm 物理凸台，上/下毯沿是真实小坎，且低于线激光最低
    #      射线 -> 属于检测盲区）。放在外间东南侧空地：避开中央走廊（|y|<0.9）、
    #      南墙沿边路径（y<-1.41）、木柜轮廓（x<1.01）与隔断沿边路径（x>1.61），
    #      不干扰沿边沿墙清扫的贴边环线 ----
    g.append('    <geom name="carpet" type="box" size="0.25 0.15 0.006" pos="1.30 -1.19 0.006" '
             'material="carpet_mat" friction="1.2 0.1 0.1"/>')

    # ---- 家具 ----
    # 木边柜（外间南墙）
    g.append('    <geom name="cabinet" type="box" size="0.35 0.20 0.14" pos="0.3 -1.55 0.14" '
             'material="wood"/>')
    # 布艺沙发（外间北墙，坐垫 + 靠背）
    g.append('    <geom name="sofa_seat" type="box" size="0.45 0.25 0.14" pos="1.5 1.5 0.14" '
             'material="fabric"/>')
    g.append('    <geom class="visual" name="sofa_back" type="box" size="0.45 0.05 0.14" '
             'pos="1.5 1.72 0.35" material="fabric"/>')
    # 白色净化器立柱（外间西南角）
    g.append('    <geom name="purifier" type="box" size="0.10 0.10 0.28" pos="-1.0 -1.45 0.28" '
             'material="wall_white"/>')
    # 回充基座（西墙）+ 前伸底板（8mm，可驶上）
    g.append('    <geom name="dock" type="box" size="0.065 0.13 0.11" pos="-1.11 0.7 0.11" '
             'material="wall_white"/>')
    g.append('    <geom name="dock_plate" type="box" size="0.105 0.13 0.004" pos="-0.94 0.7 0.004" '
             'material="dark" friction="0.8 0.05 0.05"/>')
    # 彩色泡沫积木（里间东墙，两块落地 + 一块叠放）
    g.append('    <geom name="blk_r" type="box" size="0.065 0.065 0.065" pos="3.05 0.95 0.065" '
             'material="foam_red"/>')
    g.append('    <geom name="blk_g" type="box" size="0.065 0.065 0.065" pos="3.05 1.12 0.065" '
             'material="foam_green"/>')
    g.append('    <geom name="blk_b" type="box" size="0.065 0.065 0.065" pos="3.05 0.95 0.195" '
             'material="foam_blue"/>')

    return "\n".join(g)


def build_xml(scene=True):
    xml = """<mujoco model="vacuum_cleaner">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.0025" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size njmax="500" nconmax="200"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <joint armature="0.005" damping="0.05" frictionloss="0.002"/>
    <geom contype="1" conaffinity="1" condim="3" friction="1 0.1 0.1"/>
    <default class="wheel">
      <geom friction="2.5 0.05 0.05" rgba="0.12 0.12 0.12 1"/>
    </default>
    <default class="caster">
      <!-- priority=1：接触摩擦取万向球自己的低摩擦值，而不是与地面取 max，
           否则"低摩擦万向球"实际不生效（MuJoCo 默认按逐元素最大值合成） -->
      <geom friction="0.05 0.005 0.005" priority="1" rgba="0.45 0.45 0.47 1"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
    <motor ctrllimited="true" ctrlrange="-{tau} {tau}"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.12 0.18 0.26" rgb2="0.18 0.26 0.36" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.15"/>
    <material name="threshold" rgba="0.80 0.50 0.20 1"/>
    <material name="body_white"   rgba="0.93 0.93 0.95 1" specular="0.6" shininess="0.6"/>
    <material name="lid_white"    rgba="0.97 0.97 0.98 1" specular="0.7" shininess="0.7"/>
    <material name="bumper_white" rgba="0.84 0.84 0.87 1" specular="0.4" shininess="0.4"/>
    <material name="silver"       rgba="0.72 0.73 0.76 1" specular="0.9" shininess="0.9"/>
    <material name="dark"         rgba="0.10 0.10 0.12 1" specular="0.3" shininess="0.4"/>
    <material name="brush_orange" rgba="0.95 0.45 0.10 1"/>
    <material name="mop_gray"     rgba="0.72 0.72 0.70 1"/>
    <material name="wall_white"   rgba="0.94 0.94 0.92 1" specular="0.3" shininess="0.3"/>
    <material name="tile_white"   rgba="0.90 0.90 0.88 1" reflectance="0.25" specular="0.8" shininess="0.8"/>
    <material name="tile_beige"   rgba="0.88 0.84 0.74 1" reflectance="0.25" specular="0.8" shininess="0.8"/>
    <material name="carpet_mat"   rgba="0.52 0.44 0.34 1" specular="0.05" shininess="0.05"/>
    <material name="wood"         rgba="0.55 0.38 0.22 1" specular="0.2" shininess="0.3"/>
    <material name="fabric"       rgba="0.36 0.38 0.42 1" specular="0.05" shininess="0.05"/>
    <material name="foam_red"     rgba="0.85 0.25 0.25 1"/>
    <material name="foam_green"   rgba="0.30 0.70 0.40 1"/>
    <material name="foam_blue"    rgba="0.30 0.50 0.80 1"/>
  </asset>

  <worldbody>
    <light name="top"  pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light name="side" pos="2 2 3" dir="-0.5 -0.5 -1" diffuse="0.35 0.35 0.35" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" friction="1 0.1 0.1"/>

{scene}

{boxes}

    <body name="base" pos="0 0 {base_z}">
      <freejoint name="root"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="chassis_geom" type="cylinder" size="{cr} {chh}" pos="0 0 {clz}"
            material="body_white" mass="{cm}" friction="0.5 0.05 0.05"/>
      <geom name="caster_f" class="caster" type="sphere" size="{castr}" pos="{castx} 0 {castz}" mass="{castm}"/>
      <geom name="caster_b" class="caster" type="sphere" size="{castr}" pos="-{castx} 0 {castz}" mass="{castm}"/>

{visual}

{lasers}

      <body name="left_wheel" pos="0 {wy} {wlz}">
        <joint name="left_wheel" type="hinge" axis="0 1 0"/>
        <geom class="wheel" type="cylinder" size="{wr} {whw}" quat="0.7071068 0.7071068 0 0" mass="{wm}"/>
        <geom class="visual" name="hubcap_l" type="cylinder" size="0.015 0.0128" quat="0.7071068 0.7071068 0 0" material="silver"/>
      </body>
      <body name="right_wheel" pos="0 -{wy} {wlz}">
        <joint name="right_wheel" type="hinge" axis="0 1 0"/>
        <geom class="wheel" type="cylinder" size="{wr} {whw}" quat="0.7071068 0.7071068 0 0" mass="{wm}"/>
        <geom class="visual" name="hubcap_r" type="cylinder" size="0.015 0.0128" quat="0.7071068 0.7071068 0 0" material="silver"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_wheel_motor"  joint="left_wheel"  gear="1"/>
    <motor name="right_wheel_motor" joint="right_wheel" gear="1"/>
  </actuator>

  <sensor>
{sensors}
  </sensor>
</mujoco>
""".format(
        tau=MAX_WHEEL_TORQUE,
        scene=_scene_geoms() if scene else "",
        boxes=_box_geoms(),
        visual=_visual_geoms(),
        lasers=_laser_sites(),
        sensors=_laser_sensors(),
        base_z=BASE_Z,
        cr=CHASSIS_RADIUS, chh=CHASSIS_HALF_H, clz=CHASSIS_LZ, cm=CHASSIS_MASS,
        castr=CASTER_RADIUS, castx=CASTER_X, castz=CASTER_LZ, castm=CASTER_MASS,
        wy=WHEEL_Y, wlz=WHEEL_LZ, wr=WHEEL_RADIUS, whw=WHEEL_HALF_W, wm=WHEEL_MASS,
    )
    return xml


def builder(export_path, scene=True):
    """生成 XML 并写到 export_path。scene=False 可生成不带家居场景的空场地。"""
    print("Generating vacuum-cleaner XML model...")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(build_xml(scene=scene))
    print("Exported XML model to ", export_path)
    return


if __name__ == "__main__":
    import sys
    builder(sys.argv[1])
