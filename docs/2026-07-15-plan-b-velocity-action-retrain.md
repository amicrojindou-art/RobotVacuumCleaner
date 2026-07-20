# 方案B：速度动作重训（wheel-velocity action space）实施文档

> 日期：2026-07-15
> 状态：待执行（方案A链路打通、用户看过效果后启动）
> 读者：后续执行重训的 Claude / 开发者。本文档自包含，按顺序执行即可。
> 前置阅读：`hoslam/docs/superpowers/specs/2026-07-14-rl-wall-follow-design.md`（大核集成总设计）

---

## 1. 为什么要重训（问题陈述）

当前 vacuum_wf 策略（`experiments/vacuum_wf0715`）动作 = **左右轮力矩**
（`envs/vacuum/robot.py` VacuumRobot：`torque = clip(action × 2.0, ±2)` 直接给
MuJoCo 电机执行器）。但真机大核对小核只有**轮速目标**通道
（`g_vels_shm`：左右轮 mm/s int16 + accels，小核做电机速度闭环，大核无力矩/PWM 接口）。

方案A用"虚拟动力学 + 小核速度环"去逼近力矩语义（见
`scripts/demo_plan_a_adapter.py`），存在原理性残差：

- 小核速度环带宽/饱和特性 ≠ 仿真自由动力学，堵转/爬坎/顶墙时力矩语义失真最大，
  而这恰是策略学到的关键能力（贴边过坎、阳角包边蹭角）；
- 虚拟模型的质量/惯量/阻尼参数要靠实车对拍标定，每台机器还有个体差异。

方案B让训练时的动作语义与真机接口**天然一致**：动作 = 轮速目标，仿真内嵌一个
模拟小核电机环的低层伺服。sim2real 差距从"整条动力学链"缩小为"速度环响应差异"，
且后者可以用域随机化覆盖。

## 2. 改动总览（训练仓 RobotVacuumCleaner）

| 文件 | 改动 | 规模 |
|---|---|---|
| `envs/vacuum/robot.py` | 新增 `VacuumVelocityRobot` 类（不动现有 VacuumRobot） | ~60 行 |
| `envs/vacuum/vacuum_wf_env.py` | robot 实例化换成 VacuumVelocityRobot；观测不变 | ~5 行 |
| `tasks/wall_follow_task.py` | 奖励项检查（见 §5），预计无改动或仅调 1~2 个权重 | 0~10 行 |
| `hoslam/tools/rl_wf_convert/export_onnx.py` | meta 的 action_type/scale 字段区分（见 §8） | ~5 行 |

观测（34 维）**完全不变**——`wheel_vel`、`prev_action` 维度语义保持，只是
prev_action 的物理含义从"力矩系数"变为"轮速系数"，对网络是透明的。

## 3. VacuumVelocityRobot 设计

```python
class VacuumVelocityRobot(VacuumRobot):
    """动作 = 左右轮速目标系数 ∈ [-1,1]，×V_MAX_WHEEL 得轮速目标 (rad/s)。
    内嵌子步 P 伺服模拟小核电机速度环。"""

    V_MAX_WHEEL = 8.6       # rad/s；×轮半径0.035 = 0.30 m/s 轮缘线速
                            # （真机 WF 巡航 0.12~0.24 m/s，留 25% 余量）
    KP_SERVO    = 0.5       # N·m/(rad/s)。稳定性: kp < (m/2)r²/sim_dt ≈ 0.71
    SLEW_RATE   = 40.0      # rad/s² 轮速目标斜率限幅（≈1.4 m/s² 轮缘加速度），
                            # 对应真机 g_vels_shm 的 accels 字段语义

    def step(self, action):
        action = np.clip(np.asarray(action).flatten(), -1.0, 1.0)
        w_set_raw = action * self.V_MAX_WHEEL
        # 斜率限幅（真机小核同样有加速度限制；训练内置使策略学会平滑指令）
        w_set = self._slew(w_set_raw)
        self.last_action / self.last_torque 记账与基类一致
        for _ in range(self.frame_skip):
            边刷 qvel 驱动（照抄基类 do_simulation）
            wv = 当前轮速 (rad/s)
            tau = clip(kp_eff * (w_set - wv), ±MAX_WHEEL_TORQUE)   # kp_eff 见 §4 域随机化
            set_motor_torque(tau); client.step()
        self.prev_action = action; self.prev_torque = 本控制步伺服力矩均值
        return tau
```

要点：

1. **动作进 env 前先 clip ±1**（Gaussian 采样会越界；0715 版靠力矩饱和天然
   限幅，速度版必须显式 clip，否则 V_MAX 失去意义）。
2. **prev_torque 记伺服实际力矩**（能耗奖励项继续有效——策略高频抖速度指令
   会导致伺服力矩大，能耗罚自动抑制抖动）。
3. **力矩上限保留 ±2 N·m**：堵转（顶墙/爬坎）时伺服饱和输出最大力矩，物理
   行为与真机小核一致（速度环深度饱和 = 恒最大力矩推），策略能继续学到
   "顶推过坎"的行为，只是通过速度指令表达。
4. `frame_skip`、control_dt=0.025（40Hz）不变。

## 4. 域随机化（sim2real 关键，reset 时采样）

在 `VacuumVelocityRobot` 加 `randomize()`，由 env.reset_model 调用：

| 参数 | 名义值 | 随机化范围 | 对应真机不确定性 |
|---|---|---|---|
| kp_eff（伺服增益） | 0.5 | ×U(0.6, 1.4)，上限 clip 0.7 | 小核速度环带宽个体差异 |
| 指令延迟 | 0 | 0~2 个控制步（FIFO 队列） | SPI-RPC 下发 + 小核执行延迟（实测 10~30ms） |
| V_MAX 缩放 | 1.0 | ×U(0.95, 1.05) | 轮径磨损/打滑标定误差 |
| 轮速测量噪声 | 0 | σ=0.1 rad/s 高斯 | 编码器量化 |

激光噪声课程沿用 task 现有的 `sensor_noise = 0.002 * frac`。
延迟随机化尤其重要：方案A demo 若显示伺服跟踪误差大，说明策略对执行延迟敏感，
重训时把延迟课程做足（0→2 步渐进）。

## 5. 奖励项检查清单（tasks/wall_follow_task.py calc_reward）

逐项过一遍，预计大多不动：

- **贴边 track 项**：基于激光距离，与动作空间无关，不动。
- **能耗项**（用 prev_torque）：语义保留（见 §3.2），不动。
- **动作平滑项**（用 last_action 与本步 action 差）：语义从"力矩变化率"变为
  "速度指令变化率"——量纲变了但归一化尺度相同（都是 [-1,1] 系数），
  先不动权重，看训练曲线的 wobble 指标再说。
- **wobble/yaw_rate 项**：行为级，不动。
- **press/scrape 项**：接触相关，不动。
- 若出现"速度指令高频抖动但轮子跟不上（被伺服滤掉）导致平滑罚失效"：
  把平滑项改为对 **实际轮速变化率** 惩罚（`wheel_vel` 差分）。

## 6. 训练与评估

```powershell
# RL312 环境，仓库根目录（参数与 0715 成功配置一致，只换 logdir）
$env:PYTHONPATH="."
python run_experiment.py train --env vacuum_wf --logdir experiments/vacuum_wf_vel01 `
    --no_mirror --n_itr 1500 --num_procs 12 --max_traj_len 600 `
    --target_kl 0.04 --lr_decay --eval_freq 50
```

注意（历史教训，来自 CHANGELOG 与 0714~0715 三次重训失败）：

- **激光通道保持纯净**：任何情况下不把碰撞翻译进激光观测（摧毁阳角包边）。
- torch 必须先于 ray import（run_experiment.py 已处理，勿改顺序）。
- 观测 34 维顺序不可动；改了就要同步 `hoslam/tools/rl_wf_convert/export_onnx.py`
  的 obs_layout 与大核 rl_wf_observation.cpp。

**验收标准**（对照 0715 力矩版基线）：

1. `eval.txt` 平均回报 ≥ 85（0715 水平），无 collapse；
2. `scripts/demo_plan_a_adapter.py` 加 `--mode velocity`（重训后加此模式：
   策略输出直接当轮速目标，跳过虚拟动力学）：early-term 率、贴边误差
   （目标：稳态 |d-1cm| 均值 < 5mm）、contact 占比不劣于 direct 基线；
3. 家居场景回放（`VacuumWFEnv(home_scene=True)`）能整圈巡边，含门洞阳角包边；
4. 速度指令平滑性：|Δw_set| 分布无高频满幅抖动（否则回 §5 调平滑项）。

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| 速度动作学不出"顶推过坎"（力矩饱和行为要经速度环间接表达） | 保留力矩上限=真机等效值；课程里保留门槛/凸包场景；必要时动作加第三维"boost"（不建议，先试两维） |
| 策略利用伺服暂态（学出抖指令让轮子共振） | 能耗罚 + 延迟/增益域随机化天然抑制；观察 servo 力矩谱 |
| 重训后行为风格变化影响大核闭环判据（check_wf_loop 周长门限） | 与方案A同一问题，Phase 4 实车回归覆盖 |

## 8. 重训完成后的下游动作

1. `export_onnx.py --checkpoint experiments/vacuum_wf_vel01/actor.pt ...`
   —— meta 里 `action_type` 改为 `"wheel_velocity"`、`action_scale` 改为
   `V_MAX_WHEEL×0.035 m/s`（导出脚本按 meta 版本字段区分，大核加载时校验）。
2. `convert_and_pack.py` + `check_parity.py`（用 velocity 版 rollout 录制观测）。
3. **大核 rl_wf_action.cpp 简化**：删虚拟动力学，保留 clip ±1 → ×V_MAX →
   mm/s int16 + 斜率限幅写 g_vels_shm。meta 的 action_type 字段驱动分支，
   两版模型可共存热切换（A/B 对比实车效果）。
4. 方案A的虚拟动力学参数标定工作全部作废，不要再投入。

## 9. 当前基线记录（供对比）

- 力矩版权重：`experiments/vacuum_wf0715`（回报 ~85，ep len ~300@40Hz）
- 方案A demo 结果：`experiments/vacuum_wf0715/plan_a_demo/results.json`
  （direct vs adapter 各 8 集，同种子同课程）
- 转换链产物：`hoslam/tools/rl_wf_convert/out/`（onnx/mnn/model/meta，
  三方对拍 PT↔ONNX 9.5e-07、PT↔MNN 4.8e-07 PASS）
