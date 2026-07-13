"""线激光传感器读取与特性建模。

XML 侧（见 gen_xml._laser_sites/_laser_sensors）把一个"线激光"建成同一水平
位置、不同离地高度（3.5~7cm）的一组 MuJoCo rangefinder 射线。本模块把这组
原始射线读数汇总成一次"线激光测量"：

  distance   : 命中射线中的最近距离（线激光取条纹上最近点）；无命中 = max_range
  confidence : 命中射线数 / 总射线数。>=7cm 的墙面挡住全部射线 -> 1.0（可靠）；
               4~6cm 物体只挡住部分射线 -> (0,1)（低置信）；
               2~3cm 低矮物低于最下一条射线 -> 0.0（盲区，等同没有障碍）
  hit        : 是否有任一射线命中（False 即"量程内无障碍"）

用法：
    laser = LineLaser(model, data, 'front')   # 或 'right'（右侧侧边激光，与真机一致）
    reading = laser.read()
"""

from collections import namedtuple

import numpy as np
import mujoco

from envs.vacuum.gen_xml import LASER_MAX_RANGE

LaserReading = namedtuple('LaserReading', ['distance', 'confidence', 'hit', 'rays', 'hits'])


class LineLaser(object):

    def __init__(self, model, data, prefix, max_range=LASER_MAX_RANGE, noise_std=0.0):
        """prefix: 'front' 或 'right'；noise_std: 可选的测距高斯噪声 (m)。"""
        self.model = model
        self.data = data
        self.prefix = prefix
        self.max_range = max_range
        self.noise_std = noise_std

        # 按命名约定收集本激光的所有射线（rf_<prefix>_0, rf_<prefix>_1, ...）
        self._adrs = []
        self._site_ids = []
        i = 0
        while True:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                    'rf_{}_{}'.format(prefix, i))
            if sid < 0:
                break
            self._adrs.append(model.sensor_adr[sid])
            self._site_ids.append(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, 'laser_{}_{}'.format(prefix, i)))
            i += 1
        if not self._adrs:
            raise RuntimeError("模型中没有找到线激光 rf_{}_* —— "
                               "请确认 XML 由最新 gen_xml.py 生成".format(prefix))
        self.num_rays = len(self._adrs)

    def read(self):
        raw = np.array([self.data.sensordata[a] for a in self._adrs])
        # rangefinder 无回波返回负值；XML cutoff 会把超量程命中钳到恰好 max_range，
        # 因此只有严格小于量程的读数才是有效回波
        hits = (raw >= 0.0) & (raw < self.max_range - 1e-9)
        rays = np.where(hits, raw, self.max_range)
        if self.noise_std > 0.0:
            rays = np.where(hits,
                            np.clip(rays + np.random.normal(0, self.noise_std, rays.shape),
                                    0.0, self.max_range),
                            rays)
        n_hit = int(hits.sum())
        distance = float(rays[hits].min()) if n_hit else self.max_range
        confidence = n_hit / float(self.num_rays)
        return LaserReading(distance=distance, confidence=confidence,
                            hit=(n_hit > 0), rays=rays, hits=hits)

    def ray_states(self):
        """每条射线的 (世界系起点, 方向, 显示距离, 是否命中)，供可视化用。"""
        out = []
        for sid, adr in zip(self._site_ids, self._adrs):
            origin = self.data.site_xpos[sid].copy()
            direction = self.data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
            d = float(self.data.sensordata[adr])
            hit = (0.0 <= d <= self.max_range)
            out.append((origin, direction, d if hit else self.max_range, hit))
        return out
