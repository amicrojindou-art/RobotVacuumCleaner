import os
import numpy as np
import mujoco
import mujoco_viewer

DEFAULT_SIZE = 500


def _add_marker_to_scene_compat(self, marker):
    """mujoco_viewer 0.1.4 的 _add_marker_to_scene 兼容版。

    mujoco 3.2.0 起 MjvGeom 移除了 texid/texuniform/texrepeat/emission/
    specular/shininess/reflectance 字段（改为 matid），原版会直接
    AttributeError。这里对所有默认字段加 hasattr 守卫，
    新旧 mujoco（2.3.6 / 3.4+）均可用。
    """
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError('Ran out of geoms. maxgeom: %d' % self.scn.maxgeom)

    g = self.scn.geoms[self.scn.ngeom]
    defaults = {
        'dataid': -1,
        'objtype': mujoco.mjtObj.mjOBJ_UNKNOWN,
        'objid': -1,
        'category': mujoco.mjtCatBit.mjCAT_DECOR,
        'texid': -1,
        'texuniform': 0,
        'emission': 0,
        'specular': 0.5,
        'shininess': 0.5,
        'reflectance': 0,
        'matid': -1,
        'type': mujoco.mjtGeom.mjGEOM_BOX,
    }
    for key, value in defaults.items():
        if hasattr(g, key):
            setattr(g, key, value)
    if hasattr(g, 'texrepeat'):
        g.texrepeat[:] = 1
    g.size[:] = np.ones(3) * 0.1
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)

    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == "label", "Only label is a string in mjtGeom."
            g.label = value
        elif hasattr(g, key):
            raise ValueError(
                "mjtGeom has attr {} but type {} is invalid".format(
                    key, type(value)))
        else:
            raise ValueError("mjtGeom doesn't have field %s" % key)

    self.scn.ngeom += 1


mujoco_viewer.MujocoViewer._add_marker_to_scene = _add_marker_to_scene_compat

class MujocoEnv():
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, sim_dt, control_dt):
        if os.path.isabs(model_path):
            fullpath = model_path
        else:
            raise Exception("Provide full path to robot description package.")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # set frame skip and sim dt
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        # 让相机进入“跟踪模式”，否则 trackbodyid 会被忽略，画面不会跟随机器人
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[0] = 1
        self.viewer._render_every_frame = True

    def viewer_is_paused(self):
        return self.viewer._paused

    # -----------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer_setup()
        self.viewer.render()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
