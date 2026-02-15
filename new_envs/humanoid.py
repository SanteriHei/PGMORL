import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box
from gymnasium import utils
from os import path

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 67
    }

    def __init__(self, **kwargs):
        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(path.abspath(path.dirname(__file__)), "assets/humanoid.xml"),
            frame_skip=5,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64),
            **kwargs)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self)
        obs = self._get_obs()
        data = self.data
        
        alive_bonus = 3.0
        reward_run = 1.25 * (pos_after - pos_before) / self.dt + alive_bonus
        reward_energy = 3.0 - 4.0 * np.square(data.ctrl).sum() + alive_bonus
        reward = np.array([reward_run, reward_energy])
        qpos = self.data.qpos
        terminated = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        info = {
            "reward_run": reward_run,
            "reward_energy": reward_energy,
            "obj": reward
        }
        
        return obs, np.array(0.0), terminated, False, info

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
