# HalfCheetah-v2 env
# two objectives
# running speed, energy efficiency

from os import path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 20
    }
    def __init__(self, **kwargs):
        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(path.abspath(path.dirname(__file__)), "assets/half_cheetah.xml"),
            frame_skip=5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            ),
            **kwargs
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        xposafter, ang = self.data.qpos[0], self.data.qpos[2]
        ob = self._get_obs()
        alive_bonus = 1.0

        reward_run = (xposafter - xposbefore) / self.dt
        reward_run = min(4.0, reward_run) + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(action).sum() + alive_bonus

        
        # Old condition?
        # done = not (abs(ang) < np.deg2rad(50)) 
        done = False
        terminated = False  # Timelimit should be handled by the wrapper
        reward = np.array([reward_run, reward_energy])
        info = {
            "reward_energy": reward_energy,
            "reward_run": reward_run,
            "x_position": xposafter,
            "x_velocity": (xposafter - xposafter) / self.dt,
            "obj": reward
        }

        return ob, reward_run, terminated, done, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[1:],
                self.data.qvel.flat,
            ]
        )

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + c * self.np_random.standard_normal(self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == "__main__":
    env = HalfCheetahEnv()
    print(env.dt)
