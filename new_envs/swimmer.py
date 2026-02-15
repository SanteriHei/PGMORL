# Swimmer-v2 env
# two objectives
# forward speed, energy efficiency

from os import path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 25
    }
    def __init__(self, **kwargs):
        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(path.abspath(path.dirname(__file__)), "assets/swimmer.xml"),
            frame_skip=4,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            ),
            **kwargs
        )

    def step(self, a):
        ctrl_cost_coeff = 0.15
        xposbefore = self.data.qpos[0]
        a = np.clip(a, -1, 1)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = 0.3 - ctrl_cost_coeff * np.square(a).sum()
        ob = self._get_obs()

        reward = np.array([reward_fwd, reward_ctrl])

        info = {
            "reward_fwd": reward_fwd,
            "reward_ctrl": reward_ctrl,
            "x_position": xposafter,
            "x_velocity": (xposafter - xposbefore) / self.dt,
            "obj": reward
        }

        # Truncation = False, as the timelimit should be handled by the
        # "TimeLimit" wrapper added during make
        return ob, np.array(0.0), False, False, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return self._get_obs()


if __name__ == "__main__":
    env = SwimmerEnv()
    print(env.dt)
