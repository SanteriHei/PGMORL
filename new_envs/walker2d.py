# Walker2d-v2 env
# two objectives
# running speed, energy efficiency

from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Implementation of multi-objective Walker2d env from PGMORL
    in mo-gymnasium

    """

    metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 125
    }

    def __init__(
        self,
        terminated_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        **kwargs
    ):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(path.abspath(path.dirname(__file__)), "assets/walker2d.xml"),
            frame_skip=4,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(17,)),
            **kwargs
        )

        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._terminate_when_unhealthy = terminated_when_unhealthy

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, a):
        # qpos0_sum = np.sum(self.sim.data.qpos)
        # qvel0_sum = np.sum(self.sim.data.qvel)
        posbefore = self.data.qpos[0]
        a = np.clip(a, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward_speed = (posafter - posbefore) / self.dt + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(a).sum() + alive_bonus
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()


        reward = np.array([reward_speed, reward_energy])
        info = {
            "reward_speed": reward_speed,
            "reward_energy": reward_energy,
            "x_position": posafter,
            "x_velocity": (posafter - posbefore) / self.dt,
            "obj": reward
        }

        # Timelimit handled by the wrapper added during make
        terminated = self.terminated
        return ob, np.array(0.0), terminated, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    env = Walker2dEnv()
    print(env.dt)
