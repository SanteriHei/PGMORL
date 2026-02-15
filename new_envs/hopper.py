# Hopper-v2 env
# two objectives
# running speed, jumping height

from os import path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 100
    }

    def __init__(
        self,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healhy_angle_range=(-0.2, 0.2),
        **kwargs
    ):
        self.reward_dim = 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=path.join(path.abspath(path.dirname(__file__)), "assets/hopper.xml"),
            frame_skip=5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
            ),
            **kwargs
        )
        utils.EzPickle.__init__(self)

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healhy_angle_range
        self._healthy_state_range = healthy_state_range

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, a):
        posbefore = self.data.qpos[0]
        a = np.clip(a, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward_others = alive_bonus - 2e-4 * np.square(a).sum()
        reward_run = 1.5 * (posafter - posbefore) / self.dt + reward_others
        reward_jump = 12.0 * (height - self.init_qpos[1]) + reward_others

        # NOTE: This was the old criterion for failing the env
        # s = self.state_vector()
        # done = not (
        #     (s[1] > 0.4)
        #     and abs(s[2]) < np.deg2rad(90)
        #     and abs(s[3]) < np.deg2rad(90)
        #     and abs(s[4]) < np.deg2rad(90)
        #     and abs(s[5]) < np.deg2rad(90)
        # )


        ob = self._get_obs()
        reward = np.array([reward_run, reward_jump])
        info = {
            "reward_run": reward_run,
            "reward_jump": reward_jump,
            "x_position": posafter,
            "x_velocity": (posafter - posbefore) / self.dt,
            "obj": reward
        }

        terminated = self.terminated

        return ob, reward_run, terminated, False, info

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos.flat[1:], np.clip(self.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        c = 1e-3
        new_qpos = self.init_qpos + self.np_random.uniform(
            low=-c, high=c, size=self.model.nq
        )
        new_qpos[1] = self.init_qpos[1]
        new_qvel = self.init_qvel + self.np_random.uniform(
            low=-c, high=c, size=self.model.nv
        )
        self.set_state(new_qpos, new_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    env = HopperEnv()
    print(env.dt)
