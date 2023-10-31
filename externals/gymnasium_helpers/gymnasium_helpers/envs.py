from __future__ import annotations  # noqa


import os
import pathlib
from typing import Any, Dict

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import (
    #NormalizeObservation,
    #NormalizeReward,
    StepAPICompatibility,
)
from gymnasium.wrappers.normalize import RunningMeanStd

from .file_monitor import Monitor


class RewardToInfoWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    '''
    Moves the reward vector to the info object under "obj" key, similarly
    to the environments used by the original authors
    '''
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        assert "obj" not in info,\
            f"'info' already contains 'obj', {info['obj']}"
        info["obj"] = reward
        return observation, reward, done, info

class TorchWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    '''
    Implementation of the "VecPyTorch" wrapper from PGMORL externals in
    gymnasium API
    '''

    def __init__(self, env: gym.Env, device: torch.device | None = None):
        gym.utils.RecordConstructorArgs.__init__(self, device=device)
        gym.Wrapper.__init__(self, env)
        self.device = device if device is not None else torch.device("cpu")

        # Check if the environment is vectorized or not
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return self.env.step(action)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, terminated, _, info = self.env.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        return obs, rewards, terminated, info


class NormalizeRewObs(gym.Wrapper, gym.utils.RecordConstructorArgs):
    '''
    Wrapper that normalizes the observations and rewards similarly than 
    baselines VecNormalize
    '''

    def __init__(
            self, env, ob: bool = False, ret: bool = False, clipob=10.0,
            cliprew=10.0, gamma=0.99, epsilon=1e-8, obj_rms: bool = False
    ):
        gym.Wrapper.__init__(self, env)
        gym.utils.RecordConstructorArgs.__init__(
                self, ob=ob, ret=ret, clipob=clipob, cliprew=cliprew,
                gamma=gamma, epsilon=epsilon, obj_rms=obj_rms
        )
        self.training = True


        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.ob_rms = RunningMeanStd(
                shape=self.get_wrapper_attr("single_observation_space").shape)
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.obj_rms = RunningMeanStd(shape=()) if ret and obj_rms else None

        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.obj = np.array([None] * self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.env.step_wait()

        self.ret = self.ret * self.gamma + rews
        if "obj" in infos[0]:
            for info in infos:
                info["obj_raw"] = info["obj"]
            obj = np.array([info["obj"] for info in infos])
            self.obj = self.obj * self.gamma + \
                obj if self.obj[0] is not None else obj

        obs = self.obfilt(obs)

        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var * self.epsilon),
                a_min=-self.cliprew, a_max=self.cliprew
            )

        if self.obs_rms:
            self.obj_rms.update(self.obj)
            for info in infos:
                info["obj"] = np.clip(
                    info["obj"] / np.sqrt(self.obj_rms.var + self.epsilon),
                    -self.cliprew, self.cliprew
                )
        self.ret[news] = 0.0
        if "obj" in infos[0]:
            self.obj[news] = np.zeros_like(self.obj[news])
        return obs, rews, news, infos

    def _obfilt(self, obs, update: bool = True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                a_min=-self.clipob, a_max=self.clipob
            )
        return obs

    def train(self): 
        self.training = True

    def eval(self):
        self.training = False


def make_eval_env(env_id: str, device: str): 
    env = mo_gym.make(env_id)
    env = StepAPICompatibility(env, output_truncation_bool=False)
    env = RewardToInfoWrapper(env)
    env = TorchWrapper(env, device)
    return env


def make_env(env_id, seed, rank, log_dir, allow_early_resets, env_params=None):
    def _thunk():
        # Convert the environments to use the old 'step' api
        env = StepAPICompatibility(
            mo_gym.make(env_id), output_truncation_bool=False
        )
        env = RewardToInfoWrapper(env)

        if env_params:
            env.set_params(env_params)

        # seed cannot be set without resetting anymore.
        env.reset(seed=seed + rank)
        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets
            )
        else:
            env = Monitor(
                env,
                None,
                allow_early_resets=allow_early_resets
            )
        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    seed: int,
    num_processes: int,
    gamma: float | None,
    log_dir: str | pathlib.Path | None,
    device: torch.device | str,
    allow_early_resets: bool,
    env_params: Dict[str, Any] | None = None,
    obj_rms: bool = False,
    ob_rms: bool = False
):

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, env_params)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        # Forking ok under UNIX
        envs = AsyncVectorEnv(envs, context="fork")
        if len(envs.single_observation_space.shape) == 1:
            if gamma is None:
                envs = NormalizeRewObs(
                        envs, ret=False, obj_rms=obj_rms, ob=ob_rms
                )
            else:
                envs = NormalizeRewObs(
                        envs, gamma=gamma, obj_rms=obj_rms, ob=ob_rms
                )
    else:
        print("Adding reward and observation normalization")
        envs = envs[0]()
        if len(envs.observation_space.shape) == 1:
                
            if gamma is None:
                envs = NormalizeRewObs(
                        envs, ret=False, obj_rms=obj_rms, ob=ob_rms
                )
            else:
                envs = NormalizeRewObs(
                        envs, gamma=gamma, obj_rms=obj_rms, ob=ob_rms
                )
    envs = TorchWrapper(envs, device)
    return envs
