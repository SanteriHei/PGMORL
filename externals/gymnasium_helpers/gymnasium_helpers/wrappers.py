from __future__ import annotations

import pprint

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.normalize import RunningMeanStd


class TimeLimitMask(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            print("'info' is none at TimeLimitMask")
        if truncated:
            info["bad_transition"] = True
        return obs, reward, terminated, truncated, info


class RewardToInfoWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    '''
    Moves the reward vector to the info object under "obj" key, similarly
    to the environments used by the original authors
    '''

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(
            action)
        assert "obj" not in info, \
            f"'info' already contains 'obj', {info['obj']}"
        info["obj"] = rewards
        return obs, rewards, terminated, truncated, info


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

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).double().to(self.device)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, rewards, terminated, truncated, info = self.env.step(action)
        obs = torch.from_numpy(obs).double().to(self.device)
        return obs, rewards, terminated, truncated, info

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step_wait()
        obs = torch.from_numpy(obs).double().to(self.device)
        rewards = torch.from_numpy(rewards).double().to(self.device)
        return obs, rewards, terminated, truncated, info


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

    def step(self, action):
        obs, rews, terminated, truncated, infos = self.env.step(action)
        x = self._process_step(obs, rews, terminated, truncated, infos)
        return x

    def step_wait(self):
        obs, rews, terminated, truncated, infos = self.env.step_wait()
        return self._process_step(obs, rews, terminated, truncated, infos)

    def _process_step(self, obs, rews, terminated, truncated, infos):
        # In the original multi-objective environments, the returns were always
        # zero. This was a simple hack to come around the fact that the gym
        # environments expected real valued rewards (i.e. the shape of rewards
        # was (num_envs, ) instead of (num_envs, num_objectives)). Thus, we
        # we just take the mean along the objectives to make the shapes correct
        # (This does not matter as the 'rewards' are not used, and instead the
        # 'obj' stored in the info is used.
        self.ret = self.ret * self.gamma + rews.mean(axis=1)
        if "obj" in infos:
            infos["obj_raw"] = infos["obj"].copy()
            # Required for VecInfoList Wrapper to work
            infos["_obj_raw"] = infos["_obj"].copy()
            obj = infos["obj"]
            if self.obj[0] is not None:
                # For some reason, some of the values in "obj" might be None (sync problem?)
                # So just check which values are available and update them.
                for i in range(len(obj)):
                    if obj[i] is not None:
                        self.obj[i] = self.obj[i] * self.gamma + obj[i]
            else:
                self.obj = obj

        obs = self._obfilt(obs)

        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var * self.epsilon),
                a_min=-self.cliprew, a_max=self.cliprew
            )

        if self.obj_rms:
            self.obj_rms.update(self.obj)
            
            # if this the final step, there is no "obj". 
            # Instead, it is stored i the final_info
            if "final_info" in infos:
                for i in range(len(infos["final_info"])):
                    if infos["final_info"][i]["obj"] is None:
                        continue

                    infos["final_info"][i]["obj"] = np.clip(
                        infos["final_info"][i]["obj"] / np.sqrt(self.obj_rms.var + self.epsilon),
                        -self.cliprew, self.cliprew
                    )
            else:
                for i, obj in enumerate(infos["obj"]):
                    # As this is done async, some of the objects might be missing
                    if obj is None:
                        continue

                    infos["obj"][i] = np.clip(
                        obj / np.sqrt(self.obj_rms.var + self.epsilon),
                        -self.cliprew, self.cliprew
                    )
            # for info in infos:
            #     info["obj"] = np.clip(
            #         info["obj"] / np.sqrt(self.obj_rms.var + self.epsilon),
            #         -self.cliprew, self.cliprew
            #     )
        self.ret[terminated] = 0.0
        if "obj" in infos:
            self.obj[terminated] = np.zeros_like(self.obj[terminated])
        return obs, rews, terminated, truncated, infos

    def _obfilt(self, obs, update: bool = True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) /
                np.sqrt(self.ob_rms.var + self.epsilon),
                a_min=-self.clipob, a_max=self.clipob
            )
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
