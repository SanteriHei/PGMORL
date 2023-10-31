
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.normalize import RunningMeanStd


class RewardToInfoWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    '''
    Moves the reward vector to the info object under "obj" key, similarly
    to the environments used by the original authors
    '''

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        assert "obj" not in info, \
            f"'info' already contains 'obj', {info['obj']}"
        info["obj"] = reward
        return observation, reward, terminated, truncated, info


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
            obs = torch.from_numpy(obs).to(self.device)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, rewards, terminated, truncated, info = self.env.step(action)

        print(info)
        obs = torch.from_numpy(obs).to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        return obs, rewards, terminated, truncated, info

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
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

    def step_wait(self):
        obs, rews, terminated, truncated, infos = self.env.step_wait()

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
        self.ret[terminated] = 0.0
        if "obj" in infos[0]:
            self.obj[terminated] = np.zeros_like(self.obj[terminated])
        return obs, rews, terminated, truncated, infos

    def _obfilt(self, obs, update: bool = True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) /
                np.sqrt(self.obs_rms.var + self.epsilon),
                a_min=-self.clipob, a_max=self.clipob
            )
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
