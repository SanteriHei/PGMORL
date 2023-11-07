from __future__ import annotations  # noqa


import os
import pathlib
from typing import Any, Dict

import mo_gymnasium as mo_gym
from mo_gymnasium import MOSyncVectorEnv
import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import StepAPICompatibility, VectorListInfo

from .file_monitor import Monitor
from .wrappers import RewardToInfoWrapper, TorchWrapper, NormalizeRewObs 




def make_eval_env(env_id: str, device: str): 
    env = mo_gym.make(env_id)
    env = RewardToInfoWrapper(env)
    env = TorchWrapper(env, device)
    env = StepAPICompatibility(env, output_truncation_bool=False)
    return env


def make_env(env_id, seed, rank, log_dir, allow_early_resets, env_params=None):
    def _init_env():
        # Convert the environments to use the old 'step' api
        env = RewardToInfoWrapper(mo_gym.make(env_id))

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

    return _init_env 


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
    ob_rms: bool = False,
    context: str | None = None,
    use_shared_memory: bool = False,
    daemonize: bool = False
):

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, env_params)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        # Forking ok under UNIX
        envs = AsyncVectorEnv(
                envs, context=context, daemon=daemonize,
                shared_memory=use_shared_memory
        )
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
        # Use the synchnorized environment if there is only one environment 
        # for easier debugging time
        envs = MOSyncVectorEnv(envs)
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
    
    # Make the env to use the 'old' style step and info API
    envs = VectorListInfo(envs)
    envs = StepAPICompatibility(envs, output_truncation_bool=False)
    return envs
