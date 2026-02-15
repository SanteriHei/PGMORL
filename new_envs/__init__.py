from gymnasium.envs.registration import register

register(
    id = 'MO-PGMORL-Ant-v4',
    entry_point = 'new_envs.ant:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-Hopper-v4',
    entry_point = 'new_envs.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-Hopper-v3',
    entry_point = 'new_envs.hopper_v3:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-HalfCheetah-v4',
    entry_point = 'new_envs.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-Walker2d-v4',
    entry_point = 'new_envs.walker2d:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-Swimmer-v4',
    entry_point = 'new_envs.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-PGMORL-Humanoid-v4',
    entry_point = 'new_envs.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)
