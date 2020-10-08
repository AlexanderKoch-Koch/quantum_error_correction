from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='SurfaceCode-v0',
    entry_point='qec.Environments:Surface_Code_Environment_Multi_Decoding_Cycles',
    # max_episode_steps=200,
    # reward_threshold=25.0,
)
