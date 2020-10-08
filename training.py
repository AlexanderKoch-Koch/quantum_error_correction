import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from rlpyt.envs.gym import GymEnvWrapper
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
# from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.async_rl import AsyncRlEval
from qec.fixed_length_env_wrapper import FixedLengthEnvWrapper
from traj_info import EnvInfoTrajInfo
from rlpyt.utils.logging.context import logger_context
# from imitation_learning.environments.env_info_wrapper import EnvInfoWrapper
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
import qec
import gym

def build_and_train(id="SurfaceCode-v0", name='run', log_dir='./logs'):
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=4,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        sample_gpu_per_run=0,
        async_sample=True,
        set_affinity=False,
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    # sampler = AsyncCpuSampler(
    sampler=SerialSampler(
        EnvCls=make_gym_env,
        # TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(id=id),
        batch_T=10,
        batch_B=64,
        max_decorrelation_steps=100,
        eval_env_kwargs=dict(id=id, fixed_episode_length=500),
        eval_n_envs=8,
        eval_max_steps=int(10e3),
        eval_max_trajectories=8,
        TrajInfoCls=EnvInfoTrajInfo
    )
    algo = DQN(
        replay_ratio=64,
        learning_rate=1e-4,
        min_steps_learn=1e4,
        replay_size=int(1e6),
        batch_size=32,
        double_dqn=True
    )
    agent = AtariDqnAgent(model_kwargs=dict(channels=[64, 32, 32],
                                            kernel_sizes=[3, 2, 2],
                                            strides=[2, 1, 1],
                                            paddings=[0, 0, 0],
                                            fc_sizes=[512, ]))
    # runner = AsyncRlEval(
    runner=MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e8,
        log_interval_steps=2e4,
        affinity=affinity,
    )
    config = dict(game=id)
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    # with logger_context(log_dir, run_ID, name, config):
    runner.train()

def make_gym_env(**kwargs):
    import qec
    info_example = {'timeout': 0}
    print('making env: ' + str(kwargs))
    static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_X_p5'
    # from keras.models import load_model
    # static_decoder = load_model(static_decoder_path)
    if 'fixed_episode_length' in kwargs.keys():
        fixed_episode_length = kwargs['fixed_episode_length']
        kwargs.pop('fixed_episode_length')
    else:
        fixed_episode_length=None

    env = FixedLengthEnvWrapper(gym.make(**kwargs), fixed_episode_length=fixed_episode_length)
    # return GymEnvWrapper(EnvInfoWrapper(env, info_example))
    return GymEnvWrapper(env)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of the run', type=str, default='run')
    parser.add_argument('--log_dir', help='log dir', type=str, default='./logs')
    args = parser.parse_args()
    build_and_train(
        name=args.name,
        log_dir=args.log_dir
    )
