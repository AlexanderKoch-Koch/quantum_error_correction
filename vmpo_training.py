import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from rlpyt.envs.gym import GymEnvWrapper
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from traj_info import EnvInfoTrajInfo
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.utils.logging.context import logger_context
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
import torch
from rlpyt_models import QECModel, VmpoQECModel
from qec.Environments import Surface_Code_Environment_Multi_Decoding_Cycles
from imitation_learning.vmpo.async_vmpo import AsyncVMPO
from imitation_learning.vmpo.v_mpo import VMPO
from imitation_learning.vmpo.multivariate_gaussian_agent import MujocoVmpoAgent
from qec_vmpo_agent import QECVmpoAgent


def build_and_train(id="SurfaceCode-v0", name='run', log_dir='./logs'):
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=8,  # Use 16 cores across all experiments.
        cpu_per_run=8,
        n_gpu=0,  # Use 8 gpus across all experiments.
        # sample_gpu_per_run=0,
        async_sample=False,
        alternating=False
    )
    env_kwargs = dict(id='SurfaceCode-v0', error_model='X', volume_depth=5)
    state_dict = torch.load('./logs/run_12/params.pkl', map_location='cpu')
    agent_state_dict = None #state_dict['agent_state_dict']
    optim_state_dict = None #state_dict['optimizer_state_dict']

    # sampler = AsyncCpuSampler(
    sampler = CpuSampler(
        # sampler=SerialSampler(
        EnvCls=make_gym_env,
        # TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(id=id),
        batch_T=40,
        batch_B=64,
        max_decorrelation_steps=100,
        eval_env_kwargs=dict(id=id, fixed_episode_length=500),
        eval_n_envs=1,
        eval_max_steps=int(1e5),
        eval_max_trajectories=5,
        TrajInfoCls=EnvInfoTrajInfo
    )
    # algo = AsyncVMPO(batch_B=32, batch_T=40)
    algo = VMPO(pop_art_reward_normalization=True, discrete_actions=True, T_target_steps=40)
    agent = QECVmpoAgent(ModelCls=VmpoQECModel, model_kwargs=dict(linear_value_output=False))
    # runner = AsyncRlEval(
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e8,
        log_interval_steps=3e4,
        affinity=affinity,
    )
    config = dict(game=id)
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    # with logger_context(log_dir, run_ID, name, config):
    runner.train()


def make_gym_env(**kwargs):
    import qec
    info_example = {'timeout': 0}
    # print('making env: ' + str(kwargs))
    static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_X_p5'
    # from keras.models import load_model
    # static_decoder = load_model(static_decoder_path)
    if 'fixed_episode_length' in kwargs.keys():
        fixed_episode_length = kwargs['fixed_episode_length']
        kwargs.pop('fixed_episode_length')
    else:
        fixed_episode_length = None

    env = Surface_Code_Environment_Multi_Decoding_Cycles(error_model='DP', volume_depth=5, p_meas=0.011, p_phys=0.011)
    # env = gym.make(**kwargs)
    # env = FixedLengthEnvWrapper(env, fixed_episode_length=fixed_episode_length)
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
