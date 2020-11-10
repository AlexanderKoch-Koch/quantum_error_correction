import os
import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from rlpyt.envs.gym import GymEnvWrapper
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from traj_info import EnvInfoTrajInfo
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
import torch
from rlpyt_models import QECModel, VmpoQECModel, RecurrentVmpoQECModel
from qec.Environments import Surface_Code_Environment_Multi_Decoding_Cycles
from imitation_learning.vmpo.async_vmpo import AsyncVMPO
from imitation_learning.vmpo.v_mpo import VMPO
from imitation_learning.vmpo.categorical_vmpo_agent import CategoricalVmpoAgent
from imitation_learning.vmpo.categorical_models import CategorialFfModel
from qec_vmpo_agent import QECVmpoAgent
from qec.qec_collectors import QecCpuEvalCollector, QecDbCpuResetCollector
from qec.optimized_environment import OptimizedSurfaceCodeEnvironment
from qec.general_environment import GeneralSurfaceCodeEnv
from qec.fixed_length_env_wrapper import FixedLengthEnvWrapper


def build_and_train(id="SurfaceCode-v0", name='run', log_dir='./logs', async_mode=True, restore_path=None):
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=24,  # Use 16 cores across all experiments.
        cpu_per_run=24,
        n_gpu=1,  # Use 8 gpus across all experiments.
        async_sample=async_mode,
        alternating=False
    )
    agent_state_dict = optim_state_dict = None
    if restore_path is not None:
        state_dict = torch.load(restore_path, map_location='cpu')
        agent_state_dict = state_dict['agent_state_dict']
        optim_state_dict = state_dict['optimizer_state_dict']
    if async_mode:
        SamplerCls = AsyncCpuSampler
        RunnerCls = AsyncRlEval
        algo = AsyncVMPO(batch_B=64, batch_T=40, discrete_actions=True, T_target_steps=40, epochs=4, initial_optim_state_dict=optim_state_dict)
        sampler_kwargs=dict(CollectorCls=QecDbCpuResetCollector, eval_CollectorCls=QecCpuEvalCollector)
    else:
        SamplerCls = SerialSampler
        RunnerCls = MinibatchRlEval
        algo = VMPO(discrete_actions=True, epochs=4, minibatches=16, T_target_steps=10, initial_optim_state_dict=optim_state_dict)
        sampler_kwargs = dict()

    env_kwargs = dict(error_model='DP', error_rate=0.011)

    sampler = SamplerCls(
        EnvCls=make_qec_env,
        # TrajInfoCls=AtariTrajInfo,
        env_kwargs=env_kwargs,
        batch_T=40,
        batch_B=23 * 32,
        max_decorrelation_steps=100,
        eval_env_kwargs=env_kwargs,
        eval_n_envs=23,
        eval_max_steps=int(1e5),
        eval_max_trajectories=23 * 1,
        TrajInfoCls=EnvInfoTrajInfo,
        **sampler_kwargs
    )
    # agent = CategoricalVmpoAgent(ModelCls=RecurrentVmpoQECModel, model_kwargs=dict(linear_value_output=False), initial_model_state_dict=agent_state_dict)
    agent = CategoricalVmpoAgent(ModelCls=VmpoQECModel, model_kwargs=dict(linear_value_output=False), initial_model_state_dict=agent_state_dict)
    runner = RunnerCls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e10,
        log_interval_steps=1e6,
        affinity=affinity,

    )
    config = dict(game=id)
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    runner.train()

def make_qec_env(error_model, error_rate, volume_depth=5):
    env = OptimizedSurfaceCodeEnvironment(error_model=error_model, volume_depth=volume_depth,
                                          p_meas=error_rate, p_phys=error_rate)
    return GymEnvWrapper(env)


def make_gym_env(error_model, **kwargs):
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
        fixed_episode_length = None

    # env = Surface_Code_Environment_Multi_Decoding_Cycles(error_model='X', volume_depth=5, p_meas=0.001, p_phys=0.001)
    env = OptimizedSurfaceCodeEnvironment(error_model='DP', volume_depth=5, p_meas=0.011, p_phys=0.011)
    # env = GeneralSurfaceCodeEnv(error_model='DP', p_meas=0.011, p_phys=0.011)
    # env = gym.make('CartPole-v0')
    # env = FixedLengthEnvWrapper(env, fixed_episode_length=fixed_episode_length)
    # return GymEnvWrapper(EnvInfoWrapper(env, info_example))
    return GymEnvWrapper(env)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of the run', type=str, default='run')
    parser.add_argument('--log_dir', help='log dir', type=str, default='./logs')
    parser.add_argument('--restore', help='path to old .pkl file', type=str, default=None)
    args = parser.parse_args()
    build_and_train(
        name=args.name,
        restore_path=args.restore,
        log_dir=args.log_dir
    )
