import os
import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import rlpyt
from rlpyt.envs.gym import GymEnvWrapper
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from traj_info import EnvInfoTrajInfo
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
import torch
from qec.recurrent_models import MultiActionRecurrentQECModel
from imitation_learning.vmpo.async_vmpo import AsyncVMPO
from imitation_learning.vmpo.v_mpo import VMPO
from qec.qec_collectors import QecCpuEvalCollector, QecDbCpuResetCollector, QecCpuResetCollector
from qec.optimized_environment import OptimizedSurfaceCodeEnvironment
from qec.general_environment import GeneralSurfaceCodeEnv
from qec.multi_action_vmpo_agent import MultiActionVmpoAgent
import GPUtil
import multiprocessing
from qec.synchronous_runner import QECSynchronousRunner


def build_and_train(id="SurfaceCode-v0", name='run', log_dir='./logs', async_mode=False, restore_path=None):
    num_cpus = multiprocessing.cpu_count()
    num_gpus = 0 #len(GPUtil.getGPUs())
    # print(f"num cpus {num_cpus} num gpus {num_gpus}")
    if num_gpus == 0:
        # affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=0, set_affinity=False)
        affinity = make_affinity(n_cpu_core=num_cpus//2, cpu_per_run=num_cpus//2, n_gpu=num_gpus, async_sample=False,
                                 set_affinity=True)
        affinity['workers_cpus'] = tuple(range(num_cpus))
        affinity['master_torch_threads'] = 28
    else:
        affinity = make_affinity(
            run_slot=0,
            n_cpu_core=num_cpus,  # Use 16 cores across all experiments.
            cpu_per_run=num_cpus,#24,
            n_gpu=num_gpus,  # Use 8 gpus across all experiments.
            async_sample=async_mode,
            alternating=False,
            set_affinity=True,
        )
    # num_worker_cpus = len(affinity.sampler['workers_cpus'])
    print(f'affinity: {affinity}')
    agent_state_dict = optim_state_dict = None
    if restore_path is not None:
        state_dict = torch.load(restore_path, map_location='cpu')
        agent_state_dict = state_dict['agent_state_dict']
        optim_state_dict = state_dict['optimizer_state_dict']
    if async_mode:
        SamplerCls = AsyncCpuSampler
        RunnerCls = AsyncRlEval
        algo = AsyncVMPO(batch_B=64, batch_T=40, discrete_actions=True, T_target_steps=40, epochs=1, initial_optim_state_dict=optim_state_dict)
        sampler_kwargs=dict(CollectorCls=QecDbCpuResetCollector, eval_CollectorCls=QecCpuEvalCollector)
    else:
        SamplerCls = CpuSampler
        # SamplerCls = SerialSampler
        # RunnerCls = MinibatchRlEval
        RunnerCls = QECSynchronousRunner
        algo = VMPO(discrete_actions=True, epochs=4, minibatches=100, initial_optim_state_dict=optim_state_dict, epsilon_alpha=0.01)
        sampler_kwargs=dict(CollectorCls=QecCpuResetCollector, eval_CollectorCls=QecCpuEvalCollector)

    env_kwargs = dict(error_model='DP', error_rate=0.005, volume_depth=1)

    sampler = SamplerCls(
        EnvCls=make_qec_env,
        env_kwargs=env_kwargs,
        batch_T=40,
        batch_B=64 * 100,
        max_decorrelation_steps=50,
        eval_env_kwargs=env_kwargs,
        eval_n_envs=num_cpus,
        eval_max_steps=int(1e6),
        eval_max_trajectories=num_cpus,
        TrajInfoCls=EnvInfoTrajInfo,
        **sampler_kwargs
    )
    agent = MultiActionVmpoAgent(ModelCls=MultiActionRecurrentQECModel,
                                 model_kwargs=dict(linear_value_output=False),
                                 initial_model_state_dict=agent_state_dict)
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
    # env = OptimizedSurfaceCodeEnvironment(error_model=error_model, volume_depth=volume_depth,
    #                                       p_meas=error_rate, p_phys=error_rate)
    env = GeneralSurfaceCodeEnv(error_model=error_model, p_meas=error_rate, p_phys=error_rate, use_Y=False)
    # env = gym.make('CartPole-v0')
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
