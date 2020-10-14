import argparse
import gym
import torch
import time
import numpy as np
from gym.wrappers.monitor import Monitor
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent

from rlpyt.envs.gym import GymEnvWrapper
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
# from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.async_rl import AsyncRlEval

def simulate_policy(env, agent, render):
    obs = env.reset()
    observation = buffer_from_example(obs, 1)
    loop_time = 0.01
    returns = []
    mses = []
    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        agent.reset()
        done = False
        step = 0
        reward_sum = 0
        # env.render()
        # time.sleep(1.1)
        forward_reward = 0
        while not done:
            loop_start = time.time()
            step += 1
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)[0]
            # print('action: '+ str(action))
            # action = env.oracle_policy.get_action(obs.state)
            # mse = ((env.oracle_policy.get_action(obs.state) - action) ** 2).sum()
            # mse = 0
            # print('mse: '+ str(mse))
            # mses.append(mse)

            # action = np.argmax(observation[0].demonstration_actions)
            # print(np.argmax(obs_pyt[0].demonstration_actions) == action)
            obs, reward, done, info = env.step(action)
            # forward_reward += info.forward_reward
            reward_sum += reward
            # print('reward: ' + str(reward))
            observation[0] = obs
            rew_pyt[0] = float(reward)
            sleep_time = loop_time - (time.time() - loop_start)
            sleep_time = 0 if (sleep_time < 0) else sleep_time
            if render:
                time.sleep(sleep_time)
                env.render(mode='human')

        # print('episode success: ' + str(info.episode_success))
        returns.append(reward_sum)
        print('avg return: ' + str(sum(returns) / len(returns)) + ' return: ' + str(reward_sum) + '  num_steps: ' + str(
            step))
        # print(f'forward reward: {forward_reward}')
        # print(' avg mse: ' + str(sum(mses) / len(mses)))


def make_env(**kwargs):
    info_example = {'timeout': 0}
    import qec
    env = gym.make(**kwargs)
    env =  GymEnvWrapper(EnvInfoWrapper(env, info_example))
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        # default='/home/alex/important_logs/transformer_ml3/params.pkl')
                        default='/home/alex/quantum_error_correction/logs/run_9/params.pkl')
    parser.add_argument('--env', default='HumanoidPrimitivePretraining-v0',
                        choices=['HumanoidPrimitivePretraining-v0', 'TrackEnv-v0'])
    parser.add_argument('--algo', default='ppo', choices=['sac', 'ppo'])
    args = parser.parse_args()

    snapshot = torch.load(args.path, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = make_env(id='SurfaceCode-v0', error_model='X', volume_depth=5)
    agent = AtariDqnAgent(model_kwargs=dict(channels=[64, 32, 32],
                                            kernel_sizes=[3, 2, 2],
                                            strides=[2, 1, 1],
                                            paddings=[0, 0, 0],
                                            fc_sizes=[512, ],
                                            dueling=True),
                          eps_eval=0.001)
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict['model'])
    agent.eval_mode(1)
    # agent.sample_mode(0)
    from rlpyt.samplers.serial.collectors import SerialEvalCollector
    from rlpyt.samplers.collections import TrajInfo
    eval_collector = SerialEvalCollector(envs=[env],
                                         agent=agent,
                                         TrajInfoCls=TrajInfo,
                                         max_T=10000)
    # x  = eval_collector.collect_evaluation(0)
    simulate_policy(env, agent, render=False)
