import argparse
import torch
import time
import numpy as np
from qec.Environments import Surface_Code_Environment_Multi_Decoding_Cycles
from rlpyt.envs.gym import EnvInfoWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.envs.gym import GymEnvWrapper
from qec.rlpyt_models import VmpoQECModel
from imitation_learning.vmpo.categorical_vmpo_agent import CategoricalVmpoAgent
from keras.models import load_model


def simulate_policy(env, agent, render):
    static_decoder_path = './qec/referee_decoders/nn_d5_DP_p5'
    static_decoder = load_model(static_decoder_path, compile=True)
    obs = env.reset()
    observation = buffer_from_example(obs, 1)
    loop_time = 0.01
    returns = []
    mses = []
    lifetimes = []
    while True:
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        agent.reset()
        done = False
        step = 0
        reward_sum = 0
        while not done:
            loop_start = time.time()
            step += 1
            act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)[0]
            obs, reward, done, info = env.step(action)
            # done = np.argmax(static_decoder(info.static_decoder_input)[0]) != info.correct_label
            reward_sum += reward
            observation[0] = obs
            rew_pyt[0] = float(reward)

        returns.append(reward_sum)
        lifetimes.append(info.lifetime)
        print('avg return: ' + str(sum(returns) / len(returns)) + ' return: ' + str(reward_sum) + '  num_steps: ' + str(
            step))
        print(f'average lifetime: {sum(lifetimes)/len(lifetimes)} lifetime: {info.lifetime}')


def make_env(**kwargs):
    info_example = {'timeout': 0}
    # env = gym.make('CartPole-v0')
    env = Surface_Code_Environment_Multi_Decoding_Cycles(error_model='DP', volume_depth=5, p_meas=0.011, p_phys=0.011, use_Y=False)
    # env = OptimizedSurfaceCodeEnvironment(error_model='X', volume_depth=5, p_meas=0.011, p_phys=0.011)
    # env = GeneralSurfaceCodeEnv(error_model='DP', p_meas=0.011, p_phys=0.011, use_Y=False)
    env =  GymEnvWrapper(EnvInfoWrapper(env, info_example))
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        # default='/home/alex/important_logs/transformer_ml3/params.pkl')
                        default='./logs/vmpo_0/params.pkl')
    parser.add_argument('--env', default='HumanoidPrimitivePretraining-v0',
                        choices=['HumanoidPrimitivePretraining-v0', 'TrackEnv-v0'])
    parser.add_argument('--algo', default='ppo', choices=['sac', 'ppo'])
    args = parser.parse_args()

    snapshot = torch.load(args.path, map_location=torch.device('cpu'))
    # agent_state_dict = snapshot['agent_state_dict']
    agent_state_dict = snapshot['agent_state_dict']
    env = make_env()
    # agent = AtariDqnAgent(model_kwargs=dict(channels=[32, 64, 64],
    #                                         kernel_sizes=[3, 2, 2],
    #                                         strides=[2, 1, 1],
    #                                         paddings=[0, 0, 0],
    #                                         fc_sizes=[512, ],
    #                                         dueling=True),
    #                       ModelCls=QECModel,
    #                       eps_init=1,
    #                       eps_final=0.02,
    #                       eps_itr_max=int(5e6),
    #                       eps_eval=0)
    # agent = CategoricalVmpoAgent(ModelCls=CategorialFfModel, model_kwargs=dict(linear_value_output=False))
    agent = CategoricalVmpoAgent(ModelCls=VmpoQECModel, model_kwargs=dict(linear_value_output=False))
    #agent = CategoricalVmpoAgent(ModelCls=SingleActionRecurrentQECModel,
    #                             model_kwargs=dict(linear_value_output=False, lstm_layers=1),
    #                             initial_model_state_dict=agent_state_dict)
    # agent = MultiActionVmpoAgent(ModelCls=MultiActionRecurrentQECModel,
    #                              model_kwargs=dict(linear_value_output=False, lstm_layers=1),
    #                              initial_model_state_dict=agent_state_dict)
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
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
