import ray
import time
import gym
import qec
ray.init()
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer
from models import ConvModel
import argparse
from ray.rllib.models import ModelCatalog
import tensorflow as tf
tf.compat.v1.enable_eager_execution()  # necessary for SurfaceCode env which uses a tf model

ModelCatalog.register_custom_model("QECConvModel", ConvModel)


def make_env():
    import qec
    import gym
    return gym.make('SurfaceCode-v0', channels_first=True)

tune.register_env('SurfaceCode-v0', lambda config: make_env())

if __name__ == '__main__':
    config = dict(env='SurfaceCode-v0',
                  framework='torch',
                  model=dict(custom_model='QECConvModel'),
                  exploration_config=dict(epsilon_timesteps=int(2e5)),
                  )
    trainer = DQNTrainer(config)
    trainer.restore(
        '/home/alex/ray_results/DQN_SurfaceCode-v0_2020-10-14_08-47-0492btz96a/checkpoint_1560/checkpoint-1560')
    env = gym.make('SurfaceCode-v0')
    while True:
        done = False
        obs = env.reset()
        step = 0
        return_ = 0
        while not done:
            step += 1
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            return_ += reward

        print(f'return: {return_} num steps {step}')
