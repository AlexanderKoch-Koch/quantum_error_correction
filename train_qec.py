import ray
import time

s = time.time()
ray.init()
print(f'ray init took :{time.time() - s}')
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer
from models import ConvModel
import argparse
import tensorflow as tf
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("QECConvModel", ConvModel)


def make_env():
    import qec
    import gym
    return gym.make('SurfaceCode-v0', channels_first=True, error_model='X', volume_depth=5)


tune.register_env('SurfaceCode-v0', lambda config: make_env())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--logdir', required=False, default='./logs',
                        help='path to directory where log folder will be; Overwrites log_dir_positional')

    args = parser.parse_args()
    config = dict(env='SurfaceCode-v0',
                  framework='torch',
                  num_workers=7,
                  evaluation_interval=10,
                  evaluation_config=dict(explore=False),
                  model=dict(custom_model='QECConvModel'),
                  prioritized_replay=False,
                  target_network_update_freq=5000,
                  buffer_size=int(5e4),
                  lr=5e-5,
                  # n_step=3,
                  exploration_config=dict(epsilon_timesteps=int(2e5),
                                          initial_epsilon=1.0,
                                          final_epsilon=0.001),
                  # timesteps_per_iteration=int(3e3)
                  # model={"dim": 11, "conv_filters": [[64, [3, 3], 2], [128, [2, 2], 2], [512, [2, 2], 2]],
                  #        "fcnet_activation": "relu"}
                  )
    # tune.run(DQNTrainer, config=config, local_dir=args.logdir)
    tf.compat.v1.enable_eager_execution()  # necessary for SurfaceCode env which uses a tf model
    trainer = DQNTrainer(config)
    for i in range(int(1e10)):
        result = trainer.train()
        if 'evaluation' in result.keys():
            print(result['evaluation'])
            additional_logs = dict(
                timesteps_total=result['timesteps_total'],
                evaluation=result['evaluation']
            )
            trainer.log_result(additional_logs)
            trainer.save(trainer.logdir)
