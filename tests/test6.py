import bsuite
import dm_env
from gym.spaces import Discrete
import gym
import numpy as np

from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis
from ray.tune.logger import pretty_print


SAVE_PATH_DQN = 'reports/bsuite/rllib'
raw_env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_DQN, overwrite=True)
env = gym_wrapper.GymFromDMEnv(raw_env)


class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym_wrapper.GymFromDMEnv(bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_DQN, overwrite=True))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        pass


# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = gym.spaces.Discrete(2)
#         self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4))
#
#     def step(self, action):
#         return np.array([[-0.5, 0.5, 0.2, -0.1]]), 1, True, {}
#
#     def reset(self):
#         return np.array([[-0.5, 0.5, 0.2, -0.1]])
#
#     def render(self, mode="human"):
#         pass


from ray.tune.registry import register_env
def env_creator(env_config):
    return MyEnv(env_config)

register_env("h", env_creator)

from ray.rllib.algorithms.dqn import DQNConfig

config = (  # 1. Configure the algorithm,
    DQNConfig()
    .environment("h")
    .rollouts(num_rollout_workers=2)
    .framework("tf2")
    .training(gamma=0.99, lr=1e-3, model={"fcnet_hiddens": [64, 64]})
    .exploration()
)

config.exploration_config.update({"initial_epsilon": 1.0, "final_epsilon" : 0.01, "epsilon_timesteps" : 5000})

print(pretty_print(config.to_dict()))

algo = config.build()  # 2. build the algorithm,


import pandas as pd

for i in range(1000):
    print("Training iteration: ", i)
    algo.train()  # 3. train it,
    results = pd.read_csv('reports/bsuite/rllib/bsuite_id_-_bandit_noise-0.csv')
    csv_len = len(results)
    if csv_len >= 49:
        diff = csv_len - 49
        if diff > 0:
            results.drop(results.tail(diff).index, inplace=True)
        break


from bsuite.logging import csv_load
DF, _ = csv_load.load_bsuite(SAVE_PATH_DQN)
BSUITE_SCORE = summary_analysis.bsuite_score(DF)
print(BSUITE_SCORE)