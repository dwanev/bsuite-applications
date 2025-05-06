#!/usr/bin/env python3
import bsuite
from bsuite.utils import gym_wrapper
from bsuite_utils.model_configs import ModelConfig
from stable_baselines3 import DQN, A2C
from bsuite_utils.mini_sweep import SWEEP, SWEEP_SETTINGS
from bsuite_utils.nace_based_model import NaceAlgorithm
import numpy as np

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')


def use_nace():
    # bsuite_id = 'deep_sea/5'
    # bsuite_id = 'catch/0'
    # bsuite_id = 'umbrella_length/5' # 6 (5+1) steps in total (last 5 steps have no effect), reward at end, with 20 distractor variables.
    # bsuite_id = 'umbrella_distract/5' # 6 (5+1) distractors, 20 steps before reward given.
    # bsuite_id = 'umbrella_distract/0' # 1 (0+1) distractors, 20 steps before reward given.
    # bsuite_id = 'umbrella_length/0'  # 20 distractors, 1 (0+1) steps before reward given.
    bsuite_id = 'bandit/0'

    save_path = './tmp_direct/NACE_default'
    overwrite = True
    # nace_default = [ModelConfig(name="nace_default", cls=NaceAlgorithm)]

    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)
    env.render_mode = 'human'

    # env.render_mode("human")

    model_conf = ModelConfig(name='nace_default', cls=NaceAlgorithm, policy='MlpPolicy', env_wrapper=None, kwargs={},
                             wrapper_kwargs={})

    model_conf.kwargs["context"] = bsuite_id
    model = model_conf.cls(policy=model_conf.policy, env=env, **model_conf.kwargs) # these param go into the class constructor of the model

    exp_conf = SWEEP_SETTINGS[bsuite_id]
    # TODO: don't need both
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)


if __name__ == "__main__":
    # print("bsuite_id list:", SWEEP_SETTINGS.keys())
    use_nace()

