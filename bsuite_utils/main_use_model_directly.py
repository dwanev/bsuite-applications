#!/usr/bin/env python3
import bsuite
from bsuite.utils import gym_wrapper
from bsuite_utils.model_configs import ModelConfig
from stable_baselines3 import DQN, A2C
from bsuite_utils.mini_sweep import SWEEP, SWEEP_SETTINGS
from bsuite_utils.nace_based_model import NaceAlgorithm
import numpy as np


def use_a2c():
    bsuite_id = 'deep_sea/5'
    save_path = './tmp2/A2C_default'
    overwrite = True
    a2c_default = [ModelConfig(name="A2C_default", cls=A2C)]


    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)

    model_conf = ModelConfig(name='A2C_default', cls=A2C, policy='MlpPolicy', env_wrapper=None, kwargs={}, wrapper_kwargs={})
    model = model_conf.cls(policy=model_conf.policy, env=env, **model_conf.kwargs)

    exp_conf = SWEEP_SETTINGS[bsuite_id]
    # TODO: don't need both
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)


def use_nace():
    bsuite_id = 'deep_sea/5'
    save_path = './tmp2/nace_default'
    overwrite = True
    # nace_default = [ModelConfig(name="nace_default", cls=A2C)]

    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)
    env.render_mode = 'human'

    # env.render_mode("human")

    model_conf = ModelConfig(name='nace_default', cls=NaceAlgorithm, policy='MlpPolicy', env_wrapper=None, kwargs={},
                             wrapper_kwargs={})
    model = model_conf.cls(policy=model_conf.policy, env=env, **model_conf.kwargs) # these param go into the class constructor of the model

    exp_conf = SWEEP_SETTINGS[bsuite_id]
    # TODO: don't need both
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)


if __name__ == "__main__":



    print("bsuite_id list:", SWEEP_SETTINGS.keys())
    # use_a2c()
    use_nace()