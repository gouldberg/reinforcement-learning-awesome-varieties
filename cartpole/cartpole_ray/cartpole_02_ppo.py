# -*- coding: utf-8 -*-

import os, sys
import ray


# ----------
# REFERENCE
# Mastering Reinforcement Learning with Python, Chapter 7 (Policy-Based Methods)
# https://docs.ray.io/en/latest/rllib/rllib-training.html


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# set config
# -----------------------------------------------------------------------------------------------------------

import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print



# ----------
# starting local ray
ray.init()


# ----------
# default config
config = ppo.DEFAULT_CONFIG.copy()

for k, v in config.items():
    print(f'{k} : {v}')


# ----------
# Can optionally call algo.restore(path) to load a checkpoint.
algo = ppo.PPO(config=config, env="CartPole-v0")
print(algo)


# -----------------------------------------------------------------------------------------------------------
# train:  basic usage
# -----------------------------------------------------------------------------------------------------------

# for i in range(1000):
#    result = algo.train()
#    print(pretty_print(result))
#
#    if i % 100 == 0:
#        checkpoint = algo.save()
#        print("checkpoint saved at", checkpoint)
#
#
# ray.shutdown()


# -----------------------------------------------------------------------------------------------------------
# train:  basic usage
# -----------------------------------------------------------------------------------------------------------

from ray import air, tune

ray.init()

# Tune will schedule the trials to run in parallel on your Ray cluster:
tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"episode_reward_mean": 200},),
    param_space={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
).fit()


