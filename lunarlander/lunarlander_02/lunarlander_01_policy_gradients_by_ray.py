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

import ray.rllib.algorithms.pg as pg
from ray.tune.logger import pretty_print

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'
log_dir = os.path.join(base_path, '04_output\\lunarlander\\logs\\pg_ray')


# ----------
# starting local ray
ray.init()


# ----------
# default config
config = pg.DEFAULT_CONFIG.copy()

for k, v in config.items():
    print(f'{k} : {v}')


# ----------
# update config
config["num_gpus"] = 0
config["num_workers"] = 1
# config["evaluation_num_workers"] = 2
# config["evaluation_interval"] = 1


# ----------
# Can optionally call algo.restore(path) to load a checkpoint.
algo = pg.PG(config=config, env="LunarLanderContinuous-v2")
print(algo)


# -----------------------------------------------------------------------------------------------------------
# train:  basic usage
# -----------------------------------------------------------------------------------------------------------

for i in range(1000):
   result = algo.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = algo.save()
       print("checkpoint saved at", checkpoint)


ray.shutdown()

