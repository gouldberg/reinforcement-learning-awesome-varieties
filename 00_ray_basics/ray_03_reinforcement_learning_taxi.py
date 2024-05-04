# -*- coding: utf-8 -*-

import os, sys
import ray


# ----------
# REFERENCE
#https://docs.ray.io/en/latest/rllib/index.html
# REQUIRED:  pip install "gym[atari]" "gym[accept-rom-license]" atari_py


# -----------------------------------------------------------------------------------------------------------
# import RL algorithm and configure
# -----------------------------------------------------------------------------------------------------------

from ray.rllib.algorithms.ppo import PPO


# Configure the algorithm.

config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "Taxi-v3",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}


# -----------------------------------------------------------------------------------------------------------
# Create RLlib Trainer
# -----------------------------------------------------------------------------------------------------------

algo = PPO(config=config)


# -----------------------------------------------------------------------------------------------------------
# Run and evaluate
# -----------------------------------------------------------------------------------------------------------

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.

for _ in range(3):
    print(algo.train())



# Evaluate the trained Trainer (and render each timestep to the shell's output).
algo.evaluate()

