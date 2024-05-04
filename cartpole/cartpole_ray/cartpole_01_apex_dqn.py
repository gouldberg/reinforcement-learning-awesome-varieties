# -*- coding: utf-8 -*-

import os, sys
import datetime
import time
import numpy as np
from collections import deque

import gym
import ray

import tensorflow as tf
tf.get_logger().setLevel('WARNING')


# ----------
# REFERENCE
# Mastering Reinforcement Learning with Python, Chapter 6 (Ray Implementation of a DQN variate)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Actor
#   - responsible for collecting experiences from its environment given an exploratory policy
#   - stores the experiences locally before pushing it to the replay buffer to reduce the communication overhead
#   - differentiate between a training and evaluation actor since we run the sampling step
#     for the evaluation actors only for a single episode
#   - periodically pull the latest Q network weights to update their policies
# -----------------------------------------------------------------------------------------------------------

@ray.remote
class Actor:
    def __init__(self,
                 actor_id,
                 replay_buffer,
                 parameter_server,
                 config,
                 eps,
                 eval=False):
        self.actor_id = actor_id
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.config = config
        self.eps = eps
        self.eval = eval
        self.Q = get_Q_network(config)
        self.env = gym.make(config["env"])
        self.local_buffer = []
        self.obs_shape = config["obs_shape"]
        self.n_actions = config["n_actions"]
        self.multi_step_n = config.get("n_step", 1)
        self.q_update_freq = config.get("q_update_freq", 100)
        self.send_experience_freq = config.get("send_experience_freq", 100)
        self.continue_sampling = True
        self.cur_episodes = 0
        self.cur_steps = 0

    def update_q_network(self):
        if self.eval:
            pid = self.parameter_server.get_eval_weights.remote()
        else:
            pid = self.parameter_server.get_weights.remote()
        new_weights = ray.get(pid)
        if new_weights:
            self.Q.set_weights(new_weights)
        else:
            print("Weights are not available yet, skipping.")

    def get_action(self, observation):
        observation = observation.reshape((1, -1))
        q_estimates = self.Q.predict(observation)[0]
        if np.random.uniform() <= self.eps:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(q_estimates)
        return action

    def get_n_step_trans(self, n_step_buffer):
        gamma = self.config['gamma']
        discounted_return = 0
        cum_gamma = 1
        for trans in list(n_step_buffer)[:-1]:
            _, _, reward, _ = trans
            discounted_return += cum_gamma * reward
            cum_gamma *= gamma
        observation, action, _, _ = n_step_buffer[0]
        last_observation, _, _, done = n_step_buffer[-1]
        experience = (observation, action, discounted_return,
                      last_observation, done, cum_gamma)
        return experience

    def stop(self):
        self.continue_sampling = False

    def sample(self):
        print("Starting sampling in actor {}".format(self.actor_id))
        self.update_q_network()
        observation = self.env.reset()
        episode_reward = 0
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        while self.continue_sampling:
            action = self.get_action(observation)
            next_observation, reward, \
            done, info = self.env.step(action)
            n_step_buffer.append((observation, action,
                                  reward, done))
            if len(n_step_buffer) == self.multi_step_n + 1:
                self.local_buffer.append(
                    self.get_n_step_trans(n_step_buffer))
            self.cur_steps += 1
            episode_reward += reward
            episode_length += 1
            if done:
                if self.eval:
                    break
                next_observation = self.env.reset()
                if len(n_step_buffer) > 1:
                    self.local_buffer.append(
                        self.get_n_step_trans(n_step_buffer))
                self.cur_episodes += 1
                episode_reward = 0
                episode_length = 0
            observation = next_observation
            if self.cur_steps % self.send_experience_freq == 0 and not self.eval:
                self.send_experience_to_replay()
            if self.cur_steps % self.q_update_freq == 0 and not self.eval:
                self.update_q_network()
        return episode_reward

    def send_experience_to_replay(self):
        rf = self.replay_buffer.add.remote(self.local_buffer)
        ray.wait([rf])
        self.local_buffer = []


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Parameter Server
#   - receive the updated parameters (weights) from the learner and serve them to actors
#   - stores the actual Q network structure to be able to use TensorFlow's convenient save functionality
# -----------------------------------------------------------------------------------------------------------

@ray.remote
class ParameterServer:
    def __init__(self, config):
        self.weights = None
        self.eval_weights = None
        self.Q = get_Q_network(config)

    def update_weights(self, new_parameters):
        self.weights = new_parameters
        return True

    def get_weights(self):
        return self.weights

    def get_eval_weights(self):
        return self.eval_weights

    def set_eval_weights(self):
        self.eval_weights = self.weights
        return True

    def save_eval_weights(self,
                          filename=
                          'checkpoints/model_checkpoint'):
        self.Q.set_weights(self.eval_weights)
        self.Q.save_weights(filename)
        print("Saved.")


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Replay Buffer
#   - standard replay buffer (without prioritized sampling)
#   - receive experiences from actors and send sampled ones to the learner
#   - keep track of how many total experience tuples it has received that far in the training
# -----------------------------------------------------------------------------------------------------------

@ray.remote
class ReplayBuffer:
    def __init__(self, config):
        self.replay_buffer_size = config["buffer_size"]
        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.total_env_samples = 0

    def add(self, experience_list):
        experience_list = experience_list
        for e in experience_list:
            self.buffer.append(e)
            self.total_env_samples += 1
        return True

    def sample(self, n):
        if len(self.buffer) > n:
            sample_ix = np.random.randint(
                len(self.buffer), size=n)
            return [self.buffer[ix] for ix in sample_ix]

    def get_total_env_samples(self):
        return self.total_env_samples


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Model Geenration
# -----------------------------------------------------------------------------------------------------------

from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


# a masking input based on the selected action
def masked_loss(args):
    y_true, y_pred, mask = args
    masked_pred = K.sum(mask * y_pred, axis=1, keepdims=True)
    loss = K.square(y_true - masked_pred)
    return K.mean(loss, axis=-1)


# Note that the compiled Q-network model will neber be trained alone.
# the optimiser and loss function is just placeholders
# Q-network predicts Q-values for all possible actions.
def get_Q_network(config):
    obs_input = Input(shape=config["obs_shape"],
                      name='Q_input')

    x = Flatten()(obs_input)
    for i, n_units in enumerate(config["fcnet_hiddens"]):
        layer_name = 'Q_' + str(i + 1)
        x = Dense(n_units,
                  activation=config["fcnet_activation"],
                  name=layer_name)(x)
    q_estimate_output = Dense(config["n_actions"],
                              activation='linear',
                              name='Q_output')(x)
    # Q Model
    Q_model = Model(inputs=obs_input,
                    outputs=q_estimate_output)
    Q_model.summary()
    Q_model.compile(optimizer=Adam(), loss='mse')
    return Q_model


def get_trainable_model(config):
    Q_model = get_Q_network(config)
    obs_input = Q_model.get_layer("Q_input").output
    q_estimate_output = Q_model.get_layer("Q_output").output
    mask_input = Input(shape=(config["n_actions"],),
                       name='Q_mask')
    sampled_bellman_input = Input(shape=(1,),
                                  name='Q_sampled')

    # Trainable model
    loss_output = Lambda(masked_loss,
                         output_shape=(1,),
                         name='Q_masked_out')\
                        ([sampled_bellman_input,
                          q_estimate_output,
                          mask_input])
    trainable_model = Model(inputs=[obs_input,
                                    mask_input,
                                    sampled_bellman_input],
                            outputs=loss_output)
    trainable_model.summary()
    trainable_model.compile(optimizer=
                            Adam(lr=config["lr"],
                            clipvalue=config["grad_clip"]),
                            loss=[lambda y_true,
                                         y_pred: y_pred])
    return Q_model, trainable_model


# -----------------------------------------------------------------------------------------------------------
# Learner
# -----------------------------------------------------------------------------------------------------------

from tensorflow.keras.models import clone_model

@ray.remote
class Learner:
    def __init__(self, config, replay_buffer, parameter_server):
        self.config = config
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.Q, self.trainable = get_trainable_model(config)
        self.target_network = clone_model(self.Q)
        self.train_batch_size = config["train_batch_size"]
        self.total_collected_samples = 0
        self.samples_since_last_update = 0
        self.send_weights_to_parameter_server()
        self.stopped = False

    def send_weights_to_parameter_server(self):
        self.parameter_server.update_weights.remote(self.Q.get_weights())

    def start_learning(self):
        print("Learning starting...")
        self.send_weights()
        while not self.stopped:
            sid = self.replay_buffer.get_total_env_samples.remote()
            total_samples = ray.get(sid)
            if total_samples >= self.config["learning_starts"]:
                self.optimize()

    def optimize(self):
        samples = ray.get(self.replay_buffer
                          .sample.remote(self.train_batch_size))
        if samples:
            N = len(samples)
            self.total_collected_samples += N
            self.samples_since_last_update += N
            ndim_obs = 1
            for s in self.config["obs_shape"]:
                if s:
                    ndim_obs *= s
            n_actions = self.config["n_actions"]
            obs = np.array([sample[0] for sample \
                        in samples]).reshape((N, ndim_obs))
            actions = np.array([sample[1] for sample \
                        in samples]).reshape((N,))
            rewards = np.array([sample[2] for sample \
                        in samples]).reshape((N,))
            last_obs = np.array([sample[3] for sample \
                        in samples]).reshape((N, ndim_obs))
            done_flags = np.array([sample[4] for sample \
                        in samples]).reshape((N,))
            gammas = np.array([sample[5] for sample \
                        in samples]).reshape((N,))
            masks = np.zeros((N, n_actions))
            masks[np.arange(N), actions] = 1
            dummy_labels = np.zeros((N,))
            # double DQN
            maximizer_a = np.argmax(self.Q.predict(last_obs),
                                    axis=1)
            target_network_estimates = \
                self.target_network.predict(last_obs)
            q_value_estimates = \
                np.array([target_network_estimates[i,
                                      maximizer_a[i]]
                        for i in range(N)]).reshape((N,))
            sampled_bellman = rewards + gammas * \
                              q_value_estimates * \
                              (1 - done_flags)
            trainable_inputs = [obs, masks,
                                sampled_bellman]
            self.trainable.fit(trainable_inputs,
                               dummy_labels, verbose=0)
            self.send_weights()

            if self.samples_since_last_update > 500:
                self.target_network.set_weights(self.Q.get_weights())
                self.samples_since_last_update = 0
            return True
        else:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

    def send_weights(self):
        id = self.parameter_server.update_weights.remote(self.Q.get_weights())
        ray.get(id)

    def stop(self):
        self.stopped = True


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------------------

def get_env_parameters(config):
    env = gym.make(config["env"])
    config['obs_shape'] = env.observation_space.shape
    config['n_actions'] = env.action_space.n


# ----------
# max_samples = 500000
max_samples = 50000

config = {
    "env": "CartPole-v1",
    # ----------
    # "num_workers": 50,
    # "eval_num_workers": 10,
    # ----------
    "num_workers": 10,
    "eval_num_workers": 2,
    # ----------
    "n_step": 3,
    "max_eps": 0.5,
    "train_batch_size": 512,
    "gamma": 0.99,
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "tanh",
    "lr": 0.0001,
    # ----------
    # "buffer_size": 1000000,
    # "learning_starts": 5000,
    # "timesteps_per_iteration": 10000,
    # ----------
    "buffer_size": 100000,
    "learning_starts": 500,
    "timesteps_per_iteration": 1000,
    "grad_clip": 10
}


# -----------------------------------------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'

get_env_parameters(config)

print(config['obs_shape'])
print(config['n_actions'])


# ----------
log_dir = os.path.join(base_path, '04_output\\cartpole\\logs\\apex_dqn\\scalars\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
metrics_dir = os.path.join(log_dir, 'metrics')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

config['log_dir'] = log_dir

file_writer = tf.summary.create_file_writer(metrics_dir)
file_writer.set_as_default()


# ----------
ray.init()

parameter_server = ParameterServer.remote(config)

replay_buffer = ReplayBuffer.remote(config)

learner = Learner.remote(config,
                         replay_buffer,
                         parameter_server)

print(parameter_server)
print(replay_buffer)
print(learner)


# ----------
learner.start_learning.remote()


# ----------
# Create training actors
training_actor_ids = []

for i in range(config["num_workers"]):
    eps = config["max_eps"] * i / config["num_workers"]
    actor = Actor.remote("train-" + str(i),
                         replay_buffer,
                         parameter_server,
                         config,
                         eps)
    actor.sample.remote()
    training_actor_ids.append(actor)

print(training_actor_ids)


# ----------
# Create eval actors
eval_actor_ids = []

for i in range(config["eval_num_workers"]):
    eps = 0
    actor = Actor.remote("eval-" + str(i),
                         replay_buffer,
                         parameter_server,
                         config,
                         eps,
                         True)
    eval_actor_ids.append(actor)


Wtotal_samples = 0
best_eval_mean_reward = np.NINF
eval_mean_rewards = []
while total_samples < max_samples:
    tsid = replay_buffer.get_total_env_samples.remote()
    new_total_samples = ray.get(tsid)
    if (new_total_samples - total_samples
            >= config["timesteps_per_iteration"]):
        total_samples = new_total_samples
        print("Total samples:", total_samples)
        parameter_server.set_eval_weights.remote()
        eval_sampling_ids = []
        for eval_actor in eval_actor_ids:
            sid = eval_actor.sample.remote()
            eval_sampling_ids.append(sid)
        eval_rewards = ray.get(eval_sampling_ids)
        print("Evaluation rewards: {}".format(eval_rewards))
        eval_mean_reward = np.mean(eval_rewards)
        eval_mean_rewards.append(eval_mean_reward)
        print("Mean evaluation reward: {}".format(eval_mean_reward))
        tf.summary.scalar('Mean evaluation reward', data=eval_mean_reward, step=total_samples)
        if eval_mean_reward > best_eval_mean_reward:
            print("Model has improved! Saving the model!")
            best_eval_mean_reward = eval_mean_reward
            parameter_server.save_eval_weights.remote()

print("Finishing the training.")
for actor in training_actor_ids:
    actor.stop.remote()
learner.stop.remote()

