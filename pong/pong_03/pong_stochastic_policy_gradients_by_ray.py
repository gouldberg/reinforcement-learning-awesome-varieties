# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import time
import pickle

import gym
import ray

# ----------
# REFERENCE
# https://docs.ray.io/en/latest/ray-core/examples/plot_pong_example.html


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------------------------------------

def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def process_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # for t in reversed(xrange(0, r.size)):
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# -----------------------------------------------------------------------------------------------------------
# policy network (agent)
# -----------------------------------------------------------------------------------------------------------

class Model(object):
    """This class holds the neural network weights."""

    def __init__(self):
        self.weights = {}
        self.weights["W1"] = np.random.randn(H, D) / np.sqrt(D)
        self.weights["W2"] = np.random.randn(H) / np.sqrt(H)

    def policy_forward(self, x):
        h = np.dot(self.weights["W1"], x)
        h[h < 0] = 0  # ReLU nonlinearity.
        logp = np.dot(self.weights["W2"], h)
        # Softmax
        p = 1.0 / (1.0 + np.exp(-logp))
        # Return probability of taking action 2, and hidden state.
        return p, h

    def policy_backward(self, eph, epx, epdlogp):
        """Backward pass to calculate gradients.

        Arguments:
            eph: Array of intermediate hidden states.
            epx: Array of experiences (observations).
            epdlogp: Array of logps (output of last layer before softmax).

        """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.weights["W2"])
        # Backprop relu.
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        return {"W1": dW1, "W2": dW2}

    def update(self, grad_buffer, rmsprop_cache, lr, decay):
        """Applies the gradients to the model parameters with RMSProp."""
        for k, v in self.weights.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g ** 2
            self.weights[k] += lr * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)


def zero_grads(grad_buffer):
    """Reset the batch gradient buffer."""
    for k, v in grad_buffer.items():
        grad_buffer[k] = np.zeros_like(v)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# rollout
#  - plays an entire game of Pong (until either the computer or the RL agent loses).
# -----------------------------------------------------------------------------------------------------------

def rollout(model, env):
    """Evaluates  env and model until the env returns "Done".

    Returns:
        xs: A list of observations
        hs: A list of model hidden states per observation
        dlogps: A list of gradients
        drs: A list of rewards.

    """
    # Reset the game.
    observation = env.reset()
    # Note that prev_x is used in computing the difference frame.
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    done = False
    while not done:
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = model.policy_forward(x)
        # Sample an action.
        action = 2 if np.random.uniform() < aprob else 3

        # The observation.
        xs.append(x)
        # The hidden state.
        hs.append(h)
        y = 1 if action == 2 else 0  # A "fake label".
        # The gradient that encourages the action that was taken to be
        # taken (see http://cs231n.github.io/neural-networks-2/#losses if
        # confused).
        dlogps.append(y - aprob)

        observation, reward, done, info = env.step(action)

        # Record reward (has to be done after we call step() to get reward
        # for previous action).
        drs.append(reward)
    return xs, hs, dlogps, drs


# -----------------------------------------------------------------------------------------------------------
# parallelizing gradients
# -----------------------------------------------------------------------------------------------------------

# By adding the `@ray.remote` decorator,
# a regular Python function becomes a Ray remote function.
@ray.remote
class RolloutWorker(object):
    def __init__(self):
        # Tell numpy to only use one core. If we don't do this, each actor may
        # try to use all of the cores and the resulting contention may result
        # in no speedup over the serial version. Note that if numpy is using
        # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
        # probably need to do it from the command line (so it happens before
        # numpy is imported).
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make("Pong-v0")

    def compute_gradient(self, model):
        # Compute a simulation episode.
        xs, hs, dlogps, drs = rollout(model, self.env)
        reward_sum = sum(drs)
        # Vectorize the arrays.
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        # Compute the discounted reward backward through time.
        discounted_epr = process_rewards(epr)
        # Standardize the rewards to be unit normal (helps control the gradient
        # estimator variance).
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        # Modulate the gradient with advantage (the policy gradient magic
        # happens right here).
        epdlogp *= discounted_epr
        return model.policy_backward(eph, epx, epdlogp), reward_sum


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# hyper parameters
# -----------------------------------------------------------------------------------------------------------

# 2-layer policy network with 200 hidden layer units
H = 200  # number of hidden layer neurons

# learning_rate = 1e-4
learning_rate = 1e-3

gamma = 0.99  # discount factor for reward

decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2


# ----------
D = 80 * 80  # input dimensionality: 80x80 grid


# -----------------------------------------------------------------------------------------------------------
# training
#   - This example is easy to parallelize because the network can play ten games in parallel and
#     no information needs to be shared between the games.
#     In the loop, the network repeatedly plays games of Pong and records a gradient from each game.
#     Every ten games, the gradients are combined together and used to update the network.
# -----------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'
model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_by_ray\\model.p')


# ----------
# Calling ray.init() starts a local Ray instance on your laptop/machine
ray.init()
print(ray.is_initialized())


# ----------
# every how many episodes to do a param update ?
# batch_size = 10
batch_size = 4

# invoke remote function by remote().
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
actors = [RolloutWorker.remote() for _ in range(batch_size)]


# ----------
# "Xavier" initialization.
# Update buffers that add up gradients over a batch.
grad_buffer = {k: np.zeros_like(v) for k, v in model.weights.items()}

# Update the rmsprop memory.
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.weights.items()}


# ----------
model = Model()

running_reward = None

iterations = 200
hist = []

for i in range(1, 1 + iterations):
    model_id = ray.put(model)
    gradient_ids = []
    # ----------
    # Launch tasks to compute gradients from multiple rollouts in parallel.
    start_time = time.time()
    gradient_ids = [actor.compute_gradient.remote(model_id) for actor in actors]
    for batch in range(batch_size):
        # Return as soon as one of the tasks finished execution by ray.wait()
        [grad_id], gradient_ids = ray.wait(gradient_ids)
        grad, reward_sum = ray.get(grad_id)
        # Accumulate the gradient over batch.
        for k in model.weights:
            grad_buffer[k] += grad[k]
        running_reward = (
            reward_sum
            if running_reward is None
            else running_reward * 0.99 + reward_sum * 0.01
        )
    end_time = time.time()
    # ----------
    print(f'iteration {i} computed {batch_size} rollouts in {end_time - start_time: .2f} secs, running reward {running_reward: .2f}')
    hist.append((i, reward_sum, running_reward))
    if i % 100 == 0: pickle.dump(model, open(model_path, 'wb'))
    # ----------
    model.update(grad_buffer, rmsprop_cache, learning_rate, decay_rate)
    zero_grads(grad_buffer)


# ----------
ray.shutdown()


# ----------
print(hist)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# play
# -----------------------------------------------------------------------------------------------------------

def policy_forward(x):
    h = np.dot(model.weights['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model.weights['W2'], h)
    p = 1.0 / (1.0 + np.exp(-logp))
    return p, h  # return probability of taking action 2 (move Up), and hidden state


from matplotlib import animation
import matplotlib.pyplot as plt


model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_by_ray\\model.p')
animation_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_by_ray\\anim.gif')

model = pickle.load(open(model_path, 'rb'))


# ----------
# env = gym.make('ALE/Pong-v5')
env = gym.make("Pong-v0")


# ----------
observation = env.reset()

frames = []
cumulated_reward = 0

prev_x = None # used in computing the difference frame

for t in range(1000):
    frames.append(env.render(mode='rgb_array'))

    # ----------
    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # ----------
    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!
    observation, reward, done, info = env.step(action)
    cumulated_reward += reward
    # ----------
    if done:
        print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
        break

print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
env.close()


# ----------
plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 144)
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
plt.close(anim._fig)


# -----------
anim.save(animation_path, writer='PillowWriter')

# display(HTML(anim.to_jshtml()))
# display(anim)
