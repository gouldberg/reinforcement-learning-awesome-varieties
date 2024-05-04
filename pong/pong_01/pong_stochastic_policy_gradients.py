# -*- coding: utf-8 -*-

import os, sys
import numpy as np
# import cPickle as pickle
# import _pickle.pickle as pickle
import pickle
import gym


# ----------
# REFERENCE
# Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

# Karpathy blog:
# http://karpathy.github.io/2016/05/31/rl/


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------------------------------------

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    # ----------
    I = I[::2, ::2, 0]  # downsample by factor of 2
    # ----------
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # ----------
    # return I.astype(np.float).ravel()
    return I.astype(float).ravel()


def discount_rewards(r):
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


# take the state of the game and decide what we should do (move Up or Down)
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2 (move Up), and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# hyper parameters
# -----------------------------------------------------------------------------------------------------------

# 2-layer policy network with 200 hidden layer units
H = 200  # number of hidden layer neurons

batch_size = 10  # every how many episodes to do a param update?

# learning_rate = 1e-4
learning_rate = 1e-3

gamma = 0.99  # discount factor for reward

decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

resume = False  # resume from previous checkpoint?

render = False


# -----------------------------------------------------------------------------------------------------------
# model initialization
# -----------------------------------------------------------------------------------------------------------

# base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'
# model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients\\model.p')

base_path = '/home/kswada/kw/reinforcement_learning'
model_path = os.path.join(base_path, '04_output/pong/stochastic_policy_gradients/model.p')


# ----------
D = 80 * 80  # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open(model_path, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch

# rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


# -----------------------------------------------------------------------------------------------------------
# training:
# http://karpathy.github.io/2016/05/31/rl/
# I trained a 2-layer policy network with 200 hidden layer units using RMSProp on batches of 10 episodes
# (each episode is a few dozen games, because the games go up to score of 21 for either player).
# After training for 3 nights I ended up with a policy that is slightly better than the AI player.
# The total number of episodes was approximately 8,000
# so the algorithm played roughly 200,000 Pong games (quite a lot isn’t it!) and made a total of ~800 updates.
# -----------------------------------------------------------------------------------------------------------

env = gym.make("Pong-v0")
# env = gym.wrappers.Monitor(env, '.', force=True)

# ----------
# observation = env.reset()
observation = env.reset()[0]
# ----------
print(observation)


# ----------
prev_x = None  # used in computing the difference frame

xs, hs, dlogps, drs = [], [], [], []

running_reward = None

reward_sum = 0

episode_number = 0

while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    # Ideally you'd want to feed at least 2 frames to the policy network so that it can detect motion.
    # To make thins a bit simpler, feed difference frames to the network (i.e. subtraction of current and last frame)
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    # a + 1 reward if the ball went past the opponent, a - 1 reward if we missed the ball,
    # ----------
    # observation, reward, done, info = env.step(action)
    observation, reward, done, info, _ = env.step(action)
    # ----------
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            # for k, v in model.iteritems():
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('ep %d:  resetting env. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open(model_path, 'wb'))
        reward_sum = 0
        # ----------
        # observation = env.reset()  # reset env
        observation = env.reset()[0]  # reset env
        # ----------
        prev_x = None

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep %d: game finished, reward: %f' % (episode_number, reward))


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# load model
# -----------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/reinforcement_learning'

model_path = os.path.join(base_path, '04_output/pong/stochastic_policy_gradients/model.p')


# ----------
D = 80 * 80  # input dimensionality: 80x80 grid
model = pickle.load(open(model_path, 'rb'))


# -----------------------------------------------------------------------------------------------------------
# render on screen
# -----------------------------------------------------------------------------------------------------------

# env = gym.make('ALE/Pong-v5')
# env = gym.make("Pong-v0")

# render_mode requires 'human'
env = gym.make('Pong-v0', render_mode='human')


# ----------
# observation = env.reset()
observation = env.reset()[0]

frames = []
cumulated_reward = 0

prev_x = None # used in computing the difference frame

for t in range(1000):
    # ----------
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # ----------
    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!
    # ----------
    # env.step renders automatically ...
    # observation, reward, done, info = env.step(action)
    observation, reward, done, info, _ = env.step(action)
    # ----------
    cumulated_reward += reward
    # ----------
    if done:
        print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
        break

print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
env.close()



# -----------------------------------------------------------------------------------------------------------
# render on animation  -->  NOT WORKING ....
# -----------------------------------------------------------------------------------------------------------

from matplotlib import animation
import matplotlib.pyplot as plt

animation_path = os.path.join(base_path, '04_output/pong/stochastic_policy_gradients/anim.gif')


# ----------
# env = gym.make('ALE/Pong-v5')
# env = gym.make("Pong-v0")

# render_mode requires 'human'
env = gym.make('Pong-v0', render_mode='human')


# ----------
# observation = env.reset()
observation = env.reset()[0]

frames = []
cumulated_reward = 0

prev_x = None # used in computing the difference frame

for t in range(1000):
    frames.append(env.render())

    # ----------
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # ----------
    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!
    # ----------
    # observation, reward, done, info = env.step(action)
    observation, reward, done, info, _ = env.step(action)
    # ----------
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

