# -*- coding: utf-8 -*-
"""CS885_spring20_a2_part2.ipynb

Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
This is a modified version of Pytorch DQN tutorial from 
http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
that does not use the rendered screen.
"""

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""Hyper parameters"""

EPISODES = 2000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.99  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 10  # NN hidden layer size
BATCH_SIZE = 500  # Q-learning batch size
TARGET_UPDATE = 100  # frequency of target update
BUFFER_SIZE = 10000  # capacity of the replay buffer 

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

"""Replay Buffer class"""

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

"""Simple QNetwork corresponding to a fully connected network with two hidden layers"""

class QNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

"""Initialize environment and variables"""

env = gym.make('CartPole-v0')

model = QNetwork()
target = QNetwork()
if use_cuda:
    model.cuda()
    target.cuda()
memory = ReplayMemory(BUFFER_SIZE)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []

"""Epsilon greedy function to select actions"""

def select_epsilon_greedy_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return target(state).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

"""Execute an episode.  At each step, the experience is stored in the replay buffer and the agent learns from a sampled batch of experience from the replay buffer."""

def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        action = select_epsilon_greedy_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(int(action[0, 0]))
        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward]),
                     FloatTensor([int(done)])))

        learn()

        state = next_state
        steps += 1

        if done:
            episode_durations.append(steps)
            if len(episode_durations) % 10 == 0:
                print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                plot_durations()
            break

"""Compute the max of the next Q values."""

def max_next_q_values(batch_next_state):
    # expected Q values are estimated from actions which gives maximum Q value
    return target(batch_next_state).detach().max(1)[0]

"""Train the agent with a batch of experiences sampled from the replay buffer."""

def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    batch_done = Variable(torch.cat(batch_done))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action).squeeze()

    expected_future_rewards = max_next_q_values(batch_next_state)
    expected_q_values = batch_reward + (GAMMA * expected_future_rewards) * (1-batch_done)

    # loss is measured from error between current and newly expected Q values
    loss = F.mse_loss(current_q_values, expected_q_values)

    # backpropagation of loss to QNetwork
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""Plot how many steps the pole was balanced before falling.  The maximum is # of steps is 200.  The environment automatically terminates an episode when the number of steps reaches 200.  This code produces a figure with two curves.  The first curve shows the number of steps before the pole fell in each episode.  The second curve shows the average of the number of steps before the pole fell in the past 100 episodes (starting after 100 episodes)."""

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

"""Execute a series of episodes, then close the environment."""

for e in range(EPISODES):
    global optimizer
    run_episode(e, env)
    if e % TARGET_UPDATE == 0:
        target.load_state_dict(model.state_dict())


print('Complete')
env.close()
plt.ioff()
plt.show()