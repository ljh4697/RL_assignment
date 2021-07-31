import numpy as np
from tensorflow.python.keras.mixed_precision import policy
import MDP
import matplotlib.pyplot as plt
import math
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
from RL2 import RL2




class DNN(tf.keras.Model):
    def __init__(self, action_size):
        super(DNN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy

class trained_REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


        #hyper parameter#
        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.model = DNN(self.action_size)
      #  self.model.load_weights(self.dirpath + '/save_model/reinforce_model')

    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
      #  return np.random.choice(self.action_size, 1, p=policy)[0]
        return policy



if __name__ == "__main__":

    agent = trained_REINFORCEAgent(17,4)
    policies = []
    for e in range(17):
        done = False
        s = e
        state = np.zeros(17)
        state[s] = 1
        state = np.reshape(state , [1, 17])
        policy = agent.get_action(state)
        policies.append(policy)
    policies = np.array(policies)
    print(policies)
