
import os
import sys
import gym
import random
from gym.core import ActionWrapper
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import math
import matplotlib.pyplot as plt


class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc2 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1, 1))
    
    def call(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #dqn hyper_parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 500
        self.train_start = 1000

        # replay memory size
        self.memory = deque(maxlen=3000)

        # make model, target_model
        self.critic = DQN(action_size)
        self.target_model = DQN(action_size)
        self.actor = DNN(action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # initialize target_model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return


    def get_action(self, state):
        policy = self.actor(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

        
    def append_sample_replay(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        #train parameter
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            #target 신경망은 업데이트를 안함
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis = 1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # with tf.GradientTape() as tape:
        #     q_val = self.model(state)
        #     one_hot_action = tf.one_hot([action], self.action_size)
        #     predict = tf.reduce_sum(one_hot_action * q_val, axis=1)

        #     target_predict = self.target_model(next_state)
        #     target_predict = tf.stop_gradient(target_predict)

        #     max_q = np.amax(target_predict, axis = 1)

        #     next_q_val = self.model(next_state)
        #     target = reward + (1-done)*self.discount_factor*next_q_val

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))




#REINFORCE code
class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #hyper parameter#
        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.model = DNN(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):

        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_model(self, state, action, reward, next_state, done):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy =  - policies * tf.math.log(policies)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)



def main():
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actor = DQNAgent(state_size, action_size)
    critic = Critic(state_size, action_size)

    scores, episodes = [], []
    socre_past100 = deque(maxlen=100)

    num_episode = 2000
    avg_score = 0
    dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

    for e in range(num_episode):
        done = False
        score = 0
        # initialize env
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        i = 0

        while not done:
            i+=1
            # get action from current state
            action = critic.get_action(state)
            # get nextstate
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward

            actor.append_sample_replay(state, action, reward, next_state, done)

            if len(actor.memory) >= actor.train_start:
                actor.train_model()
                critic.train_model(state, action, reward, next_state, done)


            

    


            
            




if __name__ == "__main__":
    main()