# -*- coding: utf-8 -*-
"""DQN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1adQLpYZtXdMwgp8cy2TZPyducSy0iuxO
"""

import os
import sys
import gym
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

# Selection action a and execute it
# Receive immediate reward r
# Observe new state s`
# Add (s, a, s`, r) to experience buffer
# Sample mini-batch of experiences from buffer
# For each experience (s, a, s`, r) in mini-batch
# update w
# update s = s`
# every c step updqte w`(target network) = w





#feat deep newral network

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
        self.memory = deque(maxlen=10000)

        # make model, target_model
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # initialize target_model
        self.update_target_model()

    def update_target_model(self):
        print(self.model.get_weights())
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        self.target_model.set_weights(self.model.get_weights())
        return

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])
        
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

            max_q = np.amax(target_predicts, axis = -1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


def main():


    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent1 = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_past100 = deque(maxlen=100)

    num_episode = 300
    avg_score = 0
    avg_scores = []
    dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

    for e in range(num_episode):
        done = False
        score = 0
        # initialize env
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        i = 0

        while not done:
            i += 1
            # get action from current state
            action = agent1.get_action(state)
            # get nextstate
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            #reward = 0.1 if not done or score == 500 else -1

            # append replay memory
            agent1.append_sample_replay(state, action, reward, next_state, done)

            if len(agent1.memory) >= agent1.train_start:
                agent1.train_model()
            
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent1.update_target_model()
                # 에피소드마다 학습 결과 출력
                score_past100.append(score)
                print("step: {:3d} | episode: {:3d} | score : {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(
                    i, e, score, len(agent1.memory), agent1.epsilon))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score)
                episodes.append(e)

                if e >= 100:
                    avg_score = sum(score_past100) /100
                    avg_scores.append(avg_score)
                    print("Avg 100 episodes score: {:3.2f}".format(avg_score))


                # 이동 평균이 180 이상일 때 종료
                if avg_score > 180:
                    agent1.model.save_weights(dirpath + "/save_model/model", save_format="tf")
                    plt.figure(figsize=(14,7))
                    
                    plt.subplot(211)
                    plt.plot(episodes, scores, 'b', alpha=0.7)
                    plt.xlabel("episode")
                    plt.ylabel("scores")
                    plt.xlim([0, e])

                    plt.subplot(212)
                    plt.plot(range(100, e+1), avg_scores, 'r', alpha=0.7)
                    plt.xlabel("episode")
                    plt.ylabel("average scores")
                    plt.xlim([0, e])

                    plt.savefig(dirpath  + "/save_graph/graph.png")
                    plt.show()
                    sys.exit()






def query_environment(name):

    env = gym.make(name)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

def info():
    query_environment("CartPole-v0")


if __name__ == '__main__':
    main()




