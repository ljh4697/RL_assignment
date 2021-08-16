
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


class critic_DNN(tf.keras.Model):
    def __init__(self, action_size):
        super(critic_DNN, self).__init__()
        self.fc1 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc2 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc_out = Dense(1,
                            kernel_initializer=RandomUniform(-1, 1))
    
    def call(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        v = self.fc_out(x)
        return v

class actor_DNN(tf.keras.Model):
    def __init__(self, action_size):
        super(actor_DNN, self).__init__()
        self.fc1 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc2 = Dense(24, activation='relu', kernel_initializer=RandomUniform(-1, 1))
        self.fc_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1, 1))

    def call(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy


class A2Cagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #dqn hyper_parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.learning_rate_b = 0.001

        # replay memory size
        self.memory = deque(maxlen=3000)

        # make model, target_model
        self.critic = critic_DNN(action_size)
        self.target_model = critic_DNN(action_size)
        self.actor = actor_DNN(action_size)

        self.actor_optimizer = Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = Adam(learning_rate=self.learning_rate_b)


        # initialize target_model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.critic.get_weights())
        return

    def get_action(self, state):
        policy = self.actor(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

        
    def append_sample_replay(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self, state, action, reward, next_state, done):
        actor_params= self.actor.trainable_variables
        critic_params= self.critic.trainable_variables

        with tf.GradientTape() as actor_t:
            actor_t.watch(actor_params)

            policy = self.actor(state) # shape(1,2)
            value = self.critic(state) # shape(1,2)
            next_val = self.target_model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_val[0]

            #one_hot_next_action = tf.one_hot([next_action], self.action_size)
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action*policy, axis=1) # shape = (1)
            cross_entropy = -tf.math.log(action_prob + 1e-5) # shape = (1)
            advantage = tf.stop_gradient(target-value[0]) # shape = (1)
            actor_loss = tf.reduce_mean(cross_entropy*advantage) # shape = []

            ##actor update


        actor_grads = actor_t.gradient(actor_loss, actor_params)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

        with tf.GradientTape() as critic_t:
            critic_t.watch(critic_params)

            value = self.critic(state) # shape(1,2)
            next_val = self.target_model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_val[0]


            ## critic update
            critic_loss = 0.5*tf.square(tf.stop_gradient(target)-value[0])
            critic_loss = tf.reduce_mean(critic_loss) # shape = ()
            
        critic_grads = critic_t.gradient(critic_loss, critic_params)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_params))






def main():
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2Cagent(state_size, action_size)

    scores, episodes = [], []
    score_past100 = deque(maxlen=100)

    num_episode = 2000
    avg_score = 0
    dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    loss = 0

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
            # get nextstate
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            reward = 0.1 if not done or score == 200 else -1

            agent.train_model(state, action, reward, next_state, done)

            state = next_state
           
            if done:
                agent.update_target_model()
                score_past100.append(score)
                print("episode: {:3d} | score: {:3.2f} |loss: {:3.2f} ".format(e, score, loss))
                scores.append(score) ; episodes.append(e)
                if len(score_past100) ==100:
                    avg_score = np.mean(score_past100)
                    print(avg_score)

                

                    



            


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



            
            




if __name__ == "__main__":
    main()