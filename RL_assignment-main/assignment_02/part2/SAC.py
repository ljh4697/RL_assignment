
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

from tensorflow.keras.activations import softmax
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


class SACa:
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


class SAC_DNN(tf.keras.Model):
    def __init__(self, action_size):
        super(SAC_DNN, self).__init__()
        self.actor_fc = Dense(24, activation='relu')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))


        self.critic_fc1 = Dense(24, activation='relu')
        self.critic_fc2 = Dense(24, activation='relu')
        self.critic_out = Dense(action_size,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


# 카트폴 예제에서의 액터-크리틱(SAC_DNN) 에이전트
class SAC:
    def __init__(self, action_size):
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 500
        self.lamb = 0.2

        # replay memory 
        self.memory = deque(maxlen=3000)




        # 정책신경망과 가치신경망 생성
        self.model = SAC_DNN(self.action_size)
        self.target_model = SAC_DNN(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return
    
    def append_sample_replay(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model(state)
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)

            policies, q_value = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            q_predicts = tf.reduce_sum(one_hot_action*q_value, axis=1)

            _, next_q_value = self.target_model(next_states)
            __, t_q_value = self.target_model(states)


            next_actions = self.get_action(next_states)
            one_hot_next_action = tf.one_hot(next_actions, self.action_size)
            next_q_value = tf.reduce_sum(one_hot_next_action*next_q_value, axis=1)

            targets = rewards + (1 - dones) * self.discount_factor * (next_q_value + self.lamb*tf.reduce_sum(-policies*tf.math.log(policies + 1e-5), axis=1))

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(targets) - q_predicts)
            critic_loss = tf.reduce_mean(critic_loss)

            # 정책 신경망 오류 함수 구하기
            softmax_q_lambda = tf.nn.softmax(t_q_value/self.lamb, axis=1)
            softmax_q_lambda = tf.stop_gradient(softmax_q_lambda)


            kl = tf.keras.losses.KLDivergence()
            #actor_loss = tf.reduce_mean(kl(policies, softmax_q_lambda))
            actor_loss = kl(policies, softmax_q_lambda)



            # 하나의 오류 함수로 만들기
            loss = actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)






def main():
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = SAC(action_size)

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

            agent.append_sample_replay(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.batch_size:
                agent.train_model()

            state = next_state
           
            if done:
                if len(agent.memory) >= agent.batch_size and e % 2 == 0:
                    agent.update_target_model()
                score_past100.append(score)
                print("episode: {:3d} | score: {:3.2f} | memory length: {:4d}  ".format(e, score, len(agent.memory)))
                scores.append(score) ; episodes.append(e)
                if len(score_past100) ==100:
                    avg_score = np.mean(score_past100)
                    print(avg_score)

                

                    

            
            




if __name__ == "__main__":
    main()