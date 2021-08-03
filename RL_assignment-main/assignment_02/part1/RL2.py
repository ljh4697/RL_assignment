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


# DNN input 과 output 은 numpy 한가지 차원에 떠 싸여있다
# ex) input = [input] / output = [output]
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

class REINFORCEAgent:
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

    def train_model(self):
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



class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]


    def PolicyEvaluation(self):
        pass


    def PolicyImprovement(self):
        pass

    def get_action(self, state, V, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(range(self.action))
        else:
            

            V
            max_q_list = np.argwhere(q_list == np.amax(q_list))
            max_q = max_q_list.flatten().tolist()
            action = np.random.choice(max_q)
        return action


    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # with policy iteration

        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        n_s_a = np.zeros([self.mdp.nActions, self.mdp.nStates])
        n_s_a_sp = np.zeros([self.mdp.nActions, self.mdp.nStates, self.mdp.nStates])

        T = defaultT
        R = initialR
        
        scores, episodes = [], []

        dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


        for e in range(nEpisodes):
            n = 0
            done = False
            state = s0
            score = 0
            states = []


            while not done:

                
                #epsilon greedy
                if np.random.rand() < epsilon:
                    action = np.random.choice(range(self.mdp.nActions))
                else:
                    q_s_a = np.zeros(self.mdp.nActions)
                    for a in range(self.mdp.nActions):
                        q_s_a[a] = R[a, state] + self.mdp.discount*np.sum(T[a, state, :] * V[:])

                    policy[state] = np.argmax(q_s_a)
                    action = policy[state]  

                [reward, next_state] = self.sampleRewardAndNextState(state, action)

                n_s_a[action, state] += 1
                n_s_a_sp[action, state, next_state] += 1
                # update transision model
                T[action, state, :] = n_s_a_sp[action, state,:]/n_s_a[action, state]
                
                # update reward model
                R[action, state] += (1/n_s_a[action, state])*(reward - R[action, state])

                q_s_a = np.zeros(self.mdp.nActions)
                for a in range(self.mdp.nActions):
                    q_s_a[a] = R[a, state] + self.mdp.discount*np.sum(T[a, state, :] * V[:])
                
                V[state] = np.max(q_s_a)

                state = next_state
                score += reward
                n += 1
                if n == nSteps or state == 16:
                    done = True

                if done:
                    scores.append(score)
                    episodes.append(e)

                    print("step: {:3d} | episode: {:3d} | score: {:.3f} ".format(n, e, score))
                    #print(T)
                    #print(R)
                    print(V)
                    print(states)
                    print()
                    print('*******************************************')
                    print()
            #plt.plot(episodes, scores, 'r')
            #plt.xlabel('episodes')
            #plt.ylabel('scores')
            #plt.savefig(dirpath+'/save_graph/modelbasedRL.png')




        # temporary values to ensure that the code compiles until this
        # function is coded


        return [V,policy,scores]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        #reward = np.zeros(self.mdp.nActions)
        action_cnt  = np.zeros(self.mdp.nActions, int)
        epsilon = 0.5
        total_reward = 0
        for i in range(nIterations):
            p = np.random.rand(1)
            if p < epsilon:
                a = np.random.randint(self.mdp.nActions)
            else:
                a = np.argmax(empiricalMeans)
            
            action_cnt[a] += 1
            reward = self.sampleReward(self.mdp.R[a,0])
            total_reward += reward
            empiricalMeans[a] = empiricalMeans[a] + 1/action_cnt[a]*(reward - empiricalMeans[a])
        


        print("\n★  epsilonGreedyBandit results ★\n")
        print(action_cnt)
        print('epsilone greedy total_reward : ', total_reward)

          
          

        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        selected_machine_cnt = np.zeros(self.mdp.nActions, int)
        total_reward = 0

        for i in range(nIterations):

            beta_max = 0
            for a in range(self.mdp.nActions):
                beta_val = random.betavariate(prior[a,0], prior[a,1])
                if beta_val > beta_max:
                    beta_max = beta_val
                    action = a
            
            selected_machine_cnt[action] += 1
            reward = self.sampleReward(self.mdp.R[action,0])
            if reward == 1:
                prior[action, 0] += 1
            else:
                prior[action, 1] += 1
            total_reward += reward
               
        for e in range(self.mdp.nActions):
            empiricalMeans[e] = (prior[e,0]-1)/(prior[e,0] + prior[e,1] - 2)




        print("\n★  thompsonSamplingBandit results ★\n")
        print(selected_machine_cnt)
        print("thompsonSampling total reward : " , total_reward)
        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        actions_cnt = np.zeros(self.mdp.nActions, int)
        total_reward = 0
        for i in range(nIterations):
            
            bound = np.zeros(self.mdp.nActions)
            for k in range(self.mdp.nActions):
                if actions_cnt[k] == 0:
                    x = np.inf
                else:
                    x = 2*math.log(i)/actions_cnt[k]
                bound[k] = math.sqrt(x)

            a = np.argmax(empiricalMeans + bound)

            reward = self.sampleReward(self.mdp.R[a,0])
            actions_cnt[a] += 1
            total_reward += reward
            empiricalMeans[a] = empiricalMeans[a] + 1/actions_cnt[a]*(reward - empiricalMeans[a])
        

        print("\n★  UCBbandit results ★\n")
        print(actions_cnt)
        print('UCB total_reward =' ,total_reward)
        

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
        agent = REINFORCEAgent(self.mdp.nStates, self.mdp.nActions)

        scores, episodes = [], []
        dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

        for e in range(nEpisodes):
            done = False
            score = 0
            state = np.zeros(self.mdp.nStates)
            state[s0] = 1
            state = np.reshape(state, [1,self.mdp.nStates])
            
            
            n = 0
            while not done:
                action = agent.get_action(state)

                [reward, next_state] = self.sampleRewardAndNextState(np.where(state[0,:] == 1)[0][0], action)
                agent.append_sample(state, action, reward)
                score += reward

                state = np.zeros(self.mdp.nStates)
                state[next_state] = 1
                state = np.reshape(state, [1,self.mdp.nStates])


                n+=1
                if n == nSteps or state[0, 0] == 16:
                    done = True
                if done:
                    entropy = agent.train_model()
                    scores.append(score)
                    episodes.append(e)

                    print("episode: {:3d} | score: {:.3f} | entropy: {:.3f}".format(
        e, score, entropy))
                    #plt.plot(episodes, scores, 'b')
                    #plt.xlabel('episodes')
                    #plt.ylabel('scores')
                    #plt.savefig(dirpath+'/save_graph/reinforce_graph.png')

            if e % 100 == 0:
                agent.model.save_weights(dirpath + '/save_model/reinforce_model', save_format='tf')


            
        return scores   

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        action = 0
        
        return action

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''
        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ
        n_s_a = np.zeros([self.mdp.nActions, self.mdp.nStates], int)
        policy = np.zeros(self.mdp.nStates)
        discount = self.mdp.discount

        episodes, scores= [], []

        dirpath = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

        

        for e in range(nEpisodes):
            n = 0
            state = s0
            done = False
            score = 0

            while not done:


                #epsilon greedy
                if np.random.rand() < epsilon:
                    action = np.random.choice(range(self.mdp.nActions))
                else:
                    q_list = Q[:, state]
                    max_q_list = np.argwhere(q_list == np.amax(q_list))
                    max_q = max_q_list.flatten().tolist()
                    action = np.random.choice(max_q)

                [reward, next_state] = self.sampleRewardAndNextState(state,action)
 
                n_s_a[action][state] += 1
                alpha = 1/n_s_a[action][state]

                q1 = Q[action][state]
                q2 = reward + discount * np.max(Q[:, next_state])

                Q[action][state] += alpha *(q2 - q1)

                score += reward
                state = next_state

                n += 1


                if state == 16 or n == nSteps:
                    done = True
                if done:
                    scores.append(score)
                    episodes.append(e)

                    print("step: {:3d} | episode: {:3d} | score: {:.3f} ".format(n, e, score))

                    for s in range(self.mdp.nStates):
                        policy[s] = np.argmax(Q[:, s])


        #plt.plot(episodes, scores, 'b' ,alpha=0.6)
        #plt.xlabel('episodes')
        #plt.ylabel('scores')
        #plt.savefig(dirpath+'/save_graph/qlearning_graph.png')







        return [Q, policy, scores]
