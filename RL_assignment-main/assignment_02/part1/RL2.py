import numpy as np
import MDP
import matplotlib.pyplot as plt
import math
import random

class DNN:
    pass

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

        # policy iteration
        



        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        return [V,policy]    

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
            
        return policyParams    

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
