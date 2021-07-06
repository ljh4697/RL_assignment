import numpy as np
import MDP
from graphic import visualize_maze
import matplotlib.pyplot as plt
import math

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.action = mdp.nActions
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
        
        discount = self.mdp.discount



        

        for i in range(nEpisodes):
            niteration = 0

            state = s0
            maze_ = visualize_maze(Q, self.mdp.R, niteration)
            maze_.draw_maze()
            while(niteration < nSteps):

                maze_.move_agemt(state)
                plt.pause(0.3)

                action = self.get_action(state, Q, epsilon)
                [reward, next_state] = self.sampleRewardAndNextState(state,action)
 
                n_s_a[action][state] += 1
                alpha = 1/n_s_a[action][state]

                q1 = Q[action][state]
                q2 = reward + discount * max(Q[:, next_state])

                Q[action][state] += alpha *(q2 - q1)
                maze_.edit_q(state,action, round(Q[action][state],2), i+1,niteration+1)


                state = next_state

                if state == 16:
                    break


                niteration += 1
            plt.pause(1)
            plt.close()

        policy = self.extracpolicy(Q)






        return [Q,policy]



        #epsilon-greedy exploration method

    def get_action(self, state, Q, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(range(self.action))
        else:
            q_list = Q[:, state]
            max_q_list = np.argwhere(q_list == np.amax(q_list))
            max_q = max_q_list.flatten().tolist()
            action = np.random.choice(max_q)
        return action

    def extracpolicy(self, Q):

        policy = [[0] for _ in range(self.mdp.nStates)] 

        for s in range(self.mdp.nStates):
            q_f = Q[:, s]
            max_q_idx = np.argwhere(q_f == np.amax(q_f))
            policy[s] = max_q_idx.flatten().tolist()

        return policy





