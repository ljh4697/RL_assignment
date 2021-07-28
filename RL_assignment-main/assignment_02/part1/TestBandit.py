import numpy as np
import MDP
import RL2


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
empiricalMeans = banditProblem.epsilonGreedyBandit(nIterations=400)
print(empiricalMeans)
print(' -------------------------------------------------------- ')

# Test Thompson sampling strategy
empiricalMeans = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=400)
print(empiricalMeans)
print(' -------------------------------------------------------- ')



# Test UCB strategy
empiricalMeans = banditProblem.UCBbandit(nIterations=400)
print(empiricalMeans)
print(' -------------------------------------------------------- ')



# chang