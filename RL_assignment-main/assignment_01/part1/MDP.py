import numpy as np

class MDP(object):
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        self.V = np.zeros(self.nStates)
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        discount = self.discount
        T = self.T
        V = initialV
        R = self.R
        iterId = 0
        epsilon = 0
        
        while(iterId < nIterations):
            next_V = np.zeros(self.nStates)

            for s in range(self.nStates):
                max_q = []
                
                for a in range(self.nActions):
                    sigma_gamma_Pr_V = 0
                    sigma_gamma_Pr_V = discount * T[a][s] * V
                            
                    max_q.append(R[a][s] + np.sum(sigma_gamma_Pr_V))
                next_V[s] = max(max_q)
            

            epsilon = np.linalg.norm(next_V - V)
            V = next_V
            iterId += 1
            if epsilon <= tolerance:
                break







        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):

        discount = self.discount
        T = self.T
        R = self.R
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V
        

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded

        policy = [[0] for _ in range(self.nStates)]
        #policy = np.zeros([self.nStates, 1])
        for s in range(self.nStates):
            q_f = [0] * self.nActions
            q_f = np.array(q_f)
            for a in range(self.nActions):

                sigma_gamma_Pr_V = 0
                sigma_gamma_Pr_V = discount * T[a][s] * V
                        
                q_f[a] = R[a][s] + np.sum(sigma_gamma_Pr_V)
              
            max_idx_lsit = np.argwhere(q_f == np.amax(q_f))
            policy[s] = max_idx_lsit.flatten().tolist()
            
        


        return policy 

    def evaluatePolicy(self,policy):

        discount = self.discount
        T = self.T
        R = self.R

        V = np.zeros(self.nStates)
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded


        for s in range(self.nStates):
            pi_a_s = policy[s]
            
            if type(pi_a_s) ==list and len(pi_a_s) >= 2:
                for a in pi_a_s:
                    sigma_gamma_Pr_V = 0
                    for n_s in range(self.nStates):
                        if T[a][s][n_s] > 0:
                            sigma_gamma_Pr_V += discount* T[a][s][n_s] * self.V[n_s]
                    V[s] += (1/len(pi_a_s))*(R[a][s] + sigma_gamma_Pr_V)
                continue
            else:
            #for a, pa in enumerate(pi_a_s):
            #    if pa > 0:
                if type(pi_a_s) == list:
                    pi_a_s = pi_a_s[0]
                for n_s in range(self.nStates):
                    if T[pi_a_s][s][n_s] > 0:
                        V[s] +=  discount* T[pi_a_s][s][n_s] * self.V[n_s]
                V[s] += R[pi_a_s][s]
                        




        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).
        
        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = list(initialPolicy)
        V = np.zeros(self.nStates)
        self.V = V
        iterId = 0

        while (iterId < nIterations):
            iterId += 1

            #evaluatepolicy
            next_V = self.evaluatePolicy(policy)
            #impropolicy
            V = next_V
            self.V = V

            next_policy = self.extractPolicy(V)
            try:
                if next_policy == policy:
                    break
            except:
                pass
            policy = next_policy




        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        next_V = np.zeros(self.nStates)
        epsilon = 0
        T = self.T
        R = self.R
        discount = self.discount


        for s in range(self.nStates):
            pi_a_s = policy[s]
            
            if type(pi_a_s) ==list and len(pi_a_s) >= 2:
                for a in pi_a_s:
                    sigma_gamma_Pr_V = 0
                    sigma_gamma_Pr_V = discount * T[a][s] * initialV
                    next_V[s] += (1/len(pi_a_s))*(R[a][s] + np.sum(sigma_gamma_Pr_V))
                continue
            else:
            
                if type(pi_a_s) == list:
                    pi_a_s = pi_a_s[0]

                sigma_gamma_Pr_V = 0
                sigma_gamma_Pr_V = discount * T[pi_a_s][s] * initialV
                next_V[s] = R[pi_a_s][s] + np.sum(sigma_gamma_Pr_V)

        


        return next_V

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        k = nEvalIterations
        policy = [0] * self.nStates
        V = np.zeros(self.nStates)
        self.V = V
        iterId = 0
        epsilon = 0
        T = self.T
        R = self.R
        discount = self.discount
        nIterations = 10

        while(iterId < nIterations):
            # Eval : Repeat k times V^pi <-- R^pi + gamma T^pi V^pi
            for i in range(k):
                V = self.evaluatePolicyPartially(policy, V)
            # improve policy
            self.V = V
            policy = self.extractPolicy(V)
            #print(policy)
            # V <-- max_a R^a + gamma T^a V
            next_V = np.zeros(self.nStates)
            for s in range(self.nStates):
                max_q = []
                
                for a in range(self.nActions):
                    sigma_gamma_Pr_V = 0
                    sigma_gamma_Pr_V = discount * T[a][s] * V
                            
                    max_q.append(R[a][s] + np.sum(sigma_gamma_Pr_V))

                next_V[s] = max(max_q)

            epsilon = np.linalg.norm(next_V - V)
            if epsilon <= tolerance:
                break
            V = next_V

            iterId+=1
            print(epsilon)
            print(iterId)


        return [policy,V,iterId,epsilon]

