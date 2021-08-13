import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#state = np.array([0])
#state = np.reshape( state, [1, 16])



q = np.array([44.35, 44.38])
p = np.array(q + 1e-1)
exp_q= np.exp(q/0.5)
policy = exp_q/np.sum(exp_q)
#a = np.log(a + 1e-5)
#print(type(a))
print(policy),
print(p)