import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

#state = np.array([0])
#state = np.reshape( state, [1, 16])
state = np.zeros(16)
state[4] = 1
state = np.reshape(state, [1,16])

print(state)
print(state[0])