import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
import tensorflow as tf
#state = np.array([0])
#state = np.reshape( state, [1, 16])
policies = [[0.1, 0.3, 0.3, 0.3], [0.2, 0.4, 0.2, 0.2]]
actions = [[1, 0, 0, 0], [0, 1, 0, 0]]

policies = np.array(policies)
actions = np.array(actions)


action_prob = tf.reduce_sum(actions * policies, axis=1)

print(action_prob)

## entropy?