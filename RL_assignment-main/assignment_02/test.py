import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf

a = [[1, 2, 3], [4, 5, 6]]
a = np.array(a, float)
a= tf.nn.softmax(a, axis=1)
print(a)

