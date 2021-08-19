import sys
import gym
import pylab
import numpy as np
import tensorflow as tf

a = np.array([[2, 2], [3, 3]])
a = tf.constant(a)

print(a*a)
print(tf.reduce_sum(a*a))
print(tf.reduce_sum(a*a, axis=1))


