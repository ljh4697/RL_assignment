import sys
import gym
import numpy as np
import tensorflow as tf

a = np.array([[2, 2], [3, 3]])
a = tf.constant(a)



true = [[0.3, 0.3, 0.4], [0.4, 0.4, 0.2]]
true = np.array(true)
pred = [[0.2, 0.2, 0.6], [0.3, 0.3, 0.4]]
pred = np.array(pred)
kl = tf.keras.losses.KLDivergence()

def kidiv_(true, pred):
    return tf.reduce_sum(true*tf.math.log(true/pred), axis=-1)


resault1 = kl(true, pred)
resault2 = kidiv_(true, pred)


print(f"resault1 : {resault1}")
print(f"resault2 : {resault2}")
print(f"{tf.reduce_sum(resault2)/2}")


