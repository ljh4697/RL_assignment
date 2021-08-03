import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
#state = np.array([0])
#state = np.reshape( state, [1, 16])

episodes = np.array(range(200))

scores_reinforce = episodes*2
scores_mb = episodes*3
scores_ql = episodes*4

plt.figure(figsize=(12, 12))
plt.title("test")
plt.subplot(221)
plt.plot(episodes, scores_reinforce, 'r', label='REINFORCE')

plt.subplot(222)
plt.plot(episodes, scores_mb, 'g' , label='modelbased VI')

plt.subplot(223)
plt.plot(episodes, scores_ql, 'b', label='Q-learning')

plt.subplot(224)
plt.plot(episodes, scores_reinforce, 'r', alpha=0.6, label='REINFORCE')
plt.plot(episodes, scores_mb, 'g' , alpha=0.6, label='modelbased VI')
plt.plot(episodes, scores_ql, 'b' , alpha=0.6, label='Q-learning')
#plt.title('RL2 scores graph')
#plt.axis([-1, 201, -200, 105])
plt.xlabel('episodes')
plt.ylabel('scores')
#plt.savefig(dirpath+'/save_graph/scores_graph.png')
plt.legend()
plt.show()