import matplotlib.pyplot as plt
import numpy as np


class visualize_maze():
    def __init__(self, size, V, R):
        pass


maze_w = 4
maze_h = 4
V = list(i for i in range(1, 17))
R = -1 * np.ones([4,17]);

R[:,11] = 100;  # goal state
R[:,9] = -70;   # bad state
R[:,16] = 0;    # end state

plt.plot()
plt.title('value iteration')
plt.xlabel('red = V , blue=reward')


for i in range(0,5):
    plt.axhline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
    plt.axvline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
plt.axis([0.5,maze_w+0.5,0.5,maze_h+0.5])
plt.xticks([])
plt.yticks([])



for y in range(4, 0, -1):
    for x in range(4):
        plt.text(x+1, y, V[-(y-4)*4 + x], fontsize=12, color='red', horizontalalignment='center', verticalalignment='center')
        plt.text(x+1, y-0.5, R[0][-(y-4)*4 + x], fontsize=8, color='blue', horizontalalignment='center', verticalalignment='bottom')
        

plt.show()