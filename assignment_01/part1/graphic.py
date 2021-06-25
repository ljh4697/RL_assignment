import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import numpy as np
import os
from matplotlib.cbook import get_sample_data
import math


class visualize_maze():
    ## this class is for  4x4 maze
    def __init__(self, V, R, policy):
        
        self.V = V
        self.R = R
        self.policy = policy
        
        


    def draw_maze(self):
        action = ['↑', '↓', '←','→']
        
        plt.plot()
        
        V = self.V
        R = self.R
        policy = self.policy

        plt.title('')
        plt.xlabel('red = V , blue=reward')


        for i in range(0,5):
            plt.axhline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
            plt.axvline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
        plt.axis([0.5,4+0.5,0.5,4+0.5])
        plt.xticks([])
        plt.yticks([])

        for y in range(4, 0, -1):
            for x in range(1, 5, 1):
                state = -(y-4)*4 + x - 1

                #value
                plt.text(x, y, round(V[state],2), fontsize=12, color='red', horizontalalignment='center', verticalalignment='bottom')
                #reward
                plt.text(x, y-0.1, round(R[0][state],2), fontsize=8, color='blue', horizontalalignment='center', verticalalignment='top')
                #policy
                for a in policy[state]:
                    if a == 0:
                        plt.text(x, y+0.5, action[a], fontsize='13', color='black', horizontalalignment='center', verticalalignment='top')
                    elif a == 1:
                        plt.text(x, y-0.5, action[a], fontsize='13', color='black', horizontalalignment='center', verticalalignment='bottom')
                    elif a == 2:
                        plt.text(x-0.5, y, action[a], fontsize='13', color='black', horizontalalignment='left', verticalalignment='center')
                    else:
                        plt.text(x+0.5, y, action[a], fontsize='13', color='black', horizontalalignment='right', verticalalignment='center')
  

        plt.show()


        return


#maze_w = 4
#maze_h = 4
#V = list(i for i in range(1, 17))
#R = -1 * np.ones([4,17]);
#action = ['→', '←', '↓', '↑']
#
#
#R[:,11] = 100;  # goal state
#R[:,9] = -70;   # bad state
#R[:,16] = 0;    # end state
#
#plt.plot()
#
#
#plt.title('value iteration')
#plt.xlabel('red = V , blue=reward')
#
#
#for i in range(0,5):
#    plt.axhline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
#    plt.axvline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
#plt.axis([0.5,maze_w+0.5,0.5,maze_h+0.5])
#plt.xticks([])
#plt.yticks([])
#
#for y in range(4, 0, -1):
#    for x in range(4):
#        #value
#        plt.text(x+1, y, V[-(y-4)*4 + x], fontsize=12, color='red', horizontalalignment='center', verticalalignment='center')
#        #reward
#        plt.text(x+1, y-0.5, R[0][-(y-4)*4 + x], fontsize=8, color='blue', horizontalalignment='center', verticalalignment='bottom')
#        #policy
#        n_v = [0] * 4
#        try:
#            n_v[0] = max(V[-(y-4)*4 + x + 1],0)
#            if -(y-4)*4 + x - 1 >= 0:
#                n_v[1] =max(V[-(y-4)*4 + x - 1],0)
#            n_v[2] =max(V[-(y-4)*4 + x + 4],0)
#            if -(y-4)*4 + x - 4 >= 0:
#                n_v[3] = max(V[-(y-4)*4 + x - 4],0)
#        except:
#            pass
#
#        plt.text(x+1, y+0.5, action[n_v.index(max(n_v))], fontsize='15', color='black', horizontalalignment='center', verticalalignment='top')
#
#plt.show()