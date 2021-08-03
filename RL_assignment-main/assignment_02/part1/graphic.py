import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import numpy as np
import os
from matplotlib.cbook import get_sample_data
import math
from matplotlib.artist import Artist


class visualize_maze():
    ## this class is for  4x4 maze
    def __init__(self, V, R, iteration):
        
        self.V = V
        self.R = R
        self.iteration = iteration
        self.Q_visible = [[0] * 16 for i in range(4)] 

    def move_agemt(self, state):
        x = (state % 4) + 1
        y = 4 - (state // 4) 
        t = plt.text(x, y, '●', fontsize = '18', color='red', horizontalalignment='center', verticalalignment='center')
        plt.pause(0.4)
        Artist.set_visible(t, False)
        Artist.remove(t)

    
    def edit_q(self, state, action, q_val, ep,niteration):

        Artist.set_visible(self.Q_visible[action][state], False)
        Artist.remove(self.Q_visible[action][state])
        plt.pause(0.01)

        x = (state % 4) + 1
        y = 4 - (state // 4) 
        q_val = str(q_val)
        plt.title('nepisode = '+ f'{ep} '+',niteration =' + f'{niteration}')
        if action == 0:
            self.Q_visible[action][state] = plt.text(x, y+0.5, q_val, fontsize='10', color='black', horizontalalignment='center', verticalalignment='top')
        elif action == 1:
            self.Q_visible[action][state] = plt.text(x, y-0.4, q_val, fontsize='10', color='black', horizontalalignment='center', verticalalignment='bottom')
        elif action == 2:
            self.Q_visible[action][state] = plt.text(x-0.5, y, q_val, fontsize='10', color='black', horizontalalignment='left', verticalalignment='center')
        else:
            self.Q_visible[action][state] = plt.text(x+0.5, y, q_val, fontsize='10', color='black', horizontalalignment='right', verticalalignment='center')



    def draw_maze(self):
        action = ['↑', '↓', '←','→']
        
        plt.plot()
        
        Q = self.Q
        R = self.R
        

        plt.title('')
        plt.xlabel('black = q_val , blue = reward ,')


        for i in range(0,5):
            plt.axhline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
            plt.axvline(i+0.5,0, 1, color='black' , linestyle='-' ,linewidth='1')
        plt.axis([0.5,4+0.5,0.5,4+0.5])
        plt.xticks([])
        plt.yticks([])

        for y in range(4, 0, -1):
            for x in range(1, 5, 1):
                state = int(-(y-4)*4 + x - 1)

                #reward
                plt.text(x, y-0.5, round(R[0][state],2), fontsize=8, color='blue', horizontalalignment='center', verticalalignment='bottom')
                #policy
                for i, a in enumerate(Q[:, state]):
                    if i == 0:
                        self.Q_visible[i][state] = plt.text(x, y+0.5, round(Q[i][state], 2), fontsize='10', color='black', horizontalalignment='center', verticalalignment='top')
                    elif i == 1:
                        self.Q_visible[i][state] = plt.text(x, y-0.4, round(Q[i][state], 2), fontsize='10', color='black', horizontalalignment='center', verticalalignment='bottom')
                    elif i == 2:
                        self.Q_visible[i][state] = plt.text(x-0.5, y, round(Q[i][state], 2), fontsize='10', color='black', horizontalalignment='left', verticalalignment='center')
                    else:
                        self.Q_visible[i][state] = plt.text(x+0.5, y, round(Q[i][state], 2), fontsize='10', color='black', horizontalalignment='right', verticalalignment='center')
                
                
  

        #plt.show()
        
        return


if __name__ == "__main__":
    print('hello')
