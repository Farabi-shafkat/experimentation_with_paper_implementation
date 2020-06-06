
import matplotlib.pyplot as plt 
import numpy as np
import os
from opts import graph_save_directory
class graph(): #todo
    def __init__(self):
        self.training_loss=[]
        self.test_loss=[]
        self.save_name=os.path.join(graph_save_directory,"c3d modified with avg weights.png")
        self.first_time=False
    def update_graph(self,training_loss,test_loss):
        self.training_loss.append(training_loss)
        self.test_loss.append(test_loss)

    def draw_and_save(self):
        plt.plot( np.arange(0,len(self.training_loss)), self.training_loss,color='lightblue',linewidth=3,label='training_loss',marker='o', linestyle='dashed')
        plt.plot( np.arange(0,len(self.test_loss)), self.test_loss,color='darkgreen',linewidth=3,label='test_loss',marker='o', linestyle='dashed')
        if self.first_time==False:
            plt.xlabel("Epoch #")
            plt.ylabel("Total_loss")
            plt.legend()
            self.first_time = True
       # plt.show()
            plt.ylim(0,350)
            plt.xlim(0,50)
        plt.savefig(self.save_name)



if __name__=='__main__':
    test_graph = graph()
    test_graph.update_graph(9,100)
    test_graph.update_graph(93,101)
    test_graph.update_graph(942,102)
    test_graph.update_graph(93,103)
    test_graph.update_graph(9,104)
    test_graph.draw_and_save()