import os
import numpy as np
import random
from opts import *
from datetime import datetime



def save_logs(epoch,iteration,loss_statistics):
    time = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    directory = os.path.join(log_save_directory,"logs.txt")

    str1 = "{}      epoch:{}.....iteration: {}.............timesatmp:{}\n".format(loss_statistics["mode"],epoch,iteration,time)
    L=[str1]
    if loss_statistics["mode"]=="train":
        str2 = "               action_quality_loss= {} \n".format(loss_statistics["action_quality"])
        str3 = "               action_recogntion_loss= {} \n".format(loss_statistics["action_recognition_loss"])
        str4 = "               total_training_loss= {} \n".format(loss_statistics["loss"])
        str5 = "................XXXXXXx................\n\n\n\n\n"
        L.append(str2)
        L.append(str3)
        L.append(str4)
        L.append(str5)

    if loss_statistics["mode"]=="test":
        str2 = "               action_quality_loss= {} \n".format(loss_statistics["action_quality"])
        str3 = "               action_recogntion_loss= {} \n".format(loss_statistics["action_recognition_loss"])
        str4 = "               total_test_loss= {} \n".format(loss_statistics["loss"])
        str5 = "\n               RHO  = {} \n".format(loss_statistics["rho"])
        str6 = "................XXXXXXx................\n\n\n\n\n"
        L.append(str2)
        L.append(str3)
        L.append(str4)
        L.append(str5)
        L.append(str6)


    file = open(directory,"a")
    file.writelines(L)  
    file.close()
    return


if __name__=="__main__":
  # set_start_time1(str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    save_logs(1,2,3)
