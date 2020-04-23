from datetime import datetime
from opts import google_drive_dir,model_saving_dir,graph_save_directory
import os
def save_status(epoch):
    directory = os.path.join(google_drive_dir,'status')
    if os.path.exists(directory)==False:
        os.mkdir(directory)
    file = open(directory,"w")
    L="{},{},{}".format(epoch,model_saving_dir,graph_save_directory)
    file.writelines(L)  
    file.close()

def load_status():
    directory = os.path.join(google_drive_dir,'status')
    if os.path.exists(directory)==False:
        return 0
    file = open(directory,"r")
    s = file.readline([100])
    return int(s[0])
      
