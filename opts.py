###opts
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts import *
from scipy.io import loadmat

main_datasets_dir="/content/content/dataset"
input_resize = 171,128
C,H,W = 3, 112, 112
randomseed=42
alpha=1
beta=1
gamma= 0.01
train_batch_size = 3
test_batch_size = 5
num_epochs = 101
sample_length = 103
device=torch.cuda
learning_rate=0.0001
#log_save_directory="logs"
log_save_directory="/content/drive/My Drive/what_and_how_well_you_learned_paper_imeplementation/logs"
#graph_save_directory="graphs"
graph_save_directory="/content/drive/My Drive/what_and_how_well_you_learned_paper_imeplementation/graphs"
model_saving_dir="/content/drive/My Drive/what_and_how_well_you_learned_paper_imeplementation/experimental_models_flat"
#model_saving_dir = "experimental_models"
#/content/drive/My Drive/what_and_how_well_you_learned_paper_imeplementation/

google_drive_dir="/content/drive/My Drive/what_and_how_well_you_learned_paper_imeplementation"
