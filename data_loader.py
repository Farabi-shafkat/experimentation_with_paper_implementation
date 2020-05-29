
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
from random import randint
import math 
#torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
#torch.backends.cudnn.deterministic=True

def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

class VideoDataset(Dataset):

    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            trains_list = np.load('/content/train_split_0.pkl',allow_pickle=True)
            dic=np.load('/content/final_annotations_dict.pkl',allow_pickle=True)
            captions=np.load('/content/final_captions_dict.pkl',allow_pickle=True)

            self.annotations = []
            for key in trains_list:
                self.annotations.append((key,dic[key],captions[key]))
            
        else:
            #self.annotations = loadmat('input/consolidated_test_list.mat').get('consolidated_test_list')
            test_list = np.load('/content/test_split_0.pkl',allow_pickle=True)
            dic=np.load('/content/final_annotations_dict.pkl',allow_pickle=True)
            captions=np.load('/content/final_captions_dict.pkl',allow_pickle=True)

            self.annotations = []
            for key in test_list:
                self.annotations.append((key,dic[key],captions[key]))



    def __getitem__(self, ix):
       
        sample = self.annotations[ix][0]
        start_frame=self.annotations[ix][1]['start_frame']
        end_frame=self.annotations[ix][1]['end_frame']
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
        
##########################

        image_list=[]
        index_list=[]
        hori_flip = 0
        if self.mode=='train':
           # index_list=np.random.choice(np.arange(start_frame,end_frame+1),size=96)

            end_frame = end_frame + randint(-3,3)
            start_frame = end_frame - sample_length
            num_frames = end_frame-start_frame+1
            index_list = [ x for x in range(start_frame,end_frame)]
            hori_flip = randint(0,1)


        else:
            start_frame = end_frame - sample_length
            num_frames = end_frame-start_frame+1
            index_list = [ x for x in range(start_frame,end_frame)]

        for cur in index_list:
            image_list.append(os.path.join("/content/content",'video{:d}_frames'.format(sample[0]), '{:d}.jpg'.format(cur)))
        
        
        image_list=sorted(image_list)
      
        #final_image_list=[]


        #for i in range(sample_length):
          #  index = int(round(i/96*(len(image_list))))
          
         #   final_image_list.append(image_list[index])
        
        #image_list=final_image_list
        
      
       # end_frame = min(end_frame,start_frame+sample_length-1)
    
    ################################  
        images = torch.zeros(sample_length, C, H, W)
        #hori_flip = 0

        for i in np.arange(0, sample_length):
         
            if i>=len(image_list):
                break
            if self.mode == 'train':
                hori_flip += random.randint(0,1)
                images[i] = load_image_train(image_list[i], hori_flip, transform)
            else:
                images[i] = load_image(image_list[i], transform)

        label_final_score = self.annotations[ix][1]['final_score'] 
       

        data = {}
        data['video'] = images
        data['primary_view']=self.annotations[ix][1]['primary_view']
        data['label_final_score'] = label_final_score
        data['difficulty']=self.annotations[ix][1]['difficulty']
        data['action']={
            'position':self.annotations[ix][1]['position'],
            'armstand':self.annotations[ix][1]['armstand'],
            'rotation_type':self.annotations[ix][1]['rotation_type'],
            'ss_no':self.annotations[ix][1]['ss_no'],
            'tw_no':self.annotations[ix][1]['tw_no']
        }
        data['captions']=self.annotations[ix][2]


        return data


    def __len__(self):
        print('No. of samples: ', len(self.annotations))
        return len(self.annotations)


if __name__=="__main__":

    VD = VideoDataset("train")
    for i in range(0,1000):
         VD.__getitem__(i)
         VD = VideoDataset("train")
    VD = VideoDataset("test")

    for i in range(0,300):
         VD.__getitem__(i)

