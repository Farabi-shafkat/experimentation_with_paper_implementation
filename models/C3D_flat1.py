# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
#
# Code for C3D-LSTM used in:
# [1] @inproceedings{parmar2017learning,
#   title={Learning to score olympic events},
#   author={Parmar, Paritosh and Morris, Brendan Tran},
#   booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
#   pages={76--84},
#   year={2017},
#   organization={IEEE}}
#
# [2] @article{parmar2018action,
#   title={Action Quality Assessment Across Multiple Actions},
#   author={Parmar, Paritosh and Morris, Brendan Tran},
#   journal={arXiv preprint arXiv:1812.06367},
#   year={2018}}

import torch
import torch.nn as nn
#from opts import *
import torch.nn.functional as F
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.M = 512
        self.conv5aSP = nn.Conv3d(512, self.M, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv5aTM = nn.Conv3d(self.M, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv5bSP = nn.Conv3d(512, self.M, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv5bTM = nn.Conv3d(self.M, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        

       # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, 487)

       # self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("before conv1 {}".format(x.shape))
        h = self.relu(self.conv1(x))
        print("after conv1 {}".format(h.shape))
        h = self.pool1(h)
        print("after pool1 {}".format(h.shape))

        h = self.relu(self.conv2(h))
        print("after conv2 {}".format(h.shape))
        h = self.pool2(h)
        print("after pool2 {}".format(h.shape))


        h = self.relu(self.conv3a(h))
        print("after conv3a {}".format(h.shape))
        h = self.relu(self.conv3b(h))
        print("after conv3b {}".format(h.shape))
        h = self.pool3(h)
        print("after pool3 {}".format(h.shape))

        h = self.relu(self.conv4a(h))
        print("after conv4a {}".format(h.shape))
        h = self.relu(self.conv4b(h))
        print("after conv4b {}".format(h.shape)) 
        h = self.pool4(h)
        print("after pool4 {}".format(h.shape))



        h = self.relu((self.conv5aSP(h)))
        h = self.relu((self.conv5aTM(h)))
        print("after conv5a {}".format(h.shape))
        h = self.relu((self.conv5bSP(h)))
        h = self.relu((self.conv5bTM(h)))
        print("after conv5b {}".format(h.shape))
        h = self.pool5(h)
        print("after pool5 {}".format(h.shape))
        #torch.flatten(h)

       

        
       
        h = h.view(-1, 8192)

        

       # print('Shape of pool5: ', h1.shape)
       # print('Shape of pool51: ', h.shape)
       # if torch.all(h1.eq(h)):
       #     print("it makes no sense")
       # else:
       #     print("diiferent")
       # h = self.relu(self.fc6(h))
        # h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)
      #  h= F.relu(h)
        return h


if __name__ == "__main__":
    c3d = C3D()
    dummy_clip =  torch.zeros(1,3, 16, 112, 112)
    print(dummy_clip.shape)
    h = c3d(dummy_clip)

    model_CNN_pretrained_dict = torch.load('study materials/paper/action quality assesment cvpr 2019/implementation/models/c3d.pickle')
    model_CNN_dict = c3d.state_dict()

    print("{} <---shape of conv 5a in c3d pickele".format(model_CNN_pretrained_dict['conv5a.weight'].shape))
    model_CNN_dict['conv5aSP.weight'] =torch.mean( model_CNN_pretrained_dict['conv5a.weight'],dim=2).unsqueeze(2)
    model_CNN_dict['conv5aSP.bias'] = model_CNN_pretrained_dict['conv5a.bias']

    model_CNN_dict['conv5aTM.weight'] =torch.mean( model_CNN_pretrained_dict['conv5a.weight'],dim=(3,4)).unsqueeze(3).unsqueeze(4)
    model_CNN_dict['conv5aTM.bias'] = model_CNN_pretrained_dict['conv5a.bias']
   
    model_CNN_dict['conv5bSP.weight'] = torch.mean(model_CNN_pretrained_dict['conv5b.weight'],dim=2).unsqueeze(2)
    model_CNN_dict['conv5bSP.bias'] = model_CNN_pretrained_dict['conv5b.bias']

    model_CNN_dict['conv5bTM.weight'] =torch.mean( model_CNN_pretrained_dict['conv5b.weight'],dim=(3,4)).unsqueeze(3).unsqueeze(4)
    model_CNN_dict['conv5bTM.bias'] = model_CNN_pretrained_dict['conv5b.bias']

    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    c3d.load_state_dict(model_CNN_dict)

    size =0
    for param in c3d.parameters():
        add =1 
        for k in param.shape:
            add*=k
        size += add
    print(size)




















"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""