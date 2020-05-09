############avg_fc


import torch
import torch.nn as nn
import torch.nn.functional as F


class AVG_FC(nn.Module):
    def __init__(self):
        super(AVG_FC, self).__init__()
        self.fc=nn.Linear(8192,4096)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):   
       
        x = F.relu(self.dropout(self.fc(F.relu(x))))
        return x


class linearRegression(torch.nn.Module):
    def __init__(self):
        
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(4096, 1)
       # self.dropout = nn.Dropout(p=0.2)
        #self.linear2 = torch.nn.Linear(256, 1)
        #self.linear2 = torch.nn.Linear(1024, 1)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout1 = nn.Dropout(p=0.2)
    def forward(self, x):
        out = F.relu((self.linear1(x)))
        #out = F.relu(self.linear2(out))
        #out = F.relu(self.linear3(out))
        #out = self.relu(out)
        return out







class classficatiion(torch.nn.Module):
    def __init__(self):
        super(classficatiion, self).__init__()
        self.posFC=nn.Linear(4096,3)
        self.relu1 = nn.ReLU()
        self.pos=nn.Softmax(dim=0)

        self.TWFC=nn.Linear(4096,8)
        self.TW=nn.Softmax(dim=0)
        self.relu2 = nn.ReLU()

        self.rotFC=nn.Linear(4096,4)
        self.rot=nn.Softmax(dim=0)
        self.relu3 = nn.ReLU()

        self.ssFC=nn.Linear(4096,10)
        self.ss=nn.Softmax(dim=0)
        self.relu4 = nn.ReLU()

        self.armFC=nn.Linear(4096,2)
        self.arm=nn.Softmax(dim=0)
        self.relu5 = nn.ReLU()
    
    def forward(self,x):
        p=self.posFC(x)
        p=self.relu1(p)
        #print(p.shape," val")
        p=self.pos(p)

        tw=self.TWFC(x)
        tw=self.relu2(tw)
        tw=self.TW(tw)

        r=self.rotFC(x)
        r=self.relu1(r)
        r=self.rot(r)

        s=self.ssFC(x)
        s=self.relu1(s)
        s=self.ss(s)

        ar=self.armFC(x)
        ar=self.relu1(ar)
        ar=self.arm(ar)

        return p,tw,r,s,ar