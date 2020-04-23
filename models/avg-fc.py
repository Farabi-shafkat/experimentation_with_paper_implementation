import torch
import torch.nn as nn



class AVG_FC(nn.Module):
    def __init__(self):
        self.fc=nn.linear(8192,4096)

    def forward(self, x):   
        avg = torch.sum(x, dim=1)/x.shape[1]
        x = fc(x)
        return x


class linearRegression(torch.nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(4096, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


class classficatiion(torch.nn.Module):
    def __init__(self):
        self.fc=nn.linear(4096,)
