import torch
import torch.nn as nn
import torch.nn.functional as F
from EqConv2d import EqConv2D
from Minibatch_Std_Dev import MinibatchStdDev

class D_0(nn.Module):
    def __init__(self):
        super(D_0, self).__init__()
        self.MSD = MinibatchStdDev()
        self.conv_1 = EqConv2D(513,512,(3,3))
        self.conv_2 = EqConv2D(512,512,(4,4))
        self.fc = nn.Linear(512,1)
        
 
    def forward(self,x):
        print(x.shape,'here')
        x = self.MSD(x)
        print(x.shape,'after mini')
        x = self.conv_1(x)
        print(x.shape,'after conv 1')
        x = F.leaky_relu(x,0.2)
        print(x.shape)
        x = self.conv_2(x)
        print(x.shape,'after conv 2')
        x = F.leaky_relu(x,0.2)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x
layer = D_0()
data = torch.ones(size=(64,512,4,4))
print(layer(data).shape)