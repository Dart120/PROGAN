import torch
import torch.nn as nn
import torch.nn.functional as F
from EqConv2d import EqConv2D

class G_0(nn.Module):
    def __init__(self,):
        super(G_0, self).__init__()
        self.conv_1 = EqConv2D(512,512,(4,4),1,(4-1)/2)
        self.conv_2 = EqConv2D(512,512,(3,3),1,(3-1)/2)
 
    def forward(self,x):
        x = self.conv_1(x)
        x = F.leaky_relu(x,0.2)
        x = self.conv_2(x)
        x = F.leaky_relu(x,0.2)
        return x
