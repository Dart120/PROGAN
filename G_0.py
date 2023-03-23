import torch
import torch.nn as nn
import torch.nn.functional as F
from EqConv2d import EqConv2D

class G_0(nn.Module):
    def __init__(self,):
        super(G_0, self).__init__()
        #TODO #1 conv transpose 2d ?? why ?? need to equalise
        self.conv_1 = nn.ConvTranspose2d(512,512,4,1,0)
        self.conv_2 = EqConv2D(512,512,(3,3))
 
    def forward(self,x):
        x = self.conv_1(x)
        x = F.leaky_relu(x,0.2)
        x = self.conv_2(x)
        x = F.leaky_relu(x,0.2)
        return x
t = torch.ones((64,512,1,1))
layer = G_0()
print(layer(t).shape)