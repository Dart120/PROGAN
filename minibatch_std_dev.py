import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
        pass

    def forward(self,x):
        N,_,H,W = x.shape
        # x is of shape (N C H W)
        # Mean over all images for specific value in same loction in tensor
        batch_mean = torch.mean(x,dim=0,keepdim=True)
        # (1 C H W)
        absolute_deviation = (x - batch_mean) ** 2
        # (N C H W)
        # Mean over all images again
        mean_deviation = torch.mean(absolute_deviation,dim=0,keepdim=True)
        # (1 C H W)
        std_mean_deviation = torch.sqrt(mean_deviation)
        mean_of_all = torch.mean(std_mean_deviation)
        # ()
        to_concat = torch.full((N,1,H,W),mean_of_all).to(device)
        # (N 1 H W)
        return torch.concat([to_concat,x],dim=1)
        ## (N C + 1 H W)
layer = MinibatchStdDev()
fm = torch.randint(low = 1, high = 10,size = (10000,3,128,64)).to(torch.float).to(device)
print(layer(fm).shape)