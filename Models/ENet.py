from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math as m

class ENet (nn.Module):
    def __init__ (self, in_dims, feats):
        super (ENet, self).__init__ ()
        self.name = "ENet"
        self.conv1 = nn.Conv2d(in_dims [0], feats[0], 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(feats[0], feats[1], 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(feats[1], feats[2], 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(feats[2], feats[3], 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        H_dim = m.ceil (in_dims [1] / 16) - 1
        self.out_dim = feats [-1]

        self.latern = nn.Linear (H_dim * H_dim * 64, feats[-1])
        

    def forward (self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.latern (x)

        return x

def test ():
    shape = [1, 1, 64, 64]
    feats = [32, 32, 64, 64, 1024]

    inp = torch.randn (shape)
    net = ENet (in_dims=shape [1:], feats=feats)

    out = net (inp)

    print (out.shape)
    