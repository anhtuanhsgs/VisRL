from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math as m

class Net3D (nn.Module):
    def __init__ (self, in_dims, feats):
        super (Net3D, self).__init__ ()
        self.name = "Net3D"
        self.conv1 = nn.Conv3d(in_dims [0], feats[0], 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(feats[0], feats[1], 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(feats[1], feats[2], 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool3d(2, 2)
        self.conv4 = nn.Conv3d(feats[2], feats[3], 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool3d(2, 2)
        # self.conv5 = nn.Conv3d(feats[2], feats[3], 3, stride=1, padding=1)
        # self.maxp4 = nn.MaxPool3d(2, 2)

        H_dim = m.ceil (in_dims [1] / 16) - 1
        self.out_dim = feats [-1]

        self.latern = nn.Linear (H_dim * H_dim * H_dim * feats [-2], feats[-1])
        

    def forward (self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.latern (x)

        return x