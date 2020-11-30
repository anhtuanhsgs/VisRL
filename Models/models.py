import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from .ENet import ENet

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, is3D=False, bias=True):
        super(outconv, self).__init__()
        if is3D:
            self.conv = nn.Conv3d (in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)
        else:
            self.conv = nn.Conv2d (in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x

class ActorCritic (nn.Module):
    def __init__ (self, args, backbone, num_actions):
        super (ActorCritic, self).__init__ ()
        self.name = backbone.name
        self.backbone = backbone
        self.num_actions = num_actions

        if args.lstm_feats:
            self.lstm = ConvLSTMCell (args.size, last_feat_ch, args.lstm_feats, kernel_size=(3, 3), bias=True)
            last_feat_ch = args.lstm_feats
            self.use_lstm = True
        else:
            self.use_lstm = False

        latent_dim = backbone.out_dim

        self.actor = nn.Linear (latent_dim, num_actions * 2)
        self.critic = nn.Linear (latent_dim, num_actions)


    def forward (self, x):
        if (self.use_lstm):
            x, (hx, cx) = x
        x = self.backbone (x)


        if self.use_lstm:
            hx, cx = self.lstm (x, (hx, cx))
            x = hx

        critic = self.critic (x)
        actor = self.actor (x)


        critic = critic.view (critic.size (0), 1, self.num_actions)
        actor = actor.view (c)
        
        if self.use_lstm:
            ret = (critic, actor, (hx, cx))
        else:
            ret = (critic, actor)

        return ret

def get_model (args, name, input_shape, num_actions):
    if name == "ENet":
        model =  ActorCritic (args, ENet (in_dims=input_shape, feats=args.feats), num_actions)

    return model

def test ():
    class ARG:
        def __init__ (self):
            self.lstm_feats = 0
            self.feats = [32, 32, 64, 64, 1024]

    args = ARG ()
    shape = [1, 1, 32, 32]
    

    model = get_model (args, "ENet", input_shape=shape[1:], num_actions=3)

    inp = torch.randn (shape)
    value, logit = model (inp)
    print (logit.shape, value.shape)