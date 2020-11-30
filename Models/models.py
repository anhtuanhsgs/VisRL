import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time, copy

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
    def __init__ (self, args, backbone1, backbone2, num_actions):
        super (ActorCritic, self).__init__ ()
        self.name = backbone1.name
        
        self.backbone1 = backbone1
        self.backbone2 = backbone2

        self.num_actions = num_actions

        if args.lstm_feats:
            self.lstm = ConvLSTMCell (args.size, last_feat_ch, args.lstm_feats, kernel_size=(3, 3), bias=True)
            last_feat_ch = args.lstm_feats
            self.use_lstm = True
        else:
            self.use_lstm = False

        latent_dim = backbone1.out_dim * 2
        self.latent = nn.Linear (latent_dim, 64)

        self.actor = nn.Linear (64, num_actions * 2)
        self.critic = nn.Linear (64, num_actions)


    def forward (self, x):
        if (self.use_lstm):
            x, (hx, cx) = x
        raw_brach = self.backbone1 (x [:, :3, :, :])
        ref_brach = self.backbone2 (x [:, 3:, :, :])

        if self.use_lstm:
            hx, cx = self.lstm (x, (hx, cx))
            x = hx

        x = torch.cat ([raw_brach, ref_brach], 1)
        x = self.latent (x)

        critic = self.critic (x)
        actor = self.actor (x)

        critic = critic.view (critic.size (0), 1, self.num_actions)
        actor = actor.view (actor.size (0), 2, self.num_actions)
        
        if self.use_lstm:
            ret = (critic, actor, (hx, cx))
        else:
            ret = (critic, actor)

        return ret

def get_model (args, name, input_shape, num_actions):
    if name == "ENet":
        inp_shape_split = copy.deepcopy (input_shape)
        inp_shape_split [0] //= 2
        backbone1 = ENet (in_dims=inp_shape_split, feats=args.feats)
        backbone2 = ENet (in_dims=inp_shape_split, feats=args.feats)
        model =  ActorCritic (args, backbone1, backbone2, num_actions)

    return model

def test ():
    class ARG:
        def __init__ (self):
            self.lstm_feats = 0
            self.feats = [32, 32, 64, 64, 1024]

    args = ARG ()
    shape = [1, 6, 32, 32]
    

    model = get_model (args, "ENet", input_shape=shape[1:], num_actions=3)

    inp = torch.randn (shape)
    value, logit = model (inp)
    print (logit.shape, value.shape)