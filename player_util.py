from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
from Utils.utils import normal  # , pi
import copy

class Agent (object):
    def __init__ (self, model, env, args, state, rank=0):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.t_lbl = None
        self.t_gt_lbl = None

        self.gpu_id = -1
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        self.cell_probs = []
        self.probs = []
        

    def action_train (self, use_max=False, use_lbl=False, eps=0.99):
        value, logit = self.model (Variable(self.state.unsqueeze(0)))

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        prob_tp = prob.permute (0, 2, 1)
        log_prob_tp = log_prob.permute (0, 2, 1)
        distribution = torch.distributions.Categorical (prob_tp)
        # distribution = torch.distributions.Categorical (torch.clamp (prob_tp, 0.05, 0.95))
        shape = prob_tp.shape
        if not use_max:
            action_tp = distribution.sample ().reshape (1, shape[1], 1)
            action = action_tp.permute (0, 2, 1)
            self.action = action.cpu().numpy() [0][0]

            log_prob = log_prob.gather(1, Variable(action))
            state, self.reward, self.done, self.info = self.env.step(
                self.action)

            self.state = torch.from_numpy(state).float()

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward [None][None])
        return self

    def action_test (self):
        with torch.no_grad():
            value, logit = self.model(Variable (self.state.unsqueeze(0)))
            
        prob = F.softmax (logit, dim=1)
        self.probs.append (prob.data.cpu ().numpy () [0][1])

        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        # print ("action test", self.rewards)
        self.actions.append (action [0])
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        self.cell_probs = []
        self.probs = []
        return self

