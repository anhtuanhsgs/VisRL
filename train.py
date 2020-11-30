from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import *
from Utils.utils import ensure_shared_grads, EspTracker
from Models.models import *
from player_util import Agent
from torch.autograd import Variable
from Utils.Logger import Logger

import numpy as np
import time

def train_func (rank, args, shared_model, optimizer, env_conf, datasets):
    if args.deploy:
        return
    ptitle('Train {0}'.format(rank))
    print ('Start training agent: ', rank)
    
    if rank == 0:
        logger = Logger (args.log_dir [:-1] + '_losses/')
        train_step = 0

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    env_conf ["env_gpu"] = gpu_id
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    env = Debug_env (datasets, env_conf, seed=args.seed + rank)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop (shared_model.parameters (), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam (shared_model.parameters (), lr=args.lr, amsgrad=args.amsgrad)

    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    player.model = get_model (args, "ENet", input_shape=env_conf["obs_shape"], 
                                        num_actions=args.num_actions)
    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda ()
            player.model = player.model.cuda ()
    player.model.train ()

    if rank == 0:
        eps_reward = 0
        pinned_eps_reward = 0

    while True:
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.model.load_state_dict (shared_model.state_dict ())
        else:
            player.model.load_state_dict (shared_model.state_dict ())
        
        if player.done:
            player.eps_len = 0

            if rank == 0:
                if train_step % args.train_log_period == 0 and train_step > 0:
                    print ("train: step", train_step, "\teps_reward", eps_reward)
                if train_step > 0:
                    pinned_eps_reward = player.env.sum_rewards.mean ()
                    eps_reward = 0

        for step in range(args.num_steps):        
            player.action_train () 
            if rank == 0:
                eps_reward = player.env.sum_rewards.mean ()
            if player.done:
                break

        if player.done:
            state = player.env.reset ()
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

        R = torch.zeros (1, 1, args.num_actions)

        if not player.done:
            value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        
        gae = torch.zeros(1, 1, args.num_actions)

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda ()
        R = Variable(R)

        for i in reversed(range(len(player.rewards))):
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    reward_i = torch.tensor (player.rewards [i]).cuda ()
            else:
                reward_i = torch.tensor (player.rewards [i])

            R = args.gamma * R + reward_i

            advantage = R - player.values[i]
            value_loss = value_loss + (0.5 * advantage * advantage).mean ()
            delta_t = player.values[i + 1].data * args.gamma + reward_i - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i] * Variable(gae)).mean () - \
                (args.entropy_alpha * player.entropies[i]).mean ()


        player.model.zero_grad ()
        sum_loss = (policy_loss + value_loss)

        curtime = time.time ()
        sum_loss.backward ()
        ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
        
        curtime = time.time ()
        optimizer.step ()

        player.clear_actions ()
    
        if rank == 0:
            train_step += 1
            if train_step % args.log_period == 0 and train_step > 0:
                log_info = {
                    'train: value_loss': value_loss, 
                    'train: policy_loss': policy_loss, 
                    'train: eps reward': pinned_eps_reward,
                }

                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, train_step)
