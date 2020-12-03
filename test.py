from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import Debug_env
from Models.models import *
from player_util import Agent
from torch.autograd import Variable
import time, os
import logging
from Utils.Logger import Logger
from Utils.utils import create_dir
from Utils.utils import ScalaTracker, Scheduler
import numpy as np
import cv2
import skimage.io as io
from datetime import datetime
from skimage.measure import label

def deploy (args, shared_model, player, gpu_id):
    os.makedirs ("deploy/", exist_ok=True)

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model.load_state_dict (shared_model.state_dict ())
    player.model.eval ()

    tot_num_calculation = 0

    for i in range (11):
        print ("Processing ", i)
        player.clear_actions ()
        state = player.env.reset_deploy (i)
        player.state = torch.from_numpy (state).float ()
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.state = player.state.cuda ()

        player.action_test ()    
        result = np.zeros ((2560, 2560), dtype=np.int32)
        for pred_layer in player.env.pred_list:
            labeled = label (pred_layer > 128) + np.max (result) * (pred_layer > 128)
            result = np.maximum (result, labeled).astype (np.int32)

        io.imsave ("deploy/" + str (i) + ".tif", result);
    print ("avg. patches processed", tot_num_calculation / 11)
    print ("DONE")

def test_func (args, shared_model, env_conf, datasets):
    ptitle ('Valid agent')

    gpu_id = args.gpu_ids [-1]
        
    env_conf ["env_gpu"] = gpu_id

    if not args.deploy:
        logger = Logger (args.log_dir)

        saved_src_dir = arsg.log_dir + "/src/"
        create_dir (saved_code)
        os.system ("cp *.py " + saved_src_dir)
        os.system ("cp *.sh " + saved_src_dir)
        os.system ("cp -r models " + saved_src_dir)
        os.system ("cp -r runs_cripts " + saved_src_dir)
        os.system ("cp -r Utils " + saved_src_dir)

    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)

    env = Debug_env (datasets, env_conf)

    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    player.model = get_model (args, "ENet", input_shape=env_conf["obs_shape"], 
                                    num_actions=args.num_actions * 3)
    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model = player.model.cuda ()
            player.state = player.state.cuda ()
    player.model.eval ()

    flag = True
    if not args.deploy:
        create_dir (args.save_model_dir)

    recent_episode_scores = ScalaTracker (100)
    recent_rand_i = ScalaTracker (100)

    renderlist = []
    renderlist.append (player.env.render ())
    max_score = 0

    if args.deploy:
        deploy (args, shared_model, player, gpu_id)
        exit ()

    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            player.model.eval ()
            flag = False

        player.action_test ()
        reward_sum += player.reward.mean ()
        renderlist.append (player.env.render ()) 

        if player.done:
            flag = True
            num_tests += 1

            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests

            print ("VALID: Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".format (
                    time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                    reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores .push (reward_sum)

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, str (num_tests)))

            if num_tests % args.log_period == 0:
                print ("----------------------VALID SET--------------------------")
                print (args.env)
                print ("Log test #:", num_tests)
                print ("rewards: ", player.reward.mean ())
                print ("sum rewards: ", reward_sum)
                log_rewards = [int (rew * 100) for rew in player.env.sum_rewards]
                print ("rewards:", log_rewards)
                print ("action: ", player.env.actions)
                print ("reward history: ", player.env.rewards)
                print ("------------------------------------------------")

                log_img = np.concatenate (renderlist, 0)
                log_info = {"valid_sample": log_img}


                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                if not args.deploy:
                    log_info = {
                        'mean_valid_reward': reward_mean,
                        '100_mean_reward': recent_episode_scores.mean (),
                    }

                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)

            renderlist = []
            reward_sum = 0
            player.eps_len = 0            
            
            player.clear_actions ()
            state = player.env.reset ()
            renderlist.append (player.env.render ())

            time.sleep (15)
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()