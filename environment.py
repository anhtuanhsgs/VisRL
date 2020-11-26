import os, sys, glob, time, copy, random
from os import sys, path

import numpy as np
import skimage.io as io
from skimage.transform import rotate

import albumentations as A
import matplotlib.pyplot as plt



from Utils.img_aug_func import *

from natsort import natsorted
from glob import glob

import cv2


class General_env ():
    def init (self, config):
        self.T = config ['T']
        self.size = config ['size']
        self.rng = np.random.RandomState (time_seed ())
        pass

    def seed (self, seed):
        self.rng = np.random.RandomState(seed)
        return self.rng

    def aug (self, image, mask):
        pass

    def step (self, action):
        ret = (self.observation (), reward, done, info)
        return ret

    def observation (self):
        pass

    def reset (self):
        pass

    def reset_end (self):
        pass

    def render (self):
        pass


class Debug_env (General_env):
    def __init__ (self, raw_list, config, seed=0):
        self.init (config)
        self.raw_list = raw_list
        self.rots = [-10, 10]
        self.seed (seed)

    def reset (self, angle=None):
        idx = self.rng.randint (len (self.raw_list))
        self.gt = copy.deepcopy (self.raw_list [idx])
        self.raw = copy.deepcopy (self.raw_list [idx])

        print (self.raw.shape)

        if angle is None:
            angle = self.rng.randint (-30, 30)
            self.angle = angle
            self.raw = rotate (self.raw, angle)
        self.angle = angle

        self.sum_rewards = 0
        self.rewards = []

        self.reset_end ()
        return self.observation ()

    def step (self, action):
        done = False

        if action == len (self.rots):
            done = True
            reward = 0
            info = {}
            ret = (self.observation (), reward, done, info)
            return ret

        old_angle = self.angle
        self.angle += self.rots [action]
        # self.raw = rotate (self.raw, self.rots [action])
        self.raw = rotate (self.gt, self.angle)

        reward = - (0 - old_angle) + (0 - self.angle)
        info = {}

        self.sum_rewards += reward
        self.rewards.append (reward)

        ret = (self.observation (), reward, done, info)
        return ret

    def observation (self):
        return self.raw

    def render (self):
        return self.raw

def test():
    path = "Data/Cremi2D/train/A/*.tif"
    pths = natsorted (glob (path))
    X_train = read_im (pths)
    X_train = [X_train [0]]

    config = {
        "T": 10,
        "size": X_train [0].shape,
    }
    env = Debug_env (X_train, config)
    obs = env.reset ()

    plt.imshow (obs)
    plt.show ()

    for i in range (10):
        action = int (input ())
        obs, reward, done, info = env.step (action)
        print ("rew: ", reward, "rot: ", env.angle)
        plt.imshow (obs)
        plt.show ()

    print (len (X_train))