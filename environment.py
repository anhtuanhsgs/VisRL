import os, sys, glob, time, copy, random
from os import sys, path

import numpy as np
import skimage.io as io
from skimage.transform import rotate

import albumentations as A
import matplotlib.pyplot as plt

from Utils.LUT import LUT

from Utils.img_aug_func import *

from natsort import natsorted
from glob import glob

import cv2


class General_env ():
    def init (self, config):
        self.T = config ['T']
        self.size = config ['size']
        self.obs_shape = config ['obs_shape']

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
    def __init__ (self, datasets, config, seed=0):
        self.init (config)

        self.raw_list = datasets [0]
        self.ref_list = None

        self.raw = None
        self.ref = None

        self.num_actions = config ["num_actions"]
        self.lut = LUT (rng=self.rng)
        self.ref_lut = LUT (rng=self.rng)

        self.seed (seed)

    def reset (self, angle=None):
        idx = self.rng.randint (len (self.raw_list))
        self.ref = copy.deepcopy (self.raw_list [idx])
        self.raw = copy.deepcopy (self.raw_list [idx])

        self.ref_lut.rand_mod ()

        self.sum_rewards = np.zeros ([self.num_actions], dtype=np.float32)
        self.rewards = []

        self.reset_end ()
        return self.observation ()

    def step (self, action):
        done = False
        if (self.step_cnt > self.T):
            done = True

        rewards = np.zeros ([len (action)], dtype=np.float32)

        for i in range (len (action)):
            old_diff = self.lut.cmp (self.ref_lut, i)
            if (action [i] == 0):
                self.lut.update (i, -10)
            if (action [i] == 1):
                self.lut.update (i, +10)
            new_diff = self.lut.cmp (self.ref_lut, i)
            rewards [i] += old_diff - new_diff

        self.sum_rewards += reward

        ret = (self.observation (), reward, done, info)
        return ret

    def observation (self):
        return self.lut.apply (self.raw)

    def render (self):
        return self.lut.apply (self.raw)

def test():
    path = "Data/Random/train/A/*.png"
    pths = natsorted (glob (path))
    X_train = read_im (pths)
    X_train = [X_train [0]]

    config = {
        "T": 10,
        "size": X_train [0].shape,
        "obs_shape": [1, 32, 32],
        "num_actions": 2,
    }

    env = Debug_env (X_train, config)
    obs = env.reset ()

    fig, ax = plt.subplots (1, 3)

    ax [0, 1].show (env.raw)
    ax [0, 2].show (env.ref)
    ax [0, 3].show (env.observation ())

    plt.imshow (obs)
    plt.show ()

    for i in range (10):
        inp = input ()
        actions = [int (inp [0]), int (inp [1])]
        obs, reward, done, info = env.step (actions)
        print ("rew: ", reward, "rot: ", env.angle)
        plt.imshow (obs)
        plt.show ()

    print (len (X_train))

