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
        self.DEBUG = config ['DEBUG']

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
        self.color_step = 20
        self.lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step)
        self.ref_lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step)

        self.step_cnt = 0

        self.seed (seed)

    def aug (self, image, mask):
        aug = A.Compose([
            A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=1, sigma=3, alpha_affine=3, interpolation=cv2.INTER_NEAREST),
                    # A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                ], p=1.0),
                A.ShiftScaleRotate (p=1.0, shift_limit=0.4, rotate_limit=180, interpolation=cv2.INTER_NEAREST, 
                                        scale_limit=(-0.4, 0.4), border_mode=cv2.BORDER_CONSTANT),
            ])
        ret = aug (image=image, mask=np.zeros (image.shape), dtype=np.int32)        

        return ret ['image'], ret ['mask']


    def reset (self, angle=None):
        idx = self.rng.randint (len (self.raw_list))
        if (self.DEBUG):
            idx = 16
        self.ref = copy.deepcopy (self.raw_list [idx])
        self.raw = copy.deepcopy (self.raw_list [idx])

        self.lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step)
        self.ref_lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step)

        self.ref_lut.rand_mod ()

        self.diff_t0 = self.ref_lut.table - self.lut.table

        self.ref = self.aug (self.ref, self.ref) [0]

        self.sum_rewards = np.zeros ([self.num_actions * 3], dtype=np.float32)
        self.rewards = []
        self.step_cnt = 0
        self.actions = []

        self.reset_end ()
        return self.observation ()

    def step (self, action):
        self.action = action
        done = False
        if (self.step_cnt == self.T - 1):
            done = True

        rewards = np.zeros ([len (action)], dtype=np.float32)
        color_step = self.color_step

        for i in range (len (action)):
            idx, c = i//3, i%3
            old_diff = self.lut.cmp (self.ref_lut, idx, c)
            
            if (action [i] == 0):
                self.lut.modify (idx, c, -color_step)
            if (action [i] == 2):
                self.lut.modify (idx, c, +color_step)
            if (action [i] == 1):
                pass
                # if self.lut.cmp (self.ref_lut, idx, c) > self.color_step:
                #     rewards [i] -= color_step / 2
                # elif self.lut.cmp (self.ref_lut, idx, c) == 0:
                #     rewards [i] += color_step / 2

            new_diff = self.lut.cmp (self.ref_lut, idx, c)
            rewards [i] += old_diff - new_diff

        self.actions.append (self.action)
        rewards /= color_step
        self.rewards.append (rewards)

        self.sum_rewards += rewards
        info = {}
        ret = (self.observation (), rewards, done, info)
        self.step_cnt += 1
        return ret

    def observation (self):
        raw = np.transpose (self.lut.apply (self.raw), [2, 0, 1])
        ref = np.transpose (self.ref_lut.apply (self.ref), [2, 0, 1])
        obs = np.concatenate ([raw, ref], 0)
        return obs / 255.0

    def render (self):
        raw = self.lut.apply (self.raw) 
        ref = self.ref_lut.apply (self.ref) 
        img = np.concatenate ([raw, ref], 1)
        img = img.astype (np.uint8)
        return img

def test():
    path = "Data/Random/train/A/*.png"
    pths = natsorted (glob (path))
    X_train = read_im (pths)
    X_train = [X_train [0]]


    config = {
        "T": 3,
        "size": X_train [0].shape,
        "obs_shape": [1, 32, 32],
        "num_actions": 2,
    }

    env = Debug_env (X_train, config)
    obs = env.reset ()

    fig, ax = plt.subplots (1, 3)

    ax [0].imshow (env.raw)
    ax [1].imshow (env.ref)
    ax [2].imshow (env.observation ())

    plt.show ()

    for i in range (10):
        inp = input ()
        actions = [int (inp [0]), int (inp [1])]
        obs, reward, done, info = env.step (actions)
        plt.imshow (obs)
        plt.show ()


