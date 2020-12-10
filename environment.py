import os, sys, glob, time, copy, random
from os import sys, path

import numpy as np
import skimage.io as io
from skimage.transform import rotate

import albumentations as A
import albumentations.augmentations.functional as F
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
        self.is3D = config ["3D"]
        self.obs3D = config ["obs3D"]
        self.alpha_only = config ["alpha_only"]

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
        self.color_step = config ["color_step"]
        self.lut_init = config ["lut_init"]
        self.ref_lut_init = config ["ref_lut_init"]

        alpha_only = self.alpha_only
        self.lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step, is3D=self.is3D, initial=self.lut_init, alpha_only=alpha_only)
        self.ref_lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step, is3D=self.is3D, initial=self.ref_lut_init, alpha_only=alpha_only)

        self.step_cnt = 0

        self.deploy = []

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

    def aug3D (self, vols):
        num_rot_xy = self.rng.randint (4)
        # num_rot_yz = self.rng.randint (4)
        num_rot_yz = self.rng.choice ([1, 3], 1) [0]
        useFlipX = self.rng.randint (2)
        useFlipY = self.rng.randint (2)
        useFlipZ = self.rng.randint (2)

        ret = []
        for vol in vols:
            vol = np.rot90 (vol, num_rot_xy, axes=(1,2))
            vol = np.rot90 (vol, num_rot_yz, axes=(0,1))

            if useFlipZ:
                vol = np.flip (vol, axis=0)
            if useFlipX: 
                vol = np.flip (vol, axis=1)
            if useFlipY:
                vol = np.flip (vol, axis=2)

            ret.append (vol)

        # Full AUG

        if not self.alpha_only:
            if self.raw.shape [0] < 50:
                angle = self.rng.randint (10)
                scale = self.rng.uniform (0.9, 1.1)

                dx = self.rng.randint (-2, 2)
                dy = self.rng.randint (-2, 2) 
            else:           

                angle = self.rng.randint (30)
                scale = self.rng.uniform (0.8, 1.2)

                dx = self.rng.randint (-4, 4)
                dy = self.rng.randint (-4, 4)

            for vol in ret:
                for i, img in enumerate (vol):
                    vol [i] = F.shift_scale_rotate (img, angle, scale, dx, dy, 
                                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101)

        return ret

    def step_ (self):
        self.rng.seed (10)

        if not self.is3D:
            nactions = self.num_actions * 3
        else:
            nactions = self.num_actions * 4
        if self.alpha_only:
            nactions = self.num_actions

        for j in range (self.T):
            actions = [1] * nactions 
            for i in range (len (actions)):
                if not self.is3D:
                    idx, c = i//3, i%3
                else:
                    idx, c = i//4, i%4
                if self.alpha_only:
                    idx, c = i, 3

                if self.lut.table [idx][c] > self.ref_lut.table [idx][c]:
                    if self.rng.rand () > 0.05:
                        actions [i] = 0
                if self.lut.table [idx][c] < self.ref_lut.table [idx][c]:
                    if self.rng.rand () > 0.05:
                        actions [i] = 2

            self.step (actions)
        ret = copy.deepcopy (self.deploy)
        self.reset ()
        return ret

    def reset (self, angle=None):
        idx = self.rng.randint (len (self.raw_list))
        if (self.DEBUG):
            idx = 0
        self.ref = copy.deepcopy (self.raw_list [idx])
        self.raw = copy.deepcopy (self.raw_list [idx])

        if self.alpha_only:
            self.ref, self.raw = self.aug3D ([self.ref, self.raw])
        if not self.alpha_only:
            self.ref = self.aug3D ([self.ref]) [0]
        alpha_only = self.alpha_only

        self.lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step, is3D=self.is3D, initial=self.lut_init, alpha_only=alpha_only)
        self.ref_lut = LUT (rng=self.rng, n=self.num_actions, color_step=self.color_step, is3D=self.is3D, initial=self.ref_lut_init, alpha_only=alpha_only)

        # self.lut.rand_mod ()
        self.ref_lut.rand_mod ()
        if self.alpha_only:
            for c in range (3):
                for i in range (256):
                    self.lut.table [i][c] = self.ref_lut.table [i][c]

        self.diff_t0 = self.ref_lut.table - self.lut.table

        # self.ref = self.aug (self.ref, self.ref) [0]

        if not self.is3D:
            self.sum_rewards = np.zeros ([self.num_actions * 3], dtype=np.float32)
        else:
            self.sum_rewards = np.zeros ([self.num_actions * 4], dtype=np.float32)

        if self.alpha_only:
            self.sum_rewards = np.zeros ([self.num_actions], dtype=np.float32)
        self.rewards = []

        self.step_cnt = 0
        self.actions = []
        self.deploy = [
                        (self.lut.apply (self.raw), 
                        self.rasterize (self.ref_lut.apply (self.ref)),
                        copy.deepcopy (self.lut.table),
                        copy.deepcopy (self.ref_lut.table),)
                    ]

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
            if not self.is3D:
                idx, c = i//3, i%3
            else:
                idx, c = i//4, i%4

            if self.alpha_only:
                idx, c = i, 3

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
            if (old_diff - new_diff) > 0:
                rewards [i] += color_step
            elif (old_diff - new_diff) < 0:
                rewards [i] -= color_step
            # rewards [i] = 1.0 * (old_diff - new_diff)


        self.deploy += [(self.lut.apply (self.raw), 
                            self.rasterize (self.ref_lut.apply (self.ref)), 
                            copy.deepcopy (self.lut.table), 
                            copy.deepcopy (self.ref_lut.table))]

        self.actions.append (self.action)
        rewards /= color_step
        self.rewards.append (rewards)

        self.sum_rewards += rewards
        info = {}
        ret = (self.observation (), rewards, done, info)
        self.step_cnt += 1
        return ret

    def observation (self):
        if not self.is3D:
            raw = np.transpose (self.lut.apply (self.raw), [2, 0, 1])
            ref = np.transpose (self.ref_lut.apply (self.ref), [2, 0, 1])
            obs = np.concatenate ([raw, ref], 0)
        else:
            if not self.obs3D:
                raw = self.rasterize (self.lut.apply (self.raw))
                raw = np.transpose (raw, [2, 0, 1])
                ref = self.rasterize (self.ref_lut.apply (self.ref))
                ref = np.transpose (ref, [2, 0, 1])
                obs = np.concatenate ([raw, ref], 0)
            else:
                raw = self.lut.apply (self.raw)
                raw = np.transpose (raw, [3, 0, 1, 2])
                ref = self.rasterize (self.ref_lut.apply (self.ref))
                dummy = np.zeros (ref.shape[:2] + (1,), dtype=ref.dtype)
                ref = np.concatenate ([ref, dummy], -1)
                ref = np.expand_dims (np.transpose (ref, [2, 0, 1]), 1)

                obs = np.concatenate ([raw, ref], 1)
        return obs / 255.0

    def clip (self, x, l=0, r=255):
        return np.maximum (np.minimum (x, r), l) 

    def rasterize (self, rgba_vol):
        ret = np.zeros (rgba_vol.shape [1:3] + (3,), dtype=np.float32)
        alpha = np.zeros (rgba_vol.shape [1:3] + (1,), dtype=np.float32)

        alpha += 0.001     

        for i in range (len (rgba_vol)):
            ret = self.clip (ret + rgba_vol [i,:,:,:3].astype (np.float32) 
                        * rgba_vol [i,:,:,3:].astype (np.float32) * 0.6 / 255.0 
                        * (1 - alpha))
            alpha = self.clip (alpha + rgba_vol [i,:,:,3:].astype (np.float32) / 255.0  * (1. - alpha))

        return ret.astype (np.uint8)

    def render (self):
        if not self.is3D:
            raw = self.lut.apply (self.raw)
            ref = self.ref_lut.apply (self.ref) 
            img = np.concatenate ([raw, ref], 1)
            img = img.astype (np.uint8)
        else:
            raw_rgba = self.lut.apply (self.raw)
            ref_rgba = self.ref_lut.apply (self.ref)
            
            raw_ras = self.rasterize (raw_rgba)
            ref_ras = self.rasterize (ref_rgba)

            img = np.concatenate ([raw_ras, ref_ras], 1)
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


