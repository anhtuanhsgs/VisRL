import time, os, warnings, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, disk, ball
from skimage import img_as_bool
from natsort import natsorted
import cv2
import math as m
from .img_aug_func import *

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

def create_dir (directory):
    folders = directory.split ("/")
    path = ""

    for folder in folders:
        path += folder + "/"
        if not os.path.exists(path):
            os.makedirs(path)

def vols2list (vols):
    ret = []
    for vol in vols:
        for img in vol:
            ret += [img]
    return ret

def get_data (path, data_channel=1):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    train_path += natsorted (glob.glob (path + "A/*.npy"))
    train_label_path += natsorted (glob.glob (path + "B/*.npy"))

    
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        if len (X_train) == 1:
            X_train = X_train [0]
        elif X_train [0].ndim > 2:
            X_train =  vols2list (X_train)
    if (len (y_train) > 0):
        if len (y_train) == 1:
                y_train = y_train [0]
        elif y_train [0].ndim > 2:
            y_train = vols2list (y_train)
    else:
        y_train = np.zeros_like (X_train)

    return X_train, y_train


class Scheduler ():
    def __init__ (self, var, schedule, delta):
        self.var = var
        self.schedule = schedule
        self.delta = delta
        self.iter = 0
        self.schedule_index = 0

    def next (self):
        self.iter += 1
        idx = self.schedule_index
        if idx < len (self.schedule) and self.iter >= self.schedule [idx] :
            self.var += self.delta
            self.schedule_index += 1
        return self.var

    def value (self):
        return self.var

class EspTracker ():
    def __init__ (self, eps, eps_step):
        self.eps = eps
        self.eps_step = eps_step
        self.index = 0
        self.value = eps [0]
        self.nstep = 0

    def step (self, n):
        self.nstep += n
        if (self.nstep <= self.eps_step [0]):
            return
        if self.value <= self.eps[-1] or self.index >= len (self.eps) - 1:
            return
        index = self.index
        self.value -= (self.eps[index-1] - self.eps[index]) / (self.eps_step[index]-self.eps_step[index-1]) * n
        if index < len (self.eps) and self.value <= self.eps [index]:
            self.index += 1

class ScalaTracker ():
    def __init__ (self, size):
        self.arr = []
        self.size = size

    def push (self, x):
        self.arr.append (x)
        if len (self.arr) > self.size:
            self.arr.pop (0)

    def mean (self):
        return np.mean (self.arr)

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        if param.grad is None:
            shared_param._grad = None
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def normal(x, mu, sigma, gpu_id, gpu=False):
    pi = np.array([m.pi])
    pi = torch.from_numpy(pi).float()
    if gpu:
        with torch.cuda.device(gpu_id):
            pi = Variable(pi).cuda()
    else:
        pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b