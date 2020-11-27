import time, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, disk, ball
from skimage import img_as_bool
import cv2
import math as m


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

def get_data (path, relabel, data_channel=1):
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