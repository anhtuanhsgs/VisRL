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