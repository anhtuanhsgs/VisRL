from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import Debug_env
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *

from Models.models import *

from test import test_func
from train import train_func

from Utils.LUT import LUT

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='VisRL',
    )

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--gamma',
    type=float,
    default=0.98,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

parser.add_argument(
    '--workers',
    type=int,
    default=8,
    metavar='W',
    help='how many training workers to use (default: 8)')

parser.add_argument(
    '--num-steps',
    type=int,
    default=5,
    metavar='NS',
    help='n in n-step learning A3C')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=5,
    metavar='M',
    help='Maximum number of coloring steps')

parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',)

parser.add_argument(
    '--load-model-dir',
    default='../trained_models/',
    metavar='LMD',
    help='Directory to load trained models from')

parser.add_argument(
    '--save-model-dir',
    default='logs/trained_models/',
    metavar='SMD',
    help='Directory to save trained models')

parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--save-period',
    type=int,
    default=100,
    metavar='SP',
    help='Save period')

parser.add_argument(
    '--log-period',
    type=int,
    default=10,
    metavar='LP',
    help='Log period')

parser.add_argument (
    '--train-log-period',
    type=int,
    default=16,
    metavar='TLP',
)

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)

parser.add_argument (
    '--size',
    type=int,
    default= [96, 96],
    nargs='+',
    help='Input processing size of agent, if processing size is smaller than size of images in the data, input will be cropped from the image',
)

parser.add_argument (
    '--entropy-alpha',
    type=float,
    default=0.5,
)

parser.add_argument (
    '--model',
    default='ENet',
    choices=["ENet", "Net3D"]
)

parser.add_argument (
    '--data',
    default='cremi',
    choices=["cremi", "Random", "3DVols", "3DChest"]
)

parser.add_argument (
    '--deploy',
    action='store_true',
    help='Enable for test set deployment',
)

parser.add_argument (
    '--lstm-feats',
    type=int,
    default=0,
)

parser.add_argument (
    '--feats',
    type=int, nargs='+',
    default=[32, 32, 64, 64, 512],
    )

parser.add_argument (
    '--valid-gpu',
    type=int,
    default=-1,
    help='Choose gpu-id for the verbose worker',
)

parser.add_argument (
    '--DEBUG',
    action="store_true"
)

parser.add_argument (
    "--no-test",
    action="store_true"
)

parser.add_argument (
    "--num-actions",
    type=int,
    default=2,
)
########################################## VIS ENV ##################################
parser.add_argument (
    "--color-step",
    type=int,
    default=20,
)

parser.add_argument (
    "--alpha-only",
    action="store_true"
)
########################################## VIS ENV ##################################
def setup_env_conf (args):
    env_conf = {
        "data": args.data,
        "T": args.max_episode_length,
        "size": args.size,
        "DEBUG": args.DEBUG,
        "size": args.size,
        "num_actions": args.num_actions,
        "color_step": args.color_step,
        "3D": "3D" in args.data,
        "ref_lut_init": args.ref_lut_init, 
        "lut_init": args.lut_init, 
        "obs3D": "3D" in args.model,
        "alpha_only": args.alpha_only,
    }

    args.is3D = "3D" in args.data

    if not args.obs3D:
        env_conf ["obs_shape"] = [args.data_channel * 2 * 3] + env_conf ["size"]
    else:
        env_conf ["obs_shape"] = [4] + env_conf ["size"]
        env_conf ["obs_shape"] [1] += 1

    args.log_dir += "/" + args.env + "/" 
    args.save_model_dir += '/' + args.env + '/'
    args.env += "_" + args.model

    return env_conf

def setup_data (args, set_type):
    path_test = None
    datasets = {}    

    if args.data == "3DChest":
        raw = read_imgs_from_path ("Data/Medical/" + set_type + "/A/")
        raw = [ vol [::2, ::2, ::2] for vol in raw ]
        datasets = [raw]
        args.data_channel = 1
        args.lut_init = [24, 24, 24, 24, 24, 24, 24]
        args.ref_lut_init = [0, 0, 0, 0, 0, 48, 48]
        args.obs3D = True

    if args.data == "Random":
        raw = read_imgs_from_path ("Data/Random/" + set_type + "/A/")
        datasets = [raw]
        args.data_channel = 1
        args.lut_init = None
        args.ref_lut_init = None
        args.obs3D = False

    if args.data == "3DVols":
        raw = read_imgs_from_path ("Data/3DVols/" + set_type + "/A/")
        datasets = [raw]
        args.data_channel = 1
        args.lut_init = None
        args.ref_lut_init = None
        args.obs3D = False

    return datasets

def main (scripts, args):
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    args.scripts = scripts

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    mp.set_start_method('spawn')

    if not args.deploy:
        train_datasets = setup_data (args, "train")
        valid_datasets = setup_data (args, "valid")

    env_conf = setup_env_conf (args)

    nChan = 3
    if args.is3D:
        nChan = 4
    if args.alpha_only:
        nChan = 1

    if not args.is3D:
        shared_model = get_model (args, "ENet", input_shape=env_conf["obs_shape"], 
                                    num_actions=args.num_actions * nChan)
    elif not args.obs3D:
         shared_model = get_model (args, "ENet", input_shape=env_conf["obs_shape"], 
                                    num_actions=args.num_actions * nChan)
    elif args.obs3D:
        shared_model = get_model (args, "Net3D", input_shape=env_conf["obs_shape"], 
                                    num_actions=args.num_actions * nChan)

    if args.load:   
        saved_state = torch.load(
            args.load,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    if not args.deploy:
        shared_model.share_memory()

    if args.deploy:
        test_datasets = setup_data (args, "test")
        p = mp.Process(target=test_func, args=(args, shared_model, env_conf, test_datasets))
        p.start()
        exit ()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []
    p = mp.Process(target=test_func, args=(args, shared_model, env_conf, valid_datasets))
    p.start()
    processes.append(p)
    
    time.sleep(0.1)


    if not args.deploy:
        for rank in range(0, args.workers):
            p = mp.Process(
                target=train_func, args=(rank, args, shared_model, optimizer, env_conf, train_datasets))

            p.start()
            processes.append(p)
            time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()

if __name__ == '__main__':
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    main (scripts, args)