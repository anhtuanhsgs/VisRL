from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *

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
    choices=["ENet"]
)

parser.add_argument (
    '--data',
    default='cremi',
    choices=["cremi"]
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

def setup_env_conf (args):
    env_conf = {

    }

    return env_conf

def setup_data (args):
    path_test = None
    if args.data == "cremi":
        path_train = "Data/Cremi2D/train/"
        path_train = "Data/Cremi2D/train/"
        path_test = "Data/Cremi2D/train/"
        args.data_channel = 1
        args.test_lbl = True


    raw, gt = get_data (path=path_train, data_channel=args.data_channel)
    raw_val, gt_val =  get_data (path=path_valid, data_channel=args.data_channel)
    raw_test, gt_test = None, None

    print ("train: ", len (raw), raw [0].shape)
    print ("valid: ", len (raw_valid), raw_valid [0].shape)

    return raw, gt, raw_val, gt_val, raw_test, gt_test

def main (scripts, args):
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    args.scripts = scripts

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    mp.set_start_method('spawn')

    raw, gt, raw_val, gt_val, raw_test, gt_test = setup_data (args)

    print (len (raw), raw[0].shape)

    # shared_model = get_model (args)

    # if args.load:   
    #     saved_state = torch.load(
    #         args.load,
    #         map_location=lambda storage, loc: storage)
    #     shared_model.load_state_dict(saved_state)
    # if not args.deploy:
    #     shared_model.share_memory()

    # if args.deploy:
    #     print ("NOT IMPLEMENTED")
    #     pass
    #     exit ()

    # if args.shared_optimizer:
    #     if args.optimizer == 'RMSprop':
    #         optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
    #     if args.optimizer == 'Adam':
    #         optimizer = SharedAdam(
    #             shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    #     optimizer.share_memory()
    # else:
    #     optimizer = None


    # processes = []
    # if not args.no_test:
    #     if raw_test is not None:
    #         p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], (raw_test, gt_lbl_test), shared_dict))
    #     else:
    #         p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], None, shared_dict))
    #     p.start()
    #     processes.append(p)
    
    # time.sleep(0.1)

    # for rank in range(0, args.workers):
    #     p = mp.Process(
    #         target=train_func, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl], shared_dict))

    #     p.start()
    #     processes.append(p)
    #     time.sleep(0.1)

    # for p in processes:
    #     time.sleep(0.1)
    #     p.join()

if __name__ == '__main__':
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    main (scripts, args)