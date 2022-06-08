import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')

# dataloader 
parser.add_argument('--data_path', default='../datasets')
parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet-100', 'imagenet', 'coco'])
parser.add_argument('--batch_size', default=512, type=int)                          
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# model 
parser.add_argument('--arch', default='resnet18', type=str)

# training 
parser.add_argument('--method', default='simclr', choices=['simclr', 'moco', 'simsiam'])
parser.add_argument('--epochs', default=1000, type=int)                   
parser.add_argument('--knn_eval_freq', default=0, type=int)

parser.add_argument('--resume', action='store_true')
parser.add_argument('--saved_path', default='none', type=str)

parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--lr', default=0.5, type=float)                 
parser.add_argument('--wd', default=1e-4, type=float)                 
parser.add_argument('--cos', action='store_true', default=True)

parser.add_argument('--moco-k', default=65536, type=int)
parser.add_argument('--moco-m', default=0.999, type=float)

parser.add_argument('--trial', default=0, type=int)

# ddp 
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

parser.add_argument('--local', default='', type=str)
parser.add_argument('--job_id', default='', type=str)

args = parser.parse_args()
import os 
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local
import sys 
import warnings 
import builtins
import random
from torch.nn.modules.module import T
from tqdm import tqdm 

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn 
import torch.multiprocessing as mp 
import torch.distributed as dist 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from loaders.dataloader import set_loader
from methods import set_model
from methods.base import CLTrainer
from simsiam_utils_folder.util import * 

import datetime

if args.resume:
    assert args.saved_path != 'none'

# for Logging 
args.saved_path = os.path.join("../CL_logs/{}_{}_{}-{}_{}-{}-{}".format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), args.dataset, args.method, args.arch, args.seed, args.trial))
if not os.path.exists(args.saved_path):
    os.makedirs(args.saved_path)
# tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)

def main():
    print(args.saved_path)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    main_worker(args.gpu, ngpus_per_node, args) 

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating cnn model '{}'".format(args.arch))
    
    model = set_model('simsiam', args.arch, args.dataset)
    trainer = CLTrainer(args)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    # create data loader
    train_loader, train_sampler, ft_loader, ft_sampler, test_loader, memory_loader = set_loader(args)
    
    # create optimizer
    optimizer = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=args.wd)
    
    if args.resume:
        model_path = os.path.join(args.saved_path, 'last.pth.tar')
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(model_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("=> loaded checkpoint '{}' (epoch {}, lr {})"
                  .format(model_path, checkpoint['epoch'], optimizer.param_groups[0]['lr'] ))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
    else:
        args.start_epoch = 0 

    # Train 
    trainer.train(model, optimizer, train_loader, test_loader, memory_loader, train_sampler)


if __name__ == '__main__':
    main()