import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *
from core.bitchop import *
from core.networktrace import NetworkTracing

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='BitChop: Reducing memory throughput in ImageNet for MobileNetV3')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet_v3_small',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: mobilenet_v3_small)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', type=str, default="0",
                    help='GPU id to use.')
parser.add_argument('--disable-bitchop', action='store_true',  default=False, help='Disables training with BitChop')
parser.add_argument('--initial-bits-chopped', type=int, default=1, help='Number of bits BitChop starts off chopping (default: 1)')
parser.add_argument('--policy-file-name', type=str, default="discrete", help='The policy .py file name to use (default: "discrete")')
parser.add_argument('--policy-settings-path', type=str, default=None, help='The path to the policy .json file name to use as settings (default: None)')
parser.add_argument('--activity-trace', action='store_true',  default=False, help='Lets BitChop calculate the activity factor for the training values for hardware designing purposes')
parser.add_argument('--sparsity', action='store_true',  default=False, help='Lets BitChop calculate the sparsity for the training values for hardware designing purposes')
parser.add_argument('--bf16_backprop', action='store_true',  default=False, help='Lets BitChop simulate a backpropagation using the BFloat16 floating point system')

network_tracer = None

def main():
    args = parser.parse_args()

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
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')


    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global network_tracer

    device = torch.device("cuda:"+str(args.gpu) if args.gpu is not None else "cpu")

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)


    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Using BitChop: " + str(not args.disable_bitchop))
    timestr = time.strftime("%Y%m%d_%H%M%S_")

    # tracing utils
    tracing_frequency = None
    extra_path_string = ""
    if args.activity_trace:
        tracing_frequency = 1001
        print("Activity factor tracing will commence on the network")
        extra_path_string += "_acttraced"
    if args.sparsity:
        tracing_frequency = 1001
        print("Will calculate sparsity during training")
        extra_path_string += "_sparsity"
    if args.bf16_backprop:
        tracing_frequency = 1001
        print("Will simulate a back propagation using BFloat16")
        extra_path_string += "_bf16Backprop"

    # disable or enable BitChop, load policy and load BitChop
    if not args.disable_bitchop:
        policy = load_policy_class(args.policy_file_name, args.policy_settings_path, "imagenet")

        bitchop_init_params = dict()
        bitchop_init_params["initial_num_bits_chopped"] = args.initial_bits_chopped
        bitchop = BitChop(policy, model, bitchop_init_params)
        # set the logging path and enable or disable tracing for multiple things
        initialize_logging_and_tracing("bitchop_" + policy.get_policy_name() + timestr + args.arch + "_imagenet" + extra_path_string, model, bitchop=bitchop, tracing_frequency=tracing_frequency, activity_tracing=args.activity_trace, sparsity_tracing=args.sparsity, bf16_backprop=args.bf16_backprop)
    else:
        bitchop = None
        # set the logging path without BitChop and enable or disable tracing for multiple things
        initialize_logging_and_tracing("baseline_" + timestr + args.arch + "_imagenet" + extra_path_string, model, bitchop=bitchop, tracing_frequency=tracing_frequency, activity_tracing=args.activity_trace, sparsity_tracing=args.sparsity, bf16_backprop=args.bf16_backprop)

    for epoch in range(0, args.epochs):

        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(model, device, optimizer, criterion, epoch, train_loader, bitchop=bitchop ,activity_factor_frequency=tracing_frequency)
        test(model, device, criterion, val_loader, bitchop)
        scheduler.step()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()