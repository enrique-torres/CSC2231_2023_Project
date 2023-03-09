import sys
import glob
import importlib
import inspect
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from torch.optim.lr_scheduler import StepLR
import collections
import numpy as np
import time

from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR100 Training with PyTorch')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--model', type=str, default="resnet18", help='Model name to run training on (default: resnet18)')
    parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0")')

    parser.add_argument('--load-checkpoint', type=str, default=None, help='Load checkpoint from given path (default: None)')
    parser.add_argument('--save-checkpoint', action='store_true', default=True, help='Save checkpoints after training each epoch')
    parser.add_argument('--save-checkpoint-freq', type=int, default=99, help='How many epochs must happen between each checkpoint save (default: 99)')    

    parser.add_argument('--onecycle', action='store_true', default=False, help='Trains with One Cycle LR scheduler')
    parser.add_argument('--cosine-lr', action='store_true', default=False, help='Trains with cosine LR updating')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+args.device if use_cuda else "cpu")
    print("Training on: "+str(device))

    print("=> creating model '{}'".format(args.model))
    model = load_untrained_model(args.model, 100)
    
    if model is None:
        print("Model doesn't exist.")
        return
    print("Selected NN Architecture: "+str(args.model))
    model = model.to(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    trainset = datasets.CIFAR100(
        root='./cifar100data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(
        root='./cifar100data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    args.output_path = "datacollection/" + args.model + "_cifar100_" + timestr  

    train_test_loop(args, model, device, train_loader, test_loader, optimizer, criterion, scheduler, schedule_per_batch=False)

if __name__ == '__main__':
    main()