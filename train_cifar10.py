import sys
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import StepLR
import collections
import numpy as np
import time


from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *
from core.bitchop import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BitChop: Reducing memory footprint in CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--model', type=str, default="resnet18", help='Model name to run training on (default: resnet18)')
    parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0"')
    parser.add_argument('--disable-bitchop', action='store_true',  default=False, help='Disables training with BitChop')
    parser.add_argument('--initial-bits-chopped', type=int, default=1, help='Number of bits BitChop starts off chopping (default: 1)')
    parser.add_argument('--policy-file-name', type=str, default="discrete", help='The policy .py file name to use (default: "discrete")')
    parser.add_argument('--policy-settings-path', type=str, default=None, help='The path to the policy .json file name to use as settings (default: None)')
    parser.add_argument('--activity-trace', action='store_true',  default=False, help='Lets BitChop calculate the activity factor for the training values for hardware designing purposes')
    parser.add_argument('--sparsity', action='store_true',  default=False, help='Lets BitChop calculate the sparsity for the training values for hardware designing purposes')
    parser.add_argument('--bf16_backprop', action='store_true',  default=False, help='Lets BitChop simulate a backpropagation using the BFloat16 floating point system')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+args.device if use_cuda else "cpu")
    print("Training on: "+str(device))

    model = load_untrained_model(args.model)
    if model is None:
        print("Model doesn't exist.")
        return

    print("Selected NN Architecture: "+str(args.model))
    print("Using BitChop: "+str(not args.disable_bitchop))
    model.to(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./cifar10data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='./cifar10data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    print("Using BitChop: " + str(not args.disable_bitchop))
    timestr = time.strftime("%Y%m%d_%H%M%S_")

    # tracing utils
    tracing_frequency = 391
    extra_path_string = ""
    if args.activity_trace:
        print("Activity factor tracing will commence on the network")
        extra_path_string += "_acttraced"
    if args.sparsity:
        print("Will calculate sparsity during training")
        extra_path_string += "_sparsity"
    if args.bf16_backprop:
        print("Will simulate a back propagation using BFloat16")
        extra_path_string += "_bf16Backprop"

    if not args.disable_bitchop:
        policy = load_policy_class(args.policy_file_name, args.policy_settings_path, "c10")

        bitchop_init_params = dict()
        bitchop_init_params["initial_num_bits_chopped"] = args.initial_bits_chopped
        bitchop = BitChop(policy, model, bitchop_init_params)
        # set the logging path and enable or disable tracing for multiple things
        initialize_logging_and_tracing("bitchop_" + policy.get_policy_name() + timestr + args.model + "_cifar10" + extra_path_string, model, bitchop=bitchop, tracing_frequency=tracing_frequency, activity_tracing=args.activity_trace, sparsity_tracing=args.sparsity, bf16_backprop=args.bf16_backprop)
    else:
        bitchop = None
        # set the logging path without BitChop and enable or disable tracing for multiple things
        initialize_logging_and_tracing("baseline_" + timestr + args.model + "_cifar10" + extra_path_string, model, bitchop=bitchop, tracing_frequency=tracing_frequency, activity_tracing=args.activity_trace, sparsity_tracing=args.sparsity, bf16_backprop=args.bf16_backprop)

    for epoch in range(1, args.epochs + 1):
        train(model, device, optimizer, criterion, epoch, train_loader, bitchop)
        test(model, device, criterion, test_loader, bitchop)        
        scheduler.step()

if __name__ == '__main__':
    main()