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

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='NNTracer: Tracing deep learning footprint to accelerate it')
parser.add_argument('data', metavar='DIR',
					help='path to dataset')
parser.add_argument('-m', '--model', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', type=str, default="0",
					help='GPU id to use.')

parser.add_argument('--load-checkpoint', type=str, default=None, help='Load checkpoint from given path (default: None)')
parser.add_argument('--save-checkpoint', action='store_true', default=False, help='Save checkpoints after training each epoch')

parser.add_argument('--onecycle', action='store_true', default=False, help='Trains with One Cycle LR scheduler')
parser.add_argument('--cosine-lr', action='store_true', default=False, help='Trains BitChop with cosine LR updating')


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

	args.lr = (args.lr / 256) * args.batch_size
	if args.cosine_lr:
		print("Using cosine LR updating to train")
	if args.onecycle:
		print("Using OneCycle LR scheduler to train")

	main_worker(args.gpu, args)

def main_worker(gpu, args):

	device = torch.device("cuda:"+str(args.gpu) if args.gpu is not None else "cpu")

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	print("=> creating model '{}'".format(args.model))
	model = models.__dict__[args.model]()
	model = model.to(device)
	timestr = time.strftime("%Y%m%d_%H%M%S_")

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

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

	# One Cycle LR training
	if args.onecycle:
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
	else:
		scheduler = None

	timestr = time.strftime("%Y%m%d_%H%M%S_")
    args.output_path = args.model + "_imagenet_" + timestr

	train_test_loop(args, model, device, train_loader, test_loader, optimizer, criterion, scheduler, schedule_per_batch=False)

if __name__ == '__main__':
	main()