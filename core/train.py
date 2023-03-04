# general imports
import copy
from pathlib import Path
from math import cos, pi

# PyTorch imports
import torch
from torch.optim.lr_scheduler import StepLR

# Our imports
from core.utils import *

g_arguments = None
g_model = None
g_device = None
g_optimizer = None
g_criterion = None
g_scheduler = None
g_netshed = None

def train(epoch, train_loader):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_netshed

	print('\nEpoch: %2d' % epoch)
	g_model.train()

	losses  = AverageMeter()
	top1    = AverageMeter()
	top5    = AverageMeter()
	stime   = AverageMeter()
	metrics = list()

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(g_device), target.to(g_device)

		if g_arguments.cosine_lr:
			adjust_learning_rate_cosine(g_optimizer, epoch, batch_idx, len(train_loader))

		
		# compute output
		output = g_model(data)       
		loss = g_criterion(output, target)

		prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
		losses.update(loss.item(), data.size(0))
		top1.update(prec1.item(), data.size(0))
		top5.update(prec5.item(), data.size(0))

		# compute gradient and do SGD step
		g_optimizer.zero_grad()
		loss.backward()
		g_optimizer.step()
		# in case a scheduler is used to iterate per batch
		if g_scheduler:
			g_scheduler.step()

		steptime = 0
		steptime = progress_bar(batch_idx, len(train_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%%' % (losses.avg, top1.avg, top5.avg))
		stime.update(steptime, 1)

		metrics.append([stime.avg, top1.avg, top5.avg, losses.avg])

	write_to_trainlog(g_arguments.output_path, metrics)


def test(test_loader):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_netshed

	g_model.eval()

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()    
	metrics = list()

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(g_device), target.to(g_device)
			output = g_model(data)
			loss = g_criterion(output, target)
			prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

			losses.update(loss.item(), data.size(0))
			top1.update(prec1.item(), data.size(0))
			top5.update(prec5.item(), data.size(0))

			progress_bar(batch_idx, len(test_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%% ' % (losses.avg, top1.avg, top5.avg))

	metrics.append([top1.avg, top5.avg, losses.avg])
	write_to_testlog(g_arguments.output_path, metrics)

def train_test_loop(args, model, device, train_loader, val_loader, optimizer, criterion, scheduler, schedule_per_batch=False):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_netshed

	g_arguments = args
	g_model = model
	g_device = device
	g_optimizer = optimizer
	g_criterion = criterion
	g_scheduler = scheduler
	if g_arguments == None:
		print("No global arguments given to training process, exiting.")
		exit(1)
	elif g_model == None:
		print("No model given to training process, exiting.")
		exit(1)
	elif g_device == None:
		print("No device ID given to training process, exiting.")
		exit(1)
	elif g_optimizer == None:
		print("No optimizer given to training process, exiting.")
		exit(1)
	elif g_criterion == None:
		print("No criterion given to training process, exiting.")
		exit(1)

	last_trained_epoch = 0
	if args.load_checkpoint is not None:
		last_trained_epoch, g_model, g_optimizer = load_model_checkpoint(g_arguments, g_model, g_optimizer)
		print("Training from epoch " + str(last_trained_epoch))

	create_logging_folder(g_arguments.output_path)

	for epoch in range(last_trained_epoch, g_arguments.epochs):
		if g_arguments.save_checkpoint and epoch % args.save_checkpoint_freq == 0:
			save_model_checkpoint(g_arguments, g_model, g_optimizer, epoch)
			
		if not g_arguments.onecycle and not g_arguments.cosine_lr:
			adjust_learning_rate(g_optimizer, epoch, g_arguments)
		else:
			if g_arguments.onecycle:
				print("Last LR value: " + str(g_scheduler.get_last_lr()))
			else:
				print("Last LR value: " + str(g_optimizer.param_groups[0]['lr']))
		# train for one epoch
		train(epoch, train_loader)
		# test on validation set
		test(val_loader)

		if not schedule_per_batch and g_scheduler != None:
			g_scheduler.step()