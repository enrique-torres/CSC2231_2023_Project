'''utils.py: Utility Functions'''
import glob
import importlib
import inspect
import os
import platform
import sys
import time
import math
import json

from csv import writer
from pathlib import Path

import torch.nn as nn
import torch.nn.init as init

if platform.system() != "Windows":
	_, term_width = os.popen('stty size', 'r').read().split()
	term_width = int(term_width)
else:
	term_width = int(os.popen('mode con | findstr Columns','r').read().strip('\n').strip(' ').split(':')[1].strip(' '))

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

start_time          = 0
total_time          = 0

train_csv_exists    = False
test_csv_exists     = False
train_step_counter  = 0
test_step_counter   = 0


def progress_bar(current, total, msg=None):
	global last_time, begin_time
	global start_time, total_time
	
	if current == 0:
		begin_time = time.time()  # Reset for new bar.
	
	total_time = time.time() - start_time 

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

	sys.stdout.write(' [')
	for i in range(cur_len):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(rest_len):
		sys.stdout.write('.')
	sys.stdout.write(']')

	cur_time = time.time()
	step_time = cur_time - last_time
	last_time = cur_time
	tot_time = cur_time - begin_time

	L = []
	L.append('  Step: %s' % format_time(step_time))
	L.append(' | Tot: %s' % format_time(tot_time))
	if msg:
		L.append(' | ' + msg)

	msg = ''.join(L)
	sys.stdout.write(msg)
	for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
		sys.stdout.write(' ')

	# Go back to the center of the bar.
	for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
		sys.stdout.write('\b')
	sys.stdout.write(' %d/%d ' % (current+1, total))

	if current < total-1:
		sys.stdout.write('\r')
	else:
		sys.stdout.write('\n')
	sys.stdout.flush()

	return step_time

def format_time(seconds):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	i = 1
	if days > 0:
		f += str(days) + 'D'
		i += 1
	if hours > 0 and i <= 2:
		f += str(hours) + 'h'
		i += 1
	if minutes > 0 and i <= 2:
		f += str(minutes) + 'm'
		i += 1
	if secondsf > 0 and i <= 2:
		f += str(secondsf) + 's'
		i += 1
	if millis > 0 and i <= 2:
		f += str(millis) + 'ms'
		i += 1
	if f == '':
		f = '0ms'
	return f

def start_timer():
	global start_time
	start_time = time.time()


def get_total_time():
	global total_time
	return format_time(total_time)


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def adjust_learning_rate_cosine(optimizer, epoch, iteration, num_iters):
	if global_arguments is None:
			print("Arguments wasn't given to train function, exiting")
			exit(1)
	lr = optimizer.param_groups[0]['lr']
	warmup = False

	warmup_epoch = 5 if warmup else 0
	warmup_iter = warmup_epoch * num_iters
	current_iter = iteration + epoch * num_iters
	max_iter = global_arguments.epochs * num_iters

	lr = global_arguments.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

	if epoch < warmup_epoch:
			lr = global_arguments.lr * current_iter / warmup_iter

	for param_group in optimizer.param_groups:
			param_group['lr'] = lr

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def create_logging_folder(data_output_path):
	Path(data_output_path).mkdir(parents=True, exist_ok=True)

def write_to_trainlog(prefix, elems_to_log):
	global train_csv_exists, train_step_counter

	with open(prefix+"/training_metrics.csv", 'a+', newline='', encoding='utf-8') as csvfile:
		# Create a writer object from csv module
		csv_writer = writer(csvfile)
		if not train_csv_exists:
			train_csv_exists = True
			csv_writer.writerow(["step", "time", "top1", "top5", "loss"])     
		# Add contents of list as last row in the csv file
		for row in elems_to_log:
			row.insert(0,train_step_counter)
			csv_writer.writerow(row)
			train_step_counter += 1


def write_to_testlog(prefix, elems_to_log):
	global test_csv_exists, test_step_counter

	with open(prefix+"/testing_metrics.csv", 'a+', newline='', encoding='utf-8') as csvfile:
		# Create a writer object from csv module
		csv_writer = writer(csvfile)
		if not test_csv_exists:
			test_csv_exists = True
			csv_writer.writerow(["step","top1", "top5", "loss"])     
		# Add contents of list as last row in the csv file
		for row in elems_to_log:
			row.insert(0,test_step_counter)
			csv_writer.writerow(row)
			test_step_counter += 1

def save_model_checkpoint(args, model, optimizer, epoch):
	state = {
		'epoch': epoch,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict()
	}
	torch.save(state, args.output_path + "/saved_state_epoch_" + str(epoch))
	print("Stored checkpoint of model at " + args.output_path + "/saved_state_epoch_" + str(epoch))

def load_model_checkpoint(args, model, optimizer):
	try:
		checkpoint = torch.load(args.load_checkpoint)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		last_trained_epoch = checkpoint['epoch']
		print("Retrieved model checkpoint")
		return last_trained_epoch, model, optimizer
	except Exception as ex:
		print("No model checkpoint could be found, exiting:")
		print(str(ex))
		exit(1)