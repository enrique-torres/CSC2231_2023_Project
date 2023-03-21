import torch
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
import time

from core.utils import *
from netcrunchlibs.netcrunch_bruteforce import bruteforce_crunch
from netcrunchlibs.netcrunch_sa import simulated_annealing_netcrunch
from netcrunchlibs.netcrunch_ga import genetic_algorithm_netcrunch
from netcrunchlibs.netcrunch_greedy_bf import greedy_bf_netcrunch

def calculate_perlayer_relative_size(original_model):

	perlayer_parameters_weighted = []
	# Split the model into its constituent layers
	for name, param in original_model.named_parameters():
		if 'bias' not in name:
			num_params = sum(p.numel() for p in param)
			perlayer_parameters_weighted.append(num_params)
	
	total_parameters = sum(perlayer_parameters_weighted)
	for i, size in enumerate(perlayer_parameters_weighted):
		perlayer_parameters_weighted[i] = size / float(total_parameters)

	return perlayer_parameters_weighted

def run_netcrunch(train_loader, device, model, criterion, args):

	losses  = AverageMeter()
	top1    = AverageMeter()
	top5    = AverageMeter()
	stime   = AverageMeter()
	
	# Set the model to evaluation mode
	model.eval()

	# calculate how much each layer contributes to the number of elements of the network
	perlayer_relative_size = calculate_perlayer_relative_size(model)
	average_layer_relative_size = sum(perlayer_relative_size) / len(perlayer_relative_size)

	average_optimal_bitlengths = None
	min_bitlengths = None
	max_bitlengths = None
	minmax_bitlengths = None

	total_num_batches = 0
	start_time = time.time()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		print("Finding a general bitlength for all layers")
		batch_optimal_bitlengths = bruteforce_crunch(data, target, model, criterion, args)

		if args.netcrunch_alg == "sa":
			print("Performing Simulated Annealing algorithm to find better solution")
			batch_optimal_bitlengths = simulated_annealing_netcrunch(data, target, model, criterion, batch_optimal_bitlengths, batch_idx, perlayer_relative_size, args)
		elif args.netcrunch_alg == "ga":
			print("Performing Genetic Algorithm to find better solution")
			batch_optimal_bitlengths = genetic_algorithm_netcrunch(data, target, model, criterion, batch_optimal_bitlengths, batch_idx, perlayer_relative_size, args)
		elif args.netcrunch_alg == "greedy_bf":
			print("Performing Greedy Biggest-First algorithm to find better solution")
			batch_optimal_bitlengths = greedy_bf_netcrunch(data, target, model, criterion, batch_optimal_bitlengths, batch_idx, perlayer_relative_size, args)
		else:
			print("Unrecognized algorithm. Available algorithms: sa, ga, greedy_bf. Exiting.")
			exit(1)

		# deepcopy the model to keep the original intact
		trimmed_model = copy.deepcopy(model)

		with torch.no_grad():
			# prepare the final trimmed version for this batch and evaluate it on the batch
			i = 0
			for name, parameter in trimmed_model.named_parameters():
				if 'bias' not in name:
					# prepare the mask to trim the least significant N bits of the mantissas
					mask = 0xFFFFFFFF
					mask = mask >> (args.init_bitlength - batch_optimal_bitlengths[i])
					mask = mask << (args.init_bitlength - batch_optimal_bitlengths[i])
					# view the float value as if it was an int, to modify the bits specifically
					weight_as_int = parameter.data.view(torch.int32) & mask

					# reconvert to its trimmed float version
					parameter.data = weight_as_int.data.view(torch.float)
					i += 1

			# compute output and loss of trimmed model
			trimmed_output = trimmed_model(data)       
			trimmed_loss = criterion(trimmed_output, target)

			prec1, prec5 = accuracy(trimmed_output.data, target.data, topk=(1, 5))
			losses.update(trimmed_loss.item(), data.size(0))
			top1.update(prec1.item(), data.size(0))
			top5.update(prec5.item(), data.size(0))

		# get rid of the trimmed model
		del trimmed_model

		# get the worst case scenario bitlength for every layer from previous batches
		if average_optimal_bitlengths is None:
			average_optimal_bitlengths = [x for x in batch_optimal_bitlengths]
		else:
			for i, avg_bitlength in enumerate(average_optimal_bitlengths):
				average_optimal_bitlengths[i] = avg_bitlength + batch_optimal_bitlengths[i]

		# min bitlength calculations
		if min_bitlengths is None:
			min_bitlengths = [x for x in batch_optimal_bitlengths]
		else:
			for i, min_bitlength in enumerate(min_bitlengths):
				min_bitlengths[i] = min(min_bitlength, batch_optimal_bitlengths[i])

		# max bitlength calculations
		if max_bitlengths is None:
			max_bitlengths = [x for x in batch_optimal_bitlengths]
		else:
			for i, max_bitlength in enumerate(max_bitlengths):
				max_bitlengths[i] = max(max_bitlength, batch_optimal_bitlengths[i])

		# min-max bitlength calculations, if layer size is greater than average, use min, otherwise max
		if minmax_bitlengths is None:
			minmax_bitlengths = [x for x in batch_optimal_bitlengths]
		else:
			for i, minmax_bitlength in enumerate(minmax_bitlengths):
				if perlayer_relative_size[i] > average_layer_relative_size:
					minmax_bitlengths[i] = min(minmax_bitlength, batch_optimal_bitlengths[i])
				else:
					minmax_bitlengths[i] = max(minmax_bitlength, batch_optimal_bitlengths[i])

		total_num_batches += 1

		steptime = 0
		steptime = progress_bar(batch_idx, len(train_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%%' % (losses.avg, top1.avg, top5.avg))
		stime.update(steptime, 1)

	for i, summed_bitlengths in enumerate(average_optimal_bitlengths):
		# average bitlengths calculation
		average_optimal_bitlengths[i] = round((summed_bitlengths / float(total_num_batches)) + 0.5)

	end_time = time.time()
	total_time = end_time - start_time
	time_string = " seconds"
	if total_time > 60:
		total_time = total_time / 60.0
		time_string = " minutes"

	return total_time, time_string, average_optimal_bitlengths, min_bitlengths, max_bitlengths, minmax_bitlengths, perlayer_relative_size