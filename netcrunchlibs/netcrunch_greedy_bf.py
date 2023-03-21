import torch
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
import random

from core.utils import *

def evaluate_solution_loss(original_model, data, target, criterion, bitlengths, args):
	
	# deepcopy the model to have a modifiable version
	trimmed_model = copy.deepcopy(original_model)

	i = 0
	for name, parameter in trimmed_model.named_parameters():
		if 'bias' not in name:
			# prepare the mask to trim the least significant N bits of the mantissas
			mask = 0xFFFFFFFF
			mask = mask >> (args.init_bitlength - bitlengths[i])
			mask = mask << (args.init_bitlength - bitlengths[i])

			# view the float value as if it was an int, to modify the bits specifically
			weight_as_int = parameter.data.view(torch.int32) & mask

			# reconvert to its trimmed float version
			parameter.data = weight_as_int.data.view(torch.float)
			i += 1

	with torch.no_grad():
		# compute output and loss of trimmed model
		trimmed_output = trimmed_model(data)       
		trimmed_loss = criterion(trimmed_output, target)
	
	# free memory from GPU
	del trimmed_model

	# return the trimmed loss item
	return trimmed_loss

def greedy_bf_netcrunch(data, target, original_model, criterion, starter_bitlengths, batch_idx, perlayer_weighted_size_distribution, args):

	# first compute output and loss of original model
	with torch.no_grad():
		output = original_model(data)       
		loss = criterion(output, target)
	# set current solution to starter bitlengths
	current_solution = starter_bitlengths
	# calculate how much each layer is contributing to the total footprint of the model
	current_solution_perlayer_contribution = [(bitlength + args.init_exp_bitlength + 1.0) * layer_weight for bitlength, layer_weight in zip(current_solution, perlayer_weighted_size_distribution)]
	# create a boolean list that will tell the algorithm whether a specific layer cannot be trimmed more or it will cause the loss to deteriorate too much
	current_solution_perlayer_saturated = [False for _ in range(len(current_solution))]

	iteration = 0
	while not all(current_solution_perlayer_saturated):
		print("Iteration " + str(iteration) + " for greedy biggest-first search")
		# find the layer that currently contributes more towards footprint and that is not saturated
		largest_layer_index = current_solution_perlayer_contribution.index(max(current_solution_perlayer_contribution))
		# decrease the bitlength by one
		current_solution[largest_layer_index] = max(0, current_solution[largest_layer_index] - 1)
		# evaluate the loss of the solution and calculate relative error to full precision model
		solution_loss = evaluate_solution_loss(original_model, data, target, criterion, current_solution, args)
		relative_error = abs(solution_loss.item() - loss.item()) / loss.item()
		# if the relative error goes over the maximum we have allowed, return the bitlength to its previous value and set the layer as saturated
		if relative_error > args.max_loss_deviation:
			current_solution[largest_layer_index] = min(args.init_bitlength, current_solution[largest_layer_index] + 1)
			current_solution_perlayer_saturated[largest_layer_index] = True
			current_solution_perlayer_contribution[largest_layer_index] = -1.0
		# update the contribution list
		current_solution_perlayer_contribution = [(bitlength + args.init_exp_bitlength + 1.0) * layer_weight for bitlength, layer_weight in zip(current_solution, perlayer_weighted_size_distribution)]
		# set saturated layers to symbolically not contribute footprint for the algorithm
		current_solution_perlayer_contribution = [-1.0 if current_solution_perlayer_saturated[i] else bitlength for i, bitlength in enumerate(current_solution)]
		# increase the iteration number
		iteration += 1

	print("Found solution for batch " + str(batch_idx) + " with bitlengths: " + str(current_solution))
	return current_solution