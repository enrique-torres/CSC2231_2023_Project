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

def bitlength_weighted_neighbour_function(bitlengths, max_bitlength, perlayer_parameters_weighted):
	new_bitlengths = []
	for i, bitlength in enumerate(bitlengths):
		list_of_candidates = [max(0, bitlength - 1), bitlength, min(max_bitlength, bitlength + 1)]
		probability_of_candidates = [(1.0 - perlayer_parameters_weighted[i]) / 2.0, (1.0 - perlayer_parameters_weighted[i]) / 2.0, perlayer_parameters_weighted[i]]
		new_bitlength = np.random.choice(list_of_candidates, 1, p=probability_of_candidates).item()
		new_bitlengths.append(new_bitlength)
	return new_bitlengths

def bitlength_neighbour_function(bitlengths, max_bitlength):
	new_bitlengths = []
	for bitlength in bitlengths:
		new_bitlength = random.choice([max(0, bitlength - 1), bitlength, min(max_bitlength, bitlength + 1)])
		new_bitlengths.append(new_bitlength)
	return new_bitlengths

def exponential_scheduling(k=1, lam=0.1, limit=100):
	function = lambda t: (k * np.exp(-lam*t) if t < limit else 0)
	return function

def calculate_solution_cost(perlayer_weighted_size_distribution, solution_bitlengths, loss_relative_error, args):
	total_cost = 0
	for i, bitlength in enumerate(solution_bitlengths):
		layer_cost = (bitlength ** 2) * perlayer_weighted_size_distribution[i]
		total_cost += layer_cost
	if loss_relative_error > args.max_loss_deviation:
		total_cost = float("inf")#total_cost * 2.0 # we don't want to deviate too much because it will affect accuracy, so we penalize it
	else:
		total_cost = total_cost * (loss_relative_error)
	return total_cost

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


def simulated_annealing_netcrunch(data, target, original_model, criterion, starter_bitlengths, batch_idx, perlayer_weighted_size_distribution, args):
	
	# simulated annealing hyperparameters
	num_iterations = 10
	k = 1
	lam = 0.1
	
	with torch.no_grad():
		# compute output of original model
		output = original_model(data)       
		loss = criterion(output, target)

	# evaluate initial solution (all bitlengths set to a minimum within a relative loss error)
	trimmed_loss = evaluate_solution_loss(original_model, data, target, criterion, starter_bitlengths, args)

	# relative error between the original loss and the trimmed loss
	relative_error = abs(trimmed_loss.item() - loss.item()) / loss.item()

	# calculate the cost of the starter solution
	cost_starter_solution = calculate_solution_cost(perlayer_weighted_size_distribution, starter_bitlengths, relative_error, args)

	# start SA algorithm
	current_solution = starter_bitlengths
	for iter in range(num_iterations):
		# start of iteration
		print("Iteration " + str(iter) + " of SA algorithm for batch " + str(batch_idx))
		# define exponential scheduling for the iteration
		T = exponential_scheduling(k, lam, num_iterations)(iter)
		# calculate the new neighbour
		next_solution = bitlength_neighbour_function(current_solution, args.init_bitlength)
		#next_solution = bitlength_weighted_neighbour_function(current_solution, args.init_bitlength, perlayer_weighted_size_distribution)
		# evaluate the solution and get the loss
		trimmed_loss = evaluate_solution_loss(original_model, data, target, criterion, next_solution, args)
		# relative error between the original loss and the trimmed loss
		relative_error = abs(trimmed_loss.item() - loss.item()) / loss.item()
		# calculate the cost of the trimmed solution
		cost_new_solution = calculate_solution_cost(perlayer_weighted_size_distribution, next_solution, relative_error, args)
		# calculate the error delta between original solution and new solution
		delta_error = cost_new_solution - cost_starter_solution
		# check if delta error is better and stochastically decide about new solution
		if delta_error < 0 or np.exp(-1 * delta_error / T) > random.uniform(0.0, 1.0):
			current_solution = next_solution
	
	print("Found solution for batch " + str(batch_idx) + " with bitlengths: " + str(current_solution))
	return current_solution