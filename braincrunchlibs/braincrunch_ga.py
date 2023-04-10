import torch
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
import random
from heapq import nsmallest

from core.utils import *

# random mutate function but giving more probability to lowering bitlengths
def bitlength_weighted_mutate_function(bitlengths, max_bitlength, perlayer_parameters_weighted):
	new_bitlengths = []
	for i, bitlength in enumerate(bitlengths):
		list_of_candidates = [max(0, bitlength - 1), bitlength, min(max_bitlength, bitlength + 1)]
		probability_of_candidates = [(1.0 - perlayer_parameters_weighted[i]) / 2.0, (1.0 - perlayer_parameters_weighted[i]) / 2.0, perlayer_parameters_weighted[i]]
		new_bitlength = np.random.choice(list_of_candidates, 1, p=probability_of_candidates).item()
		new_bitlengths.append(new_bitlength)
	return new_bitlengths

# random mutate function
def bitlength_mutate_function(bitlengths, max_bitlength):
	new_bitlengths = []
	for bitlength in bitlengths:
		new_bitlength = random.choice([max(0, bitlength - 1), bitlength, min(max_bitlength, bitlength + 1)])
		new_bitlengths.append(new_bitlength)
	return new_bitlengths

# Order one crossover implementation
def crossover_bitlengths(solution_a, solution_b):
	try:
		sol_length = len(solution_a)
		child = list()
		start_segment = random.randint(0, sol_length // 2)
		end_segment = random.randint(sol_length // 2 + 1, sol_length - 1) + 1
		child.extend(solution_b[start_segment : end_segment])
		child.extend(solution_a[:start_segment])
		child.extend(solution_a[end_segment:])
		return child
	except Exception as ex:
		print(solution_a)
		print(solution_b)
		print(str(ex))
		exit(1)

def initialize_bitlength_population(num_population, starter_bitlengths):
	population = []
	for _ in range(num_population):
		population.append(starter_bitlengths)
	return population

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

def calculate_population_costs(perlayer_weighted_size_distribution, population_bitlengths, loss_relative_errors, args):
	population_costs = []
	for i, bitlengths in enumerate(population_bitlengths):
		sol_cost = calculate_solution_cost(perlayer_weighted_size_distribution, bitlengths, loss_relative_errors[i], args)
		population_costs.append(sol_cost)
	return population_costs

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

def evaluate_population_losses(original_model, data, target, criterion, population, args):
	population_losses = []
	for bitlengths in population:
		sol_loss = evaluate_solution_loss(original_model, data, target, criterion, bitlengths, args)
		population_losses.append(sol_loss)
	return population_losses

def genetic_algorithm_crunch(data, target, original_model, criterion, starter_bitlengths, batch_idx, perlayer_weighted_size_distribution, args):
	
	# simulated annealing hyperparameters
	num_iterations = 50 # test: 10, 20, 50
	num_population = 8
	num_parents = 4
	mutation_probability = 0.6 # test: 0.2, 0.6
	crossover_probability = 0.6 # test: 0.2, 0.6
	
	with torch.no_grad():
		# compute output of original model
		output = original_model(data)       
		loss = criterion(output, target)

	# start GA algorithm
	population = initialize_bitlength_population(num_population, starter_bitlengths)
	for iter in range(num_iterations):
		# start of iteration
		print("Iteration " + str(iter) + " of GA algorithm for batch " + str(batch_idx))
		# calculate the costs of all the constituents of the population
		population_losses = evaluate_population_losses(original_model, data, target, criterion, population, args)
		population_relative_errors = [abs(trimmed_loss.item() - loss.item()) / loss.item() for trimmed_loss in population_losses]
		population_costs = calculate_population_costs(perlayer_weighted_size_distribution, population, population_relative_errors, args)
		# order the population based on the cost, from smallest to highest
		parents = nsmallest(num_parents, population, key=lambda x: population_costs[population.index(x)])
		# generate the offspring and crossover between parents
		offspring = []
		new_population = []
		for p1, p2 in zip(parents[:len(parents)//2],parents[len(parents)//2:]):
			# Crossover probability
			if random.random() < crossover_probability:
				offspring.append(crossover_bitlengths(p1,p2))
				offspring.append(crossover_bitlengths(p2,p1))
			else:
				offspring.append(p1)
				offspring.append(p2)
		# mutate the offspring to explore the search space
		for child in offspring:
			if random.random() < mutation_probability:
				new_population.append(bitlength_mutate_function(child, args.init_bitlength))
			else:
				new_population.append(child)
		new_population.extend(parents)
		population = new_population

	# calculate the costs of all the constituents of the population
	population_losses = evaluate_population_losses(original_model, data, target, criterion, population, args)
	population_relative_errors = [abs(trimmed_loss.item() - loss.item()) / loss.item() for trimmed_loss in population_losses]
	population_costs = calculate_population_costs(perlayer_weighted_size_distribution, population, population_relative_errors, args)
	ordered_solutions = nsmallest(num_parents, population, key=lambda x: population_costs[population.index(x)])
	
	print("Found solution for batch " + str(batch_idx) + " with bitlengths: " + str(ordered_solutions[0]))
	return ordered_solutions[0]