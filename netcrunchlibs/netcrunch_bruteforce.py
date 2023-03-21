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

# First method to compare against. This method brings down the bitlength for all layers until a maximum
# relative error is reached between the original model's loss on the batch of data, and the loss of the
# trimmed model (by default the limit is 10% deviation from original loss)
def bruteforce_crunch(data, target, original_model, criterion, args):
	
	# create initial list of mantissa bitlengths for the trimmed network
	optimal_bitlengths = []
	for name, param in original_model.named_parameters():
		if 'bias' not in name:
			optimal_bitlengths.append(args.init_bitlength)
	
	with torch.no_grad():
		# compute output of original model
		output = original_model(data)       
		loss = criterion(output, target)

	# deepcopy the model to have a modifiable version
	trimmed_model = copy.deepcopy(original_model)
	
	with torch.no_grad():
		# compute output and loss of trimmed model
		trimmed_output = trimmed_model(data)       
		trimmed_loss = criterion(trimmed_output, target)

	relative_error = abs(trimmed_loss.item() - loss.item()) / loss.item()
	
	while not (trimmed_loss.item() > loss.item() and relative_error > args.max_loss_deviation):
		#print("Original loss was: " + str(loss.item()) + ", Trimmed loss is: " + str(trimmed_loss.item()) + ", Relative error is: " + str(relative_error))
		#print("Trying the following bitlengths: " + str(optimal_bitlengths))
		# we don't want torch autograd to know about this :)
		with torch.no_grad():
			# for every parameter, take its currently defined mantissa bitlength
			i = 0
			for name, parameter in trimmed_model.named_parameters():
				if 'bias' not in name:
					# prepare the mask to trim the least significant N bits of the mantissas
					mask = 0xFFFFFFFF
					mask = mask >> (args.init_bitlength - optimal_bitlengths[i])
					mask = mask << (args.init_bitlength - optimal_bitlengths[i])

					# view the float value as if it was an int, to modify the bits specifically
					weight_as_int = parameter.data.view(torch.int32) & mask

					# reconvert to its trimmed float version
					parameter.data = weight_as_int.data.view(torch.float32)
					i += 1

			# push all the bitlengths down
			for i, bitlength in enumerate(optimal_bitlengths):
				optimal_bitlengths[i] = max(bitlength - 1, 0)

			# compute output and loss of trimmed model, then compute relative error between the original and the trimmed loss
			trimmed_output = trimmed_model(data)       
			trimmed_loss = criterion(trimmed_output, target)
			relative_error = abs(trimmed_loss.item() - loss.item()) / loss.item()

	# delete the trimmed model from GPU memory, as we are constantly creating new versions of the model
	del trimmed_model

	# deepcopy the model to have a modifiable version
	trimmed_model = copy.deepcopy(original_model)

	# return to the previous state of bitlengths, as we hadn't surpassed the relative error limit
	for i, bitlength in enumerate(optimal_bitlengths):
		optimal_bitlengths[i] = min(bitlength + 1, args.init_bitlength)

	print("Found bitlengths within relative loss error: " + str(optimal_bitlengths))
	
	return optimal_bitlengths