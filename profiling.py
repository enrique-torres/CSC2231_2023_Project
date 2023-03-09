import torch
import torch.optim as optim
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy

from core.utils import *

# Calculate, for a given batch, how much each layer contributes to updating the 
# weights/contributes to the updating of the entire network
def compute_layer_contribution_gradient_based(data, target, original_model, criterion, args):

	# Set the model to evaluation mode
	original_model.eval()
	optimizer = optim.SGD(original_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	# Forward pass
	outputs = original_model(data)
	loss = criterion(outputs, target)

	# Backward pass
	optimizer.zero_grad()
	loss.backward()

	# Calculate the contribution of each layer to the loss value
	layer_contributions = []
	for name, param in original_model.named_parameters():
		if 'bias' not in name:
			grad = param.grad.clone().detach()
			param.backward(grad)
			layer_contributions.append(grad.abs().mean().item())

	return layer_contributions

# Calculate, for a given batch, how much does trimming mantissa bits degrade
# the performance of the network (loss), its outputs and how much are the gradients modified for the batch
def compute_layer_contribution_degradation(data, target, original_model, criterion, args):

	# Set the model to evaluation mode
	original_model.eval()

	# Forward pass
	outputs = original_model(data)
	loss = criterion(outputs, target)

	# For every layer, we are going to drop its mantissa bitlength from initial bitlength to 0
	# and we will test what is the variance in the outputs/activations of the layer against the original model, 
	# as well as how much the loss degrades in general
	perlayer_outputs_degradation = {}
	perlayer_loss_degradation = {}
	for name_original, _ in original_model.named_parameters():

		if "bias" not in name_original:
			perlayer_outputs_degradation[name_original] = []
			perlayer_loss_degradation[name_original] = []

			# deepcopy the model to have a modifiable version
			trimmed_model = copy.deepcopy(original_model)
			trimmed_model.eval()

			# for every bitlength going down from initial bilength, trim the layer mantissa bitlength 1 by 1
			for bitlength in range(args.init_bitlength, -1, -1):
				print("Trying bitlength " + str(bitlength) + " for layer ID " + str(name_original))
				for name_trimmed, param_trimmed in trimmed_model.named_parameters():
					if name_original == name_trimmed and 'bias' not in name_trimmed:
						# prepare the mask to trim the least significant N bits of the mantissas
						mask = 0xFFFFFFFF
						mask = mask >> (args.init_bitlength - bitlength)
						mask = mask << (args.init_bitlength - bitlength)

						# view the float value as if it was an int, to modify the bits specifically
						weight_as_int = param_trimmed.data.view(torch.int32) & mask

						# reconvert to its trimmed float version
						param_trimmed.data = weight_as_int.data.view(torch.float32)

				# compute output and loss of trimmed model
				trimmed_output = trimmed_model(data)       
				trimmed_loss = criterion(trimmed_output, target)

				# calculate outputs average differential
				outputs_differential = (abs(outputs - trimmed_output) / outputs) / 100.0
				outputs_differential_mean = outputs_differential.mean().item()
				perlayer_outputs_degradation[name_original].append(outputs_differential_mean)

				# calculate loss degradation
				loss_degradation = (abs(loss.item() - trimmed_loss.item()) / loss.item()) / 100.0
				perlayer_loss_degradation[name_original].append(loss_degradation)

			# delete the trimmed model from GPU memory because we will have to recopy it for every layer
			del trimmed_model
		
	# return the mean outputs differential per layer per bitlength, as well as the loss degradation per layer per bitlength
	return perlayer_outputs_degradation, perlayer_loss_degradation

def calculate_perlayer_relative_size(original_model, args):

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

def run_profiling(train_loader, device, model, criterion, args):
	global_outputs_degradation = []
	global_loss_degradation = []
	
	# Set the model to evaluation mode
	model.eval()

	# calculate how many parameters are there per layer
	perlayer_parameters_weighted = calculate_perlayer_relative_size(model, args)

	stime = AverageMeter()
	
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		# evaluate per layer contribution to the loss
		perlayer_outputs_degradation, perlayer_loss_degradation = compute_layer_contribution_degradation(data, target, model, criterion, args)
		global_outputs_degradation.append(perlayer_outputs_degradation)
		global_loss_degradation.append(perlayer_loss_degradation)

		steptime = 0
		steptime = progress_bar(batch_idx, len(train_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%%' % (losses.avg, top1.avg, top5.avg))
		stime.update(steptime, 1)

	# calculate the average output differential per layer per bitlength across all batches
	averaged_output_degradations = None
	for batch_output_degradations in global_outputs_degradation:
		if averaged_output_degradations is None:
			averaged_output_degradations = batch_output_degradations
		else:
			for layer_name, degradations in batch_output_degradations:
				averaged_output_degradations[layer_name] = [sum(x) for x in zip(degradations, averaged_output_degradations[layer_name])]

	for layer_name, degradations in averaged_output_degradations:
		averaged_output_degradations[layer_name] = [x / float(len(global_outputs_degradation)) for x in degradations]

	# calculate the average loss degradation per layer per bitlength across all batches
	averaged_loss_degradations = None
	for batch_loss_degradations in global_loss_degradation:
		if averaged_loss_degradations is None:
			averaged_loss_degradations = batch_loss_degradations
		else:
			for layer_name, degradations in batch_loss_degradations:
				averaged_loss_degradations[layer_name] = [sum(x) for x in zip(degradations, averaged_loss_degradations[layer_name])]

	for layer_name, degradations in averaged_loss_degradations:
		averaged_loss_degradations[layer_name] = [x / float(len(global_loss_degradation)) for x in degradations]

	# return average output relative differences, relative loss degradations and number of parameters per layer
	return averaged_output_degradations, averaged_loss_degradations, perlayer_parameters_weighted