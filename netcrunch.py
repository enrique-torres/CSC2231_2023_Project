import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy

from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# NEW IDEA: Test every layer individually to see what effect does trimming the specific layer have on the training loss
# Then, based on layer size and effect on loss, decide what to trim more or less

def compute_layer_contribution(data, target, original_model, criterion, args):

	# Set the model to evaluation mode
	original_model.eval()

	# Split the model into its constituent layers
	layers = list(original_model.children())

	# Compute the contribution of each layer to the loss
	layer_contributions = []
	with torch.no_grad():
		# Forward pass through the model
		output = original_model(data)

		# Compute the gradients of the loss with respect to the output of the model
		loss_grads = torch.autograd.grad(criterion(output, target), output)[0]

		# Compute the contribution of each layer to the loss
		grads = None
		for i in range(len(layers) - 1, -1, -1):
			if grads is None:
				# The gradient of the loss with respect to the output of the model is given by the loss_grads tensor
				grads = loss_grads
			else:
				# Compute the gradient of the loss with respect to the input to the layer
				grads = torch.autograd.grad(None, layers[i].weight, grads, retain_graph=True)[0]
			contributions = layers[i].weight * grads
			contributions = torch.mean(contributions.view(contributions.size(0), -1), dim=1)
			layer_contributions.append(contributions)

def calculate_perlayer_parameters(original_model, args):

	# Split the model into its constituent layers
	layers = list(original_model.children())
	# Compute the metric for each layer
	perlayer_parameters = []
	for i in range(len(layers)):
		num_params = sum(p.numel() for p in layers[i].parameters())
		perlayer_parameters.append(num_params)

	return perlayer_parameters


# First method to compare against. This method brings down the bitlength for all layers until a maximum
# relative error is reached between the original model's loss on the batch of data, and the loss of the
# trimmed model (by default the limit is 10% deviation from original loss)
def bruteforce_crunch(data, target, original_model, criterion, args):
	
	# create initial list of mantissa bitlengths for the trimmed network
	optimal_bitlengths = [args.init_bitlength] * len(list(original_model.parameters()))
	
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
			for i, parameter in enumerate(trimmed_model.parameters()):
				# prepare the mask to trim the least significant N bits of the mantissas
				mask = 0xFFFFFFFF
				mask = mask >> (args.init_bitlength - optimal_bitlengths[i])
				mask = mask << (args.init_bitlength - optimal_bitlengths[i])

				# view the float value as if it was an int, to modify the bits specifically
				weight_as_int = parameter.data.view(torch.int32) & mask

				# reconvert to its trimmed float version
				parameter.data = weight_as_int.data.view(torch.float32)

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


def run_netcrunch(train_loader, device, model, criterion, args):

	model.train()

	losses  = AverageMeter()
	top1    = AverageMeter()
	top5    = AverageMeter()
	stime   = AverageMeter()

	global_optimal_bitlengths = None
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		batch_optimal_bitlengths = bruteforce_crunch(data, target, model, criterion, args)

		# deepcopy the model to keep the original intact
		trimmed_model = copy.deepcopy(model)

		with torch.no_grad():
			# prepare the final trimmed version for this batch and evaluate it on the batch
			for i, parameter in enumerate(trimmed_model.parameters()):
				# prepare the mask to trim the least significant N bits of the mantissas
				mask = 0xFFFFFFFF
				mask = mask >> batch_optimal_bitlengths[i]
				mask = mask << batch_optimal_bitlengths[i]
				# view the float value as if it was an int, to modify the bits specifically
				weight_as_int = parameter.data.view(torch.int32) & mask

				# reconvert to its trimmed float version
				parameter.data = weight_as_int.data.view(torch.float)

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
		if global_optimal_bitlengths is None:
			global_optimal_bitlengths = batch_optimal_bitlengths
		else:
			for i, bitlength in enumerate(global_optimal_bitlengths):
				global_optimal_bitlengths[i] = max(bitlength, batch_optimal_bitlengths[i])

		steptime = 0
		steptime = progress_bar(batch_idx, len(train_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%%' % (losses.avg, top1.avg, top5.avg))
		stime.update(steptime, 1)

def do_test(test_loader, device, model, criterion, args):

	model.eval()

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()    

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = criterion(output, target)
			prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

			losses.update(loss.item(), data.size(0))
			top1.update(prec1.item(), data.size(0))
			top5.update(prec5.item(), data.size(0))

			progress_bar(batch_idx, len(test_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%% ' % (losses.avg, top1.avg, top5.avg))


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='NetCrunch: Finding optimal inference bitlengths heuristically for neural networks')
	parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
	parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')    
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')    
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--model', type=str, default="resnet18", help='Model name to run training on (default: resnet18)')
	parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0")')

	parser.add_argument('--run-netcrunch', action='store_true', default=False, help='Runs NetCrunch on the network')
	parser.add_argument('--pretrained-path', type=str, default=None, help='The path to the model pretrained on CIFAR100')
	parser.add_argument('--max-loss-deviation', type=float, default=0.1, help='Maximum accepted percentage deviation of the trimmed training loss versus the original loss')
	parser.add_argument('--init-bitlength', type=int, default=23, help='Initial mantissa bitlength. Set based on datatype of pretrained network (default: 23 for FP32)') 


	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda:"+args.device if use_cuda else "cpu")
	print("Training on: "+str(device))

	print("=> creating model '{}'".format(args.model))
	model = load_untrained_model(args.model, 100)
	if args.pretrained_path is not None:
		checkpoint = torch.load(args.pretrained_path)
		model.load_state_dict(checkpoint['state_dict'])
		last_trained_epoch = checkpoint['epoch']
		print("Training from epoch " + str(last_trained_epoch))
	else:
		print("Pretrained model is required for NetCrunch to work!")
		exit(1)
	model = model.to(device)

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
	])

	trainset = datasets.CIFAR100(
		root='./cifar100data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(
		trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

	testset = datasets.CIFAR100(
		root='./cifar100data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(
		testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	
	timestr = time.strftime("%Y%m%d_%H%M%S")
	args.output_path = "results/" + args.model + "_cifar100_" + timestr 

	# run the core of NetCrunch on the train dataset
	crunched_model = run_netcrunch(train_loader, device, model, criterion, args)

	# test the compressed model to see the effect on accuracy versus baseline
	do_test(test_loader, device, crunched_model, criterion, args)

if __name__ == '__main__':
	main()