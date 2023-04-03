import argparse
import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import ssl
import csv

from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *

from braincrunchlibs.braincrunch import run_braincrunch

ssl._create_default_https_context = ssl._create_unverified_context

def prepare_trimmed_model(original_model, bitlengths, args):
	with torch.no_grad():
		# deepcopy the model to keep the original intact
		trimmed_model = copy.deepcopy(original_model)
		# prepare the final trimmed version for this batch and evaluate it on the batch
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
	return trimmed_model

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
	return top1.avg, top5.avg

def run_experiment(args):
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
		print("Pretrained model is required for BrainCrunch to work!")
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
	
	timestr = time.strftime("%Y%m%d_%H%M%S")
	args.output_path = "results/" + args.model + "_cifar100_" + timestr + "_" + args.braincrunch_alg + ".txt"

	# run the core of BrainCrunch on the train dataset
	total_time, time_string, average_optimal_bitlengths, min_bitlengths, max_bitlengths, minmax_bitlengths, perlayer_relative_size, exponent_bitlengths = run_braincrunch(train_loader, device, model, criterion, args)
	print("DEBUG EXPONENT BITLENGTHS #################################################################")
	print(exponent_bitlengths)

	# test the compressed model to see the effect on accuracy versus baseline
	top1_original, top5_original = do_test(test_loader, device, model, criterion, args)
	# test with average bitlengths
	trimmed_model_average = prepare_trimmed_model(model, average_optimal_bitlengths, args)
	top1_crunched_avg, top5_crunched_avg = do_test(test_loader, device, trimmed_model_average, criterion, args)
	del trimmed_model_average
	# test with min bitlengths
	trimmed_model_min = prepare_trimmed_model(model, min_bitlengths, args)
	top1_crunched_min, top5_crunched_min = do_test(test_loader, device, trimmed_model_min, criterion, args)
	del trimmed_model_min
	# test with max bitlengths
	trimmed_model_max = prepare_trimmed_model(model, max_bitlengths, args)
	top1_crunched_max, top5_crunched_max = do_test(test_loader, device, trimmed_model_max, criterion, args)
	del trimmed_model_max
	# test with minmax bitlengths
	trimmed_model_minmax = prepare_trimmed_model(model, minmax_bitlengths, args)
	top1_crunched_minmax, top5_crunched_minmax = do_test(test_loader, device, trimmed_model_minmax, criterion, args)
	del trimmed_model_minmax

	# calculate footprint of all mixing methods
	footprint_baseline = 0.0
	footprint_average_method = 0.0
	footprint_min_method = 0.0
	footprint_max_method = 0.0
	footprint_minmax_method = 0.0
	# calculate footprint of mantissas
	footprint_baseline_mantissas = 0.0
	footprint_average_method_mantissas = 0.0
	footprint_min_method_mantissas = 0.0
	footprint_max_method_mantissas = 0.0
	footprint_minmax_method_mantissas = 0.0
	# calculate footprint of exponents
	footprint_baseline_exponents = 0.0
	footprint_average_method_exponents = 0.0
	footprint_min_method_exponents = 0.0
	footprint_max_method_exponents = 0.0
	footprint_minmax_method_exponents = 0.0
	# calculate footprint of sign
	footprint_baseline_sign = 0.0
	footprint_average_method_sign = 0.0
	footprint_min_method_sign = 0.0
	footprint_max_method_sign = 0.0
	footprint_minmax_method_sign = 0.0

	for i, layer_size in enumerate(perlayer_relative_size):
		# general footprint
		footprint_baseline += (args.init_bitlength + args.init_exp_bitlength + 1.0) * layer_size # 1.0 for sign bit
		footprint_average_method += (average_optimal_bitlengths[i] + exponent_bitlengths[i] + 1.0) * layer_size
		footprint_min_method += (min_bitlengths[i] + exponent_bitlengths[i] + 1.0) * layer_size
		footprint_max_method += (max_bitlengths[i] + exponent_bitlengths[i] + 1.0) * layer_size
		footprint_minmax_method += (minmax_bitlengths[i] + exponent_bitlengths[i] + 1.0) * layer_size
		# mantissas footprint contribution
		footprint_baseline_mantissas += (args.init_bitlength) * layer_size # 1.0 for sign bit
		footprint_average_method_mantissas += (average_optimal_bitlengths[i]) * layer_size
		footprint_min_method_mantissas += (min_bitlengths[i]) * layer_size
		footprint_max_method_mantissas += (max_bitlengths[i]) * layer_size
		footprint_minmax_method_mantissas += (minmax_bitlengths[i]) * layer_size
		# exponents footprint contribution
		footprint_baseline_exponents += (args.init_exp_bitlength) * layer_size # 1.0 for sign bit
		footprint_average_method_exponents += (exponent_bitlengths[i]) * layer_size
		footprint_min_method_exponents += (exponent_bitlengths[i]) * layer_size
		footprint_max_method_exponents += (exponent_bitlengths[i]) * layer_size
		footprint_minmax_method_exponents += (exponent_bitlengths[i]) * layer_size
		# sign footprint contribution
		footprint_baseline_sign += (1.0) * layer_size # 1.0 for sign bit
		footprint_average_method_sign += (1.0) * layer_size
		footprint_min_method_sign += (1.0) * layer_size
		footprint_max_method_sign += (1.0) * layer_size
		footprint_minmax_method_sign += (1.0) * layer_size

	# total footprint versus baseline
	footprint_average_method = (footprint_average_method / footprint_baseline)
	footprint_min_method = (footprint_min_method / footprint_baseline)
	footprint_max_method = (footprint_max_method / footprint_baseline)
	footprint_minmax_method = (footprint_minmax_method / footprint_baseline)
	# mantissas footprint versus baseline
	footprint_average_method_mantissas = (footprint_average_method_mantissas / footprint_baseline_mantissas)
	footprint_min_method_mantissas = (footprint_min_method_mantissas / footprint_baseline_mantissas)
	footprint_max_method_mantissas = (footprint_max_method_mantissas / footprint_baseline_mantissas)
	footprint_minmax_method_mantissas = (footprint_minmax_method_mantissas / footprint_baseline_mantissas)
	# exponents footprint versus baseline
	footprint_average_method_exponents = (footprint_average_method_exponents / footprint_baseline_exponents)
	footprint_min_method_exponents = (footprint_min_method_exponents / footprint_baseline_exponents)
	footprint_max_method_exponents = (footprint_max_method_exponents / footprint_baseline_exponents)
	footprint_minmax_method_exponents = (footprint_minmax_method_exponents / footprint_baseline_exponents)
	# sign footprint versus baseline
	footprint_average_method_sign = (footprint_average_method_sign / footprint_baseline_sign)
	footprint_min_method_sign = (footprint_min_method_sign / footprint_baseline_sign)
	footprint_max_method_sign = (footprint_max_method_sign / footprint_baseline_sign)
	footprint_minmax_method_sign = (footprint_minmax_method_sign / footprint_baseline_sign)


	with open(args.output_path, mode='w', newline='', encoding='utf-8') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',')
		csvwriter.writerow(["mixing_algorithm", "top1", "top5", "total_footprint", "mantissas_footprint", "exponents_footprint", "sign_footprint", "time"])
		csvwriter.writerow(["baseline", top1_original, top5_original, 1, 0.71875, 0.25, 0.03125])
		csvwriter.writerow(["average", top1_crunched_avg, top5_crunched_avg, footprint_average_method, footprint_average_method_mantissas, footprint_average_method_exponents, footprint_average_method_sign, str(total_time) + time_string])
		csvwriter.writerow(["min", top1_crunched_min, top5_crunched_min, footprint_min_method, footprint_min_method_mantissas, footprint_min_method_exponents, footprint_min_method_sign, str(total_time) + time_string])
		csvwriter.writerow(["max", top1_crunched_max, top5_crunched_max, footprint_max_method, footprint_max_method_mantissas, footprint_max_method_exponents, footprint_max_method_sign, str(total_time) + time_string])
		csvwriter.writerow(["minmax", top1_crunched_minmax, top5_crunched_minmax, footprint_minmax_method, footprint_minmax_method_mantissas, footprint_minmax_method_exponents, footprint_minmax_method_sign, str(total_time) + time_string])

	# print out the results
	#print("The original model achieved a Top-1 accuracy of " + str(top1_original) + " and Top-5 of " + str(top5_original))
	#print("The average trimmed model achieved a Top-1 accuracy of " + str(top1_crunched_avg) + " and Top-5 of " + str(top5_crunched_avg) + " with footprint vs baseline of: " + str(footprint_average_method) + "%")
	#print("Bitlengths for average method: " + str(average_optimal_bitlengths))
	#print("The min trimmed model achieved a Top-1 accuracy of " + str(top1_crunched_min) + " and Top-5 of " + str(top5_crunched_min) + " with footprint vs baseline of: " + str(footprint_min_method) + "%")
	#print("Bitlengths for min method: " + str(min_bitlengths))
	#print("The max trimmed model achieved a Top-1 accuracy of " + str(top1_crunched_max) + " and Top-5 of " + str(top5_crunched_max) + " with footprint vs baseline of: " + str(footprint_max_method) + "%")
	#print("Bitlengths for max method: " + str(max_bitlengths))
	#print("The min-max trimmed model achieved a Top-1 accuracy of " + str(top1_crunched_minmax) + " and Top-5 of " + str(top5_crunched_minmax) + " with footprint vs baseline of: " + str(footprint_minmax_method) + "%")
	#print("Bitlengths for min-max method: " + str(minmax_bitlengths))
	#print("Done! It took " + str(total_time) + time_string)
	#print("Optimal bitlengths found: " + str(mixing_algorithms_bitlengths[max_accuracy_index]) + " with mixing algorithm " + best_mixing_algorithm)


def main():
	parser = argparse.ArgumentParser(description='BrainCrunch: Finding optimal inference bitlengths heuristically for neural networks')
	parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
	parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')    
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')    
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--model', type=str, default="resnet18", help='Model name to run training on (default: resnet18)')
	parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0")')

	parser.add_argument('--run-braincrunch', action='store_true', default=False, help='Runs BrainCrunch on the network')
	parser.add_argument('--braincrunch-alg', type=str, default="sa", help='The algorithm that BrainCrunch will run: [sa, ga, greedy_bf, all]')
	parser.add_argument('--pretrained-path', type=str, default=None, help='The path to the model pretrained on CIFAR100')
	parser.add_argument('--max-loss-deviation', type=float, default=0.1, help='Maximum accepted percentage deviation of the trimmed training loss versus the original loss')
	parser.add_argument('--init-bitlength', type=int, default=23, help='Initial mantissa bitlength. Set based on datatype of pretrained network (default: 23 for FP32)')
	parser.add_argument('--init-exp-bitlength', type=int, default=8, help='Initial exponent bitlength. Set based on datatype of pretrained network (default: 8 for FP32)')


	args = parser.parse_args()
	if args.braincrunch_alg == "all":
		for algorithm in ["sa", "ga", "greedy_bf"]:
			args.braincrunch_alg = algorithm
			run_experiment(args)
	else:
		run_experiment(args)
	

if __name__ == '__main__':
	main()