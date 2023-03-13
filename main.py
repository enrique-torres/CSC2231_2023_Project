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

from core.modelzoo import load_untrained_model
from core.utils import *
from core.train import *

from netcrunch import run_netcrunch

ssl._create_default_https_context = ssl._create_unverified_context

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

def main():
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
	
	timestr = time.strftime("%Y%m%d_%H%M%S")
	args.output_path = "results/" + args.model + "_cifar100_" + timestr 

	# run the core of NetCrunch on the train dataset
	crunched_model = run_netcrunch(train_loader, device, model, criterion, args)

	# test the compressed model to see the effect on accuracy versus baseline
	top1_original, top5_original = do_test(test_loader, device, model, criterion, args)
	top1_crunched, top5_crunched = do_test(test_loader, device, crunched_model, criterion, args)

	# print out the results
	print("The original model achieved a Top-1 accuracy of " + str(top1_original) + ", while the trimmed model achieved " + str(top1_crunched))
	print("The original model achieved a Top-5 accuracy of " + str(top5_original) + ", while the trimmed model achieved " + str(top5_crunched))

	

if __name__ == '__main__':
	main()