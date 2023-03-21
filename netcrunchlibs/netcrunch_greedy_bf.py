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






def greedy_bf_netcrunch(data, target, original_model, criterion, starter_bitlengths, batch_idx, perlayer_weighted_size_distribution, args):
    current_solution = starter_bitlengths
    print("Found solution for batch " + str(batch_idx) + " with bitlengths: " + str(current_solution))
    return current_solution