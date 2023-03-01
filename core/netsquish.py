import torch
import copy

def find_optimal_mantissa_bitlength(model, test_loader, num_iterations=10, initial_bitlength=32, bitlength_factor=0.9, 
                                     max_bitlength=32, min_bitlength=1, tolerance=0.001):
    optimal_bitlengths = [initial_bitlength] * len(list(model.parameters()))
    current_bitlengths = copy.deepcopy(optimal_bitlengths)

    for iteration in range(num_iterations):
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.requires_grad:
                    dtype = param.dtype
                    scale = 2 ** (current_bitlengths[i] - 1)
                    param.data = torch.round(param.data * scale) / scale
                    param.data = param.data.to(dtype)

        layer_outputs = [[] for _ in range(len(list(model.parameters())))]
        for data in test_loader:
            inputs, _ = data
            with torch.no_grad():
                outputs = inputs
                for i, layer in enumerate(model):
                    outputs = layer(outputs)
                    layer_outputs[i].append(outputs)

        layer_means = [torch.mean(torch.cat(outputs)) for outputs in layer_outputs]
        layer_stddevs = [torch.std(torch.cat(outputs)) for outputs in layer_outputs]

        for i in range(len(list(model.parameters()))):
            old_bitlength = current_bitlengths[i]
            if layer_stddevs[i] > 0:
                new_bitlength = max(min(int(old_bitlength - 2 * torch.log10(layer_stddevs[i]).item()), max_bitlength), min_bitlength)
            else:
                new_bitlength = old_bitlength
            current_bitlengths[i] = round(bitlength_factor * current_bitlengths[i] + (1 - bitlength_factor) * new_bitlength)
            current_bitlengths[i] = max(min(current_bitlengths[i], max_bitlength), min_bitlength)

        if all(abs((current_mean - old_mean) / old_mean) < tolerance for current_mean, old_mean in zip(layer_means, optimal_layer_means)):
            break
        else:
            optimal_layer_means = layer_means
            optimal_bitlengths = current_bitlengths

    return optimal_bitlengths