'''
Modelzoo.py: exposes the API for a user to load a supported NN model

'''
import numpy as np
from core.models import *



supported_models_untrained = {
    "densenet121" : densenet.DenseNet121,
    "googlenet" : googlenet.GoogLeNet,
    "mobilenetv2" : mobilenetv2.MobileNetV2,
    "resnet18": resnet.ResNet18,
    "resnet34": resnet.ResNet34,
    "resnet50": resnet.ResNet50,
    "resnet101": resnet.ResNet101,
    "senet18" : senet.SENet18,
}


def load_untrained_model(model_name: str, num_classes=10):
    '''
    This function will return an untrained known neural network architecture
    '''
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string.")

    model_map = supported_models_untrained
    if model_name not in model_map:
        return None
    return model_map[model_name](num_classes=num_classes)
