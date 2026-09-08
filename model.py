import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import config


def build_model(freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        # CPU training is slow, so freeze everything except the last residual
        # block (layer4) and the classifier head. 
        # saves most of the compute without giving up much accuracy.
        for name, param in model.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)
    return model
