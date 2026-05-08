import torch
import torchvision
import torch.nn as nn

def efficientnet():

    model = torchvision.models.efficientnet_b0(weights="DEFAULT")

    # freeze
    for param in model.parameters():
        param.requires_grad = False

    # fine-tuning classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    return model