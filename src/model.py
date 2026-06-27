import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=9):
    """
    Build the 9-class waste classification model.
    """

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model


def build_binary_model():
    """
    Build a binary classification model.
    Used for:
    - Glass Anomaly (Broken vs Normal)
    - Glass vs Plastic
    """

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )

    return model