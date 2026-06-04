import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=9):
    # Load ResNet50 with ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer with our 9 category classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model

if __name__ == "__main__":
    model = build_model(num_classes=9)
    print(model)
    print("\nModel built successfully!")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    print("Model moved to GPU!")