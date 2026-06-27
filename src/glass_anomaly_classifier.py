import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from model import build_binary_model



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


DATA_DIR = "./glass_anomaly_detection_dataset"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=None
)

print("Classes:", full_dataset.classes)
print("Class Mapping:", full_dataset.class_to_idx)


total = len(full_dataset)

train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


train_dataset = TransformSubset(train_set, train_transform)
val_dataset = TransformSubset(val_set, val_transform)

test_dataset = TransformSubset(test_set, val_transform)

print(
    f"Train: {len(train_set)} | "
    f"Val: {len(val_set)} | "
    f"Test: {len(test_set)}"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


model = build_binary_model()

# Fine-tune only layer4 for anomaly detection
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

EPOCHS = 50

best_val_acc = 0.0

os.makedirs(
    "saved_models",
    exist_ok=True
)


for epoch in range(EPOCHS):


    model.train()

    train_loss = 0
    train_correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

    train_acc = train_correct / len(train_set)


    model.eval()

    val_loss = 0
    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

    val_acc = val_correct / len(val_set)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "saved_models/glass_anomaly_detection_model.pth"
        )

        print(
            f"Best model saved! "
            f"Val Acc: {val_acc:.4f}"
        )

print("\nTraining Complete!")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)