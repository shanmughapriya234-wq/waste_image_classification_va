import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from model import build_binary_model

results_dir = "glass_plastic_saved_results"

if not os.path.exists(results_dir):

    os.makedirs(results_dir)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


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
    root="./dataset",
    transform=None
)


glass_idx = full_dataset.class_to_idx["Glass"]
plastic_idx = full_dataset.class_to_idx["Plastic"]
print("Classes: Glass, Plastic")

print("\nClass Mapping:")
print("0 -> Glass")
print("1 -> Plastic")

class GlassPlasticDataset(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform

        self.samples = []

        for path, label in dataset.samples:

            if label == glass_idx:
                self.samples.append((path, 0))

            elif label == plastic_idx:
                self.samples.append((path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = self.dataset.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


dataset = GlassPlasticDataset(full_dataset, transform=None)

print("Glass + Plastic Images:", len(dataset))


total = len(dataset)

train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

print(
    f"Train: {len(train_set)} | "
    f"Val: {len(val_set)} | "
    f"Test: {len(test_set)}"
)

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


train_dataset = TransformSubset(train_set, transform=train_transform)
val_dataset = TransformSubset(val_set, transform=val_transform)

test_dataset = TransformSubset(test_set, transform=val_transform)

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
model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


EPOCHS = 50

best_val_acc = 0.0
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


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
        train_loss += loss.item()

        loss.backward()

        optimizer.step()

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

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

    val_acc = val_correct / len(val_set)
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            os.path.join(results_dir, "glass_plastic_model.pth")
        )

        print(
            f"Best Model Saved! "
            f"Val Acc: {val_acc:.4f}"
        )
epochs = list(range(1, EPOCHS + 1))

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(
    epochs,
    train_losses,
    marker="o",
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="s",
    linewidth=2,
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Glass vs Plastic Classification Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1,2,2)
plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    linewidth=2,
    label="Train Accuracy"
)
plt.plot(
    epochs,
    val_accuracies,
    marker="s",
    linewidth=2,
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Glass vs Plastic Classification Accuracy")
plt.legend()
plt.grid(True)

plt.suptitle(
    "Glass vs Plastic Classification Training Results",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    os.path.join(results_dir, "glass_plastic_training_results.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
print("\nTraining Complete!")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)
with open(os.path.join(results_dir, "training_summary.txt"), "w") as f:
    f.write("Glass vs Plastic Binary Classification\n")
    f.write("=" * 40 + "\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write("Batch Size: 16\n")
    f.write("Learning Rate: 0.001\n")
    f.write(f"Training Images: {len(train_set)}\n")
    f.write(f"Validation Images: {len(val_set)}\n")
    f.write(f"Test Images: {len(test_set)}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")


with open(os.path.join(results_dir, "training_history_glass_plastic.csv"), "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Train Accuracy",
        "Validation Accuracy"
    ])

    for epoch in range(EPOCHS):
        writer.writerow([
            epoch + 1,
            train_losses[epoch],
            val_losses[epoch],
            train_accuracies[epoch],
            val_accuracies[epoch]
        ])