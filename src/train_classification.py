import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import os
import csv

from model import build_model


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


DATA_DIR = "./dataset"
results_dir = "classification_saved_results"
os.makedirs(results_dir, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
print("Classes:", full_dataset.classes)

print("\nClass Mapping:")
for idx, class_name in enumerate(full_dataset.classes):
    print(f"{idx} -> {class_name}")
total = len(full_dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


model = build_model(9)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


EPOCHS = 50
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
   
    model.train()
    train_loss, train_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

  
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

   
    train_acc = train_correct / len(train_set)
    val_acc = val_correct / len(val_set)
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(results_dir, "classification_model.pth"))
        print(f"Best model saved with Val Acc: {val_acc:.4f}")


plt.figure(figsize=(12, 4))

epochs = range(1, EPOCHS + 1)

plt.subplot(1, 2, 1)

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

plt.title("Waste Image Classification Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

plt.plot(
    epochs,
    train_accs,
    marker="o",
    linewidth=2,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accs,
    marker="s",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Waste Image Classification Accuracy")
plt.legend()
plt.grid(True)
plt.suptitle(
    "Waste Classification Training Results",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    os.path.join(results_dir, "classification_training_results.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
print("Training complete!")

with open(os.path.join(results_dir, "training_summary.txt"), "w") as f:
    f.write("Waste Classification (9 Classes)\n")
    f.write("=" * 40 + "\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write("Batch Size: 32\n")
    f.write("Learning Rate: 0.001\n")
    f.write(f"Training Images: {len(train_set)}\n")
    f.write(f"Validation Images: {len(val_set)}\n")
    f.write(f"Test Images: {len(test_set)}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

with open(os.path.join(results_dir, "training_history_classification.csv"), "w", newline="") as file:

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
            train_accs[epoch],
            val_accs[epoch]
        ])