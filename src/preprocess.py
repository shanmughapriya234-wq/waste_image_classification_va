import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

DATA_DIR = "./dataset"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)

class_names = full_dataset.classes
print("Categories found:", class_names)
print("Total images:", len(full_dataset))

total = len(full_dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size


train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")


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
val_dataset = TransformSubset(val_set, val_test_transform)
test_dataset = TransformSubset(test_set, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Preprocessing complete!")
print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")