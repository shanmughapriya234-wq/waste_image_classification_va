import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from model import build_model, build_binary_model


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using:", device)


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


class BinarySubsetDataset(Dataset):
    def __init__(self, dataset, label_map, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.label_map = label_map
        self.samples = []

        for path, label in dataset.samples:
            if label in label_map:
                self.samples.append((path, label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def make_test_loader(mode, transform):
    if mode == "waste_classification":
        full_dataset = datasets.ImageFolder(root="./dataset", transform=None)
        total = len(full_dataset)
        train_size = int(0.70 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        generator = torch.Generator().manual_seed(42)

        _, _, test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        test_dataset = TransformSubset(test_set, transform)
        class_names = full_dataset.classes
        model = build_model(9)
        state_path = os.path.join("classification_saved_results","classification_model.pth")
        return test_dataset, class_names, model, state_path

    if mode == "glass_plastic":
        full_dataset = datasets.ImageFolder(root="./dataset", transform=None)
        glass_idx = full_dataset.class_to_idx["Glass"]
        plastic_idx = full_dataset.class_to_idx["Plastic"]
        label_map = {glass_idx: 0, plastic_idx: 1}
        binary_dataset = BinarySubsetDataset(full_dataset, label_map, transform=None)
        total = len(binary_dataset)
        train_size = int(0.70 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        _, _, test_set = random_split(
            binary_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        test_dataset = TransformSubset(test_set, transform)
        class_names = ["Glass", "Plastic"]
        model = build_binary_model()
        state_path = os.path.join("glass_plastic_saved_results", "glass_plastic_model.pth")
        return test_dataset, class_names, model, state_path

    if mode == "glass_anomaly":
        full_dataset = datasets.ImageFolder(root="./glass_anomaly_detection_dataset", transform=None)
        class_names = ["Broken", "Normal"]
        binary_dataset = full_dataset
        total = len(binary_dataset)
        train_size = int(0.70 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        generator = torch.Generator().manual_seed(42)

        _, _, test_set = random_split(
            binary_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        test_dataset = TransformSubset(test_set, transform)
        model = build_binary_model()
        state_path = os.path.join("glass_anomaly_saved_results", "glass_anomaly_model.pth")
        return test_dataset, class_names, model, state_path

    raise ValueError("Unsupported mode")


def evaluate(mode):
    transform = build_eval_transform()
    test_dataset, class_names, model, state_path = make_test_loader(mode, transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {state_path}. Train the model first with 'python src/train_classification.py' or the corresponding binary training script."
        )

    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[{mode}] Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names
        )

    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    if mode == "waste_classification":
        output_dir = "classification_saved_results"

    elif mode == "glass_plastic":
        output_dir = "glass_plastic_saved_results"

    elif mode == "glass_anomaly":
        output_dir = "glass_anomaly_saved_results"

    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, f"{mode}_classification_report.txt"),
        "w"
    ) as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{mode.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{mode}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["waste_classification", "glass_plastic", "glass_anomaly"],
        default="waste_classification"
    )
    args = parser.parse_args()
    evaluate(args.mode)