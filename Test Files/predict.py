import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
from model import build_model


CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]


def load_model(model_path, device):
    """Load trained model"""
    model = build_model(num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device):
    """Predict waste category for an image"""
   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

   
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100

    return class_name, confidence_score, probabilities

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model_path = "saved_models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("Loading model...")
    model = load_model(model_path, device)
    print("Model loaded successfully!")

    
    data_dir = "data"
    test_image = None

    for class_folder in Path(data_dir).iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
            if images:
                test_image = images[0]
                break

    if test_image:
        print(f"\nTesting with image: {test_image}")
        class_name, confidence, probs = predict_image(str(test_image), model, device)
        print(f"Predicted Class: {class_name}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nAll probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs[0])):
            print(f"  {name}: {prob.item()*100:.2f}%")
    else:
        print("No test images found in data folder")

if __name__ == "__main__":
    main()
