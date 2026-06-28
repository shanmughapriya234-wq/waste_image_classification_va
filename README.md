# WasteVisionX: Explainable Classification and Anomaly Detection for Waste Sorting

The project for smart waste sorting using PyTorch and transfer learning.
It classifies waste into 9 categories, resolves the common Glass vs Plastic confusion
with a dedicated binary classifier, detects broken glass through anomaly detection, and
explains its predictions using Grad-CAM and LIME.

## Features
- 9-class waste image classification
- Binary Classification for Glass vs Plastic for the most confused pair
- Glass anomaly detection (Broken vs Normal) for the safety-critical case
- Explainable AI with Grad-CAM and LIME
- Interactive Streamlit app for the full end-to-end pipeline
- Transfer learning with a pretrained ResNet50 backbone
- Data augmentation (flips, rotation, color jitter, etc.,) for better generalization
- Automatic saving of the best model based on validation accuracy

## Requirements
- Python 3.12
- PyTorch / torchvision
- scikit-learn, matplotlib, seaborn (metrics and plots)
- Grad-CAM, LIME, OpenCV (explainability)
- Streamlit (app)
- See `requirements.txt` for the full list of dependencies

## Dataset
RealWaste dataset from Kaggle: https://www.kaggle.com/datasets/joebeachcapital/realwaste

Place the 9-class images in `dataset/` (one subfolder per class) and the glass anomaly
images in `glass_anomaly_detection_dataset/` (broken / normal subfolders). For the Glass vs 
Plastic images in subfolder of `dataset/` 

## Setup
1. Create and activate a virtual environment (Python 3.12):
   
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
 
2. Install the dependencies:
   
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   

## Usage
Command to Train the models and python filenames:

python src/train_classification.py --> Classification of 9 waste categories 
python src/glass_plastic_classifier.py --> Re-Verification of Glass vs Plastic
python src/glass_anomaly_classifier.py --> Anomaly detection of Glass

Evaluate the models (accuracy, classification report, confusion matrix):

python src/evaluation_matrix.py --mode waste_classification --> For 9 waste categories 
python src/evaluation_matrix.py --mode glass_plastic --> For Glass vs Plastic
python src/evaluation_matrix.py --mode glass_anomaly --> Anomaly detection of Glass

Launch the interactive app with XAI visualizations:

streamlit run src/streamlite_ui.py


## Results
Test accuracy on held-out sets:
- 9-class waste classification: 80%
- Glass vs Plastic verification: 93%
- Glass anomaly detection (Broken vs Normal): 97%

Each model saves its outputs (trained `.pth` model, confusion matrix, classification
report, training history CSV, and summary) to its own `*_saved_results/` folder.

## Project Structure

1. dataset/                          # 9-class waste images (one folder per class)
2. glass_anomaly_detection_dataset/  # Broken / Normal glass images
3. src/                              # Training, evaluation, XAI, and Streamlit app
4.  Test Files/                       # Single-image prediction scripts
5. *_saved_results/                  # Trained models + reports + confusion matrices
6. requirements.txt
7. README.md


## Notes
- Training was performed on Apple Silicon using the MPS backend (CUDA also supported
  on NVIDIA GPUs; the scripts auto-select MPS, CUDA, or CPU).
- The main 9-class classifier handles broad sorting; the Glass/Plastic and glass
  anomaly models form the secondary verification pipeline.
- Explainable AI , Grad-Cam and LIME is used for explaination of predicted output

## Team
- Sindhuja Ponnuswamy Periyaswamy
- Shanmughapriya Mounissamy
- Harshana Dwaralu Srinivas
