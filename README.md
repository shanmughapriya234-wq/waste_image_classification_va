# waste-image-classification

## Project Overview
This project classifies waste into 9 categories, verifies ambiguous Glass/Plastic predictions, detects glass anomalies, and provides explainable AI outputs using Grad-CAM and LIME.

## Model Accuracy
- Main waste classification (ResNet-based): 80%
- Glass vs Plastic verification: 92%
- Glass anomaly detection: 93%

## Setup
### 1. Create a virtual environment
```bash
python3 -m venv venv
```

### 2. Activate the virtual environment
```bash
source venv/bin/activate
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify the environment
```bash
python -V
pip -V
```

## Run the Project
Run the scripts in the following order:

```bash
python src/preprocess.py
python src/model.py

Train Model:
python src/train_classification.py
python src/glass_plastic_classifier.py
python src/glass_anomaly_classifier.py


python src/evaluation_matrix.py --mode waste_classification
python src/evaluation_matrix.py --mode glass_plastic
python src/evaluation_matrix.py --mode glass_anomaly
python src/predict.py

streamlit run src/streamlite_ui.py
```



## Notes
- The training script uses the main 9-class classifier.
- The Glass/Plastic and Glass anomaly scripts are used for the secondary verification pipeline.
- The Streamlit app provides the full end-to-end interface with explainable AI visualizations.