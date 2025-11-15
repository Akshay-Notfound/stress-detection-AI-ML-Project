# Real-Time Stress Detection Using Machine Learning

## Objective
To design, build, and evaluate a machine learning model in Python that classifies an individual's psychological stress level (e.g., No Stress, Moderate Stress, High Stress). The model will be trained on either physiological sensor data or text-based data to identify patterns indicative of stress.

## Project Structure
```
stress-detection/
├─ data/
│  ├─ raw/                # raw dataset files
│  └─ processed/          # processed windows + labels
├─ notebooks/
│  ├─ 01-exploration.ipynb
│  ├─ 02-feature-engineering.ipynb
│  └─ 03-modeling.ipynb
├─ src/
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ models.py
│  └─ utils.py
├─ app/
│  └─ streamlit_app.py
├─ requirements.txt
└─ README.md
```

## Key Features
- Preprocessing of physiological signals (ECG/PPG, EDA/GSR, accelerometer)
- Feature extraction including heart rate variability (HRV), statistical descriptors, and spectral features
- Implementation of classical ML models (Random Forest, XGBoost) and deep learning models (CNN/LSTM)
- Model evaluation using balanced metrics (F1, ROC-AUC) and subject-independent validation
- Streamlit demo application for real-time or batch inference
- Analysis of feature importance and model explainability

## Installation
1. Run the automatic installer (Windows only):
   ```bash
   install_and_run.bat
   ```
   
2. Or install manually:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your dataset in the `data/raw/` directory
2. Run the Jupyter notebooks in sequence for data exploration, feature engineering, and modeling
3. Quick start with the run script:
   ```bash
   python run_demo.py
   ```
4. Or launch the Streamlit app for interactive stress detection:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Datasets
- WESAD (Wearable Stress and Affect Detection)
- SWELL-KW (SWELL Knowledge Work)
- Dreaddit (for text-based stress detection)

## Evaluation Strategy
- Subject-independent validation (leave-one-subject-out)
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix analysis