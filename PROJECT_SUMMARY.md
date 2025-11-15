# Stress Detection Project - Summary

## Project Overview

This project implements a complete machine learning system for detecting human stress from physiological and behavioral signals. The system is designed to preprocess raw sensor data, extract meaningful features, train and evaluate machine learning models, and provide a user-friendly interface for real-time stress detection.

## System Architecture

The project follows a modular architecture with clearly separated components:

```
stress-detection/
├─ data/                    # Data storage
│  ├─ raw/                 # Raw dataset files
│  └─ processed/           # Processed features and labels
├─ notebooks/              # Jupyter notebooks for experimentation
│  ├─ 01-exploration.ipynb        # Data exploration
│  ├─ 02-feature-engineering.ipynb # Feature extraction
│  └─ 03-modeling.ipynb           # Model training and evaluation
├─ src/                    # Core source code
│  ├─ preprocessing.py     # Signal preprocessing functions
│  ├─ features.py          # Feature extraction algorithms
│  ├─ models.py            # Machine learning models
│  ├─ utils.py             # Utility functions
│  └─ dataset_loader.py    # Dataset loading utilities
├─ app/                    # Application code
│  └─ streamlit_app.py     # Interactive web application
├─ requirements.txt        # Python dependencies
├─ setup.py               # Package setup
├─ README.md              # Project overview
├─ USAGE.md               # Detailed usage instructions
├─ PROJECT_SUMMARY.md     # This file
└─ install_and_run.bat    # Installation script (Windows)
```

## Key Components

### 1. Data Preprocessing ([src/preprocessing.py](src/preprocessing.py))

Implements signal processing functions for:
- Bandpass filtering for ECG/PPG and EDA signals
- ECG preprocessing with R-peak detection using neurokit2
- EDA signal filtering
- Accelerometer data processing
- Signal segmentation into windows
- Artifact removal using accelerometer data

### 2. Feature Engineering ([src/features.py](src/features.py))

Extracts comprehensive features from physiological signals:
- **Time-domain HRV features**: SDNN, RMSSD, pNN50
- **Frequency-domain HRV features**: LF, HF, LF/HF ratio
- **EDA features**: Mean, standard deviation, peak count
- **Accelerometer features**: Movement patterns and activity levels
- **Statistical features**: Skewness, kurtosis, entropy

### 3. Machine Learning Models ([src/models.py](src/models.py))

Implements multiple ML algorithms with consistent interface:
- **Random Forest**: Robust ensemble method
- **XGBoost**: Gradient boosting algorithm
- **Support Vector Machine**: Classical classifier
- **Logistic Regression**: Linear baseline model

Features include:
- Model training and evaluation
- Hyperparameter tuning with grid search
- Cross-validation (including leave-one-subject-out)
- Model persistence (save/load functionality)
- Comprehensive performance metrics

### 4. Interactive Demo Application ([app/streamlit_app.py](app/streamlit_app.py))

A Streamlit-based web application that provides:
- Data upload functionality
- Real-time stress detection simulation
- Model performance visualization
- Feature importance analysis
- User-friendly interface for non-technical users

### 5. Jupyter Notebooks ([notebooks/](notebooks/))

Comprehensive experimentation environment:
- **01-exploration.ipynb**: Data loading, visualization, and basic statistics
- **02-feature-engineering.ipynb**: Signal processing and feature extraction
- **03-modeling.ipynb**: Model training, evaluation, and comparison

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Automatic Installation (Windows)**:
   Run [install_and_run.bat](install_and_run.bat) to automatically create a virtual environment and install dependencies.

2. **Manual Installation**:
   ```bash
   # Create virtual environment
   python -m venv stress_detection_env
   
   # Activate virtual environment
   # On Windows:
   stress_detection_env\Scripts\activate
   # On macOS/Linux:
   source stress_detection_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage Instructions

### Running the Interactive Demo
```bash
streamlit run app/streamlit_app.py
```

### Exploring with Jupyter Notebooks
```bash
jupyter notebook
```
Then open the notebooks in order:
1. [01-exploration.ipynb](notebooks/01-exploration.ipynb)
2. [02-feature-engineering.ipynb](notebooks/02-feature-engineering.ipynb)
3. [03-modeling.ipynb](notebooks/03-modeling.ipynb)

### Testing the Pipeline
```bash
python test_pipeline.py
```

## Supported Datasets

The system is designed to work with:
- **WESAD**: Wearable Stress and Affect Detection dataset
- **SWELL-KW**: Knowledge work stress dataset
- **Dreaddit**: Text-based stress detection dataset
- Custom physiological data in CSV format

## Evaluation Methodology

The system implements robust evaluation strategies:
- **Subject-independent validation**: Leave-one-subject-out cross-validation
- **Balanced metrics**: F1-score, ROC-AUC, precision, recall
- **Confusion matrix analysis**: Detailed error analysis
- **Feature importance**: Model interpretability

## Technical Features

### Signal Processing
- Bandpass filtering for noise reduction
- R-peak detection for HRV analysis
- Artifact removal using accelerometer data
- Windowed signal segmentation

### Feature Extraction
- Heart Rate Variability (HRV) features
- Electrodermal Activity (EDA) features
- Accelerometer-based movement features
- Statistical and spectral features

### Machine Learning
- Multiple algorithm support
- Hyperparameter optimization
- Cross-validation strategies
- Model persistence and versioning

### User Interface
- Web-based interactive application
- Real-time data simulation
- Visual performance metrics
- Feature importance visualization

## Extensibility

The modular design allows for easy extension:
- Add new preprocessing functions in [preprocessing.py](src/preprocessing.py)
- Implement additional features in [features.py](src/features.py)
- Add new models in [models.py](src/models.py)
- Extend the Streamlit app in [streamlit_app.py](app/streamlit_app.py)

## Performance Considerations

- Efficient signal processing algorithms
- Optimized feature extraction pipeline
- Memory-efficient data handling
- Parallel processing where applicable

## Future Enhancements

Potential areas for improvement:
- Deep learning models (CNN, LSTM) for raw signal processing
- Real-time streaming data processing
- Mobile application deployment
- Additional sensor modalities
- Advanced model explainability techniques

## Conclusion

This stress detection system provides a complete pipeline from raw physiological data to real-time stress classification. The modular architecture, comprehensive documentation, and interactive demo make it suitable for both research and practical applications. The system's flexibility allows adaptation to different datasets and use cases while maintaining high performance and interpretability.