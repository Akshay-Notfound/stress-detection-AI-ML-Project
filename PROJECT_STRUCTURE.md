# Complete Project Structure

This document provides a comprehensive overview of all files and directories in the stress detection project.

## Root Directory

```
stress-detection/
├─ .venv/                  # Virtual environment (created during installation)
├─ app/                    # Application code
├─ data/                   # Data storage
├─ notebooks/              # Jupyter notebooks for experimentation
├─ src/                    # Core source code
├─ PROJECT_SUMMARY.md      # Project overview and technical details
├─ PROJECT_STRUCTURE.md    # This file
├─ README.md               # Project overview and quick start guide
├─ USAGE.md                # Detailed usage instructions
├─ install_and_run.bat     # Automatic installation script (Windows)
├─ requirements.txt        # Python dependencies
├─ run_demo.py             # Simple demo runner script
└─ setup.py                # Package setup
```

## Application Directory (`app/`)

```
app/
└─ streamlit_app.py        # Interactive web application using Streamlit
```

## Data Directory (`data/`)

```
data/
├─ raw/                    # Raw dataset files (user-provided)
└─ processed/              # Processed features and labels (generated)
```

## Notebooks Directory (`notebooks/`)

```
notebooks/
├─ 01-exploration.ipynb          # Data exploration and visualization
├─ 02-feature-engineering.ipynb  # Signal processing and feature extraction
└─ 03-modeling.ipynb             # Model training, evaluation, and comparison
```

## Source Code Directory (`src/`)

```
src/
├─ dataset_loader.py       # Utilities for loading common stress detection datasets
├─ features.py            # Feature extraction algorithms for physiological signals
├─ models.py              # Machine learning models with consistent interface
├─ preprocessing.py       # Signal preprocessing functions
└─ utils.py               # General utility functions
```

## Detailed File Descriptions

### Core Implementation Files

1. **[src/preprocessing.py](src/preprocessing.py)** - Signal processing functions:
   - Bandpass filtering for ECG/PPG and EDA signals
   - ECG preprocessing with R-peak detection
   - Accelerometer data processing
   - Signal segmentation and artifact removal

2. **[src/features.py](src/features.py)** - Feature extraction algorithms:
   - Time-domain HRV features (SDNN, RMSSD, pNN50)
   - Frequency-domain HRV features (LF, HF, LF/HF ratio)
   - EDA features (mean, std, peak count)
   - Accelerometer features (movement patterns)
   - Statistical features (skewness, kurtosis)

3. **[src/models.py](src/models.py)** - Machine learning models:
   - Random Forest, XGBoost, SVM, Logistic Regression
   - Model training, evaluation, and hyperparameter tuning
   - Cross-validation (including leave-one-subject-out)
   - Model persistence functionality

4. **[src/utils.py](src/utils.py)** - Utility functions:
   - Data loading and saving
   - Directory management
   - Data normalization and missing value handling

5. **[src/dataset_loader.py](src/dataset_loader.py)** - Dataset utilities:
   - Loaders for WESAD, SWELL-KW, and Dreaddit datasets
   - Data format conversion utilities
   - Dataset information and metadata

### Application Files

6. **[app/streamlit_app.py](app/streamlit_app.py)** - Interactive web application:
   - Data upload interface
   - Real-time stress detection simulation
   - Model performance visualization
   - Feature importance analysis

### Documentation Files

7. **[README.md](README.md)** - Project overview and quick start guide

8. **[USAGE.md](USAGE.md)** - Detailed usage instructions

9. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical overview and architecture

10. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - This file

### Installation and Setup Files

11. **[requirements.txt](requirements.txt)** - Python dependencies:
    - pandas, numpy, scipy
    - scikit-learn, xgboost
    - neurokit2, pyhrv
    - matplotlib, seaborn
    - streamlit

12. **[setup.py](setup.py)** - Package setup for distribution

13. **[install_and_run.bat](install_and_run.bat)** - Automatic installation script (Windows)

14. **[run_demo.py](run_demo.py)** - Simple demo runner script

### Experimentation Files

15. **[notebooks/01-exploration.ipynb](notebooks/01-exploration.ipynb)** - Data exploration:
    - Data loading and basic statistics
    - Signal visualization
    - Correlation analysis

16. **[notebooks/02-feature-engineering.ipynb](notebooks/02-feature-engineering.ipynb)** - Feature extraction:
    - Signal preprocessing
    - Windowed feature extraction
    - Feature visualization

17. **[notebooks/03-modeling.ipynb](notebooks/03-modeling.ipynb)** - Model training:
    - Model comparison
    - Hyperparameter tuning
    - Cross-validation
    - Feature importance analysis

### Testing Files

18. **[test_pipeline.py](test_pipeline.py)** - End-to-end pipeline testing:
    - Data generation
    - Preprocessing testing
    - Feature extraction testing
    - Model training and prediction testing

## Usage Flow

1. **Installation**: Run [install_and_run.bat](install_and_run.bat) or manually install dependencies
2. **Data Preparation**: Place datasets in [data/raw/](data/raw/)
3. **Exploration**: Run [01-exploration.ipynb](notebooks/01-exploration.ipynb) to understand data
4. **Feature Engineering**: Run [02-feature-engineering.ipynb](notebooks/02-feature-engineering.ipynb) to extract features
5. **Modeling**: Run [03-modeling.ipynb](notebooks/03-modeling.ipynb) to train and evaluate models
6. **Demo**: Run [run_demo.py](run_demo.py) or [streamlit_app.py](app/streamlit_app.py) for interactive application

## Key Features by Component

### Preprocessing
- Signal filtering and noise reduction
- R-peak detection for HRV analysis
- Artifact removal using accelerometer data
- Windowed signal segmentation

### Feature Engineering
- Comprehensive HRV feature extraction
- EDA signal analysis
- Movement pattern recognition
- Statistical and spectral features

### Machine Learning
- Multiple algorithm support
- Cross-validation strategies
- Hyperparameter optimization
- Model persistence

### User Interface
- Web-based interactive application
- Real-time data simulation
- Performance visualization
- Feature importance display

This structure provides a complete, modular system for stress detection from physiological signals with clear separation of concerns and easy extensibility.