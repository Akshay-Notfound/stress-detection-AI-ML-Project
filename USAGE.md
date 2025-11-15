# Usage Guide

This guide explains how to set up and use the stress detection system.

## Installation

### Option 1: Automatic Installation (Windows)

Run the [install_and_run.bat](install_and_run.bat) script to automatically create a virtual environment and install all dependencies.

### Option 2: Manual Installation

1. Create a virtual environment:
   ```bash
   python -m venv stress_detection_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     stress_detection_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source stress_detection_env/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized as follows:

```
stress-detection/
├─ data/
│  ├─ raw/                # Raw dataset files
│  └─ processed/          # Processed windows and labels
├─ notebooks/
│  ├─ 01-exploration.ipynb      # Data exploration
│  ├─ 02-feature-engineering.ipynb  # Feature extraction
│  └─ 03-modeling.ipynb     # Model training and evaluation
├─ src/
│  ├─ preprocessing.py      # Signal preprocessing functions
│  ├─ features.py           # Feature extraction functions
│  ├─ models.py             # Machine learning models
│  └─ utils.py              # Utility functions
├─ app/
│  └─ streamlit_app.py      # Streamlit demo application
├─ requirements.txt         # Python dependencies
├─ setup.py                # Package setup
├─ README.md               # Project overview
└─ USAGE.md                # This file
```

## Running the Application

### 1. Streamlit Demo Application

To run the interactive demo:

```bash
streamlit run app/streamlit_app.py
```

The application provides:
- Data upload functionality
- Real-time stress detection simulation
- Model information and performance metrics

### 2. Jupyter Notebooks

To explore the data and models:

```bash
jupyter notebook
```

Then open the notebooks in the [notebooks/](notebooks/) directory in this order:
1. [01-exploration.ipynb](notebooks/01-exploration.ipynb) - Data exploration
2. [02-feature-engineering.ipynb](notebooks/02-feature-engineering.ipynb) - Feature extraction
3. [03-modeling.ipynb](notebooks/03-modeling.ipynb) - Model training and evaluation

### 3. Test Pipeline

To run the test pipeline:

```bash
python test_pipeline.py
```

This script verifies that all components of the pipeline work correctly.

## Using Your Own Data

To use your own physiological data:

1. Place your dataset files in the [data/raw/](data/raw/) directory
2. Modify the notebooks to load your data format
3. Adjust preprocessing functions if needed for your sensor specifications
4. Retrain models with your data

## Supported Data Formats

The system currently supports:
- CSV files with physiological signals
- ECG, EDA, and accelerometer data
- Timestamp-aligned signals

## Model Training

To train models with your data:
1. Follow the notebooks to preprocess and extract features
2. Use the [models.py](src/models.py) module to train different algorithms
3. Evaluate performance using cross-validation
4. Save the best performing model

## Customization

You can customize the system by:
- Adding new preprocessing functions in [preprocessing.py](src/preprocessing.py)
- Implementing additional features in [features.py](src/features.py)
- Adding new models in [models.py](src/models.py)
- Extending the Streamlit app in [streamlit_app.py](app/streamlit_app.py)

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all packages in [requirements.txt](requirements.txt) are installed
2. **Import errors**: Check that you're running scripts from the project root directory
3. **Data loading errors**: Verify your data format matches the expected structure

### Getting Help

If you encounter issues:
1. Check the error messages for specific details
2. Verify your Python environment and dependencies
3. Ensure you're using Python 3.8 or higher
4. Consult the documentation for individual packages (scikit-learn, neurokit2, etc.)

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Test thoroughly
5. Submit a pull request