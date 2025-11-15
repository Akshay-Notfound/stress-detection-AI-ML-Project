"""
Test script to verify the stress detection pipeline.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

from utils import create_directories_if_not_exists
from preprocessing import preprocess_ecg, preprocess_eda, segment_signals
from features import extract_all_features
from models import StressDetectionModel, compare_models

def test_pipeline():
    """Test the complete stress detection pipeline."""
    print("Testing Stress Detection Pipeline")
    print("=" * 40)
    
    # Create directories
    dirs = ['data/raw', 'data/processed', 'models']
    create_directories_if_not_exists(dirs)
    print("✓ Directory structure created")
    
    # Create sample data
    print("\n1. Creating sample physiological data...")
    sample_rate = 256  # Hz
    duration = 60  # seconds
    time_points = np.linspace(0, duration, duration * sample_rate)
    
    # Simulate ECG signal
    ecg = np.sin(2 * np.pi * 1.2 * time_points) + 0.5 * np.random.normal(size=len(time_points))
    
    # Simulate EDA signal
    eda = 5 + 2 * np.sin(2 * np.pi * 0.1 * time_points) + np.random.normal(size=len(time_points))
    
    # Simulate accelerometer data
    acc_x = np.random.normal(0, 0.1, size=len(time_points))
    acc_y = np.random.normal(0, 0.1, size=len(time_points))
    acc_z = 9.8 + np.random.normal(0, 0.1, size=len(time_points))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': time_points,
        'ecg': ecg,
        'eda': eda,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z
    })
    
    print("✓ Sample data created")
    print(f"  Data shape: {data.shape}")
    
    # Test preprocessing
    print("\n2. Testing preprocessing functions...")
    ecg_processed = preprocess_ecg(data['ecg'].values, sample_rate)
    if ecg_processed:
        print("✓ ECG preprocessing successful")
    else:
        print("✗ ECG preprocessing failed")
    
    eda_processed = preprocess_eda(data['eda'].values, sample_rate)
    if eda_processed:
        print("✓ EDA preprocessing successful")
    else:
        print("✗ EDA preprocessing failed")
    
    # Test segmentation
    print("\n3. Testing signal segmentation...")
    windows = segment_signals(data, window_size=256, overlap=0.5)
    print(f"✓ Segmented data into {len(windows)} windows")
    
    # Test feature extraction
    print("\n4. Testing feature extraction...")
    if windows:
        # Extract features from first window
        features = extract_all_features(windows[0])
        print("✓ Feature extraction successful")
        print(f"  Number of features extracted: {len(features.columns)}")
    
    # Test modeling
    print("\n5. Testing modeling functions...")
    try:
        # Create sample features for modeling
        n_samples = 50
        features_df = pd.DataFrame({
            'mean_rr': np.random.normal(800, 50, n_samples),
            'sdnn': np.random.normal(50, 10, n_samples),
            'rmssd': np.random.normal(30, 15, n_samples),
            'pnn50': np.random.normal(10, 5, n_samples),
            'lf_hf_ratio': np.random.normal(1.5, 0.5, n_samples),
            'mean_eda': np.random.normal(6, 2, n_samples),
            'std_eda': np.random.normal(1.5, 0.5, n_samples),
            'num_scr_peaks': np.random.poisson(5, n_samples),
            'mean_acc_magnitude': np.random.normal(10, 2, n_samples),
            'std_acc_magnitude': np.random.normal(1, 0.3, n_samples),
            'label': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        })
        
        # Separate features and labels
        feature_columns = [col for col in features_df.columns if col not in ['label']]
        X = features_df[feature_columns]
        y = features_df['label']
        
        # Test individual model
        model = StressDetectionModel('rf')
        model.train(X, y)
        print("✓ Individual model training successful")
        
        # Test prediction
        predictions = model.predict(X[:5])
        print(f"✓ Predictions successful: {predictions}")
        
        # Test model comparison
        comparison_results = compare_models(X, y, test_size=0.3)
        print("✓ Model comparison successful")
        print("  Models compared:", list(comparison_results.keys()))
        
    except Exception as e:
        print(f"✗ Modeling test failed: {e}")
    
    print("\n" + "=" * 40)
    print("Pipeline test completed!")

if __name__ == "__main__":
    test_pipeline()