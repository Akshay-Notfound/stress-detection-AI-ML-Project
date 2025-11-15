"""
Dataset loader for common stress detection datasets.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import pickle

def load_wesad_dataset(subject_id: str, data_dir: str = "data/raw") -> Optional[Dict]:
    """
    Load WESAD dataset for a specific subject.
    
    Args:
        subject_id (str): Subject ID (e.g., 'S2', 'S3', etc.)
        data_dir (str): Directory containing the dataset files
        
    Returns:
        dict: Dictionary containing the data and labels, or None if loading failed
    """
    try:
        # WESAD dataset files are typically named like 'S2.pkl'
        file_path = os.path.join(data_dir, f"{subject_id}.pkl")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract relevant information
        chest_data = data['signal']['chest']
        wrist_data = data['signal']['wrist']
        labels = data['label']
        
        # Create a structured output
        result = {
            'subject_id': subject_id,
            'chest': chest_data,
            'wrist': wrist_data,
            'labels': labels,
            'sampling_rates': data['sampling_rate']
        }
        
        return result
        
    except Exception as e:
        print(f"Error loading WESAD dataset for {subject_id}: {e}")
        return None

def load_swell_kw_dataset(data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """
    Load SWELL-KW dataset.
    
    Args:
        data_dir (str): Directory containing the dataset files
        
    Returns:
        pd.DataFrame: DataFrame containing the data, or None if loading failed
    """
    try:
        # SWELL-KW dataset structure may vary, this is a generic loader
        # You would need to adjust this based on the actual dataset structure
        
        # Example for CSV-based dataset
        file_path = os.path.join(data_dir, "swell_kw_data.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        return data
        
    except Exception as e:
        print(f"Error loading SWELL-KW dataset: {e}")
        return None

def load_dreaddit_dataset(data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """
    Load Dreaddit dataset for text-based stress detection.
    
    Args:
        data_dir (str): Directory containing the dataset files
        
    Returns:
        pd.DataFrame: DataFrame containing the data, or None if loading failed
    """
    try:
        # Dreaddit dataset is typically in CSV format
        file_path = os.path.join(data_dir, "dreaddit.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        return data
        
    except Exception as e:
        print(f"Error loading Dreaddit dataset: {e}")
        return None

def convert_wesad_to_csv(subject_data: Dict, output_dir: str = "data/processed") -> bool:
    """
    Convert WESAD subject data to CSV format.
    
    Args:
        subject_data (dict): Data loaded from WESAD dataset
        output_dir (str): Directory to save the CSV files
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        subject_id = subject_data['subject_id']
        
        # Convert chest data to DataFrame
        chest_df = pd.DataFrame(subject_data['chest'])
        chest_df['label'] = subject_data['labels']
        chest_file = os.path.join(output_dir, f"{subject_id}_chest.csv")
        chest_df.to_csv(chest_file, index=False)
        
        # Convert wrist data to DataFrame
        wrist_df = pd.DataFrame(subject_data['wrist'])
        wrist_df['label'] = subject_data['labels']
        wrist_file = os.path.join(output_dir, f"{subject_id}_wrist.csv")
        wrist_df.to_csv(wrist_file, index=False)
        
        print(f"Converted WESAD data for {subject_id} to CSV format")
        return True
        
    except Exception as e:
        print(f"Error converting WESAD data to CSV: {e}")
        return False

def load_generic_physiological_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load generic physiological data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the data, or None if loading failed
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Verify that required columns exist
        required_columns = ['timestamp', 'ecg', 'eda', 'acc_x', 'acc_y', 'acc_z']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in the dataset: {missing_columns}")
            print(f"Available columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"Error loading generic physiological data: {e}")
        return None

def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Information about the dataset
    """
    dataset_info = {
        'wesad': {
            'name': 'WESAD (Wearable Stress and Affect Detection)',
            'description': 'Contains physiological data from 15 subjects during neutral, stress, and amusement conditions',
            'sensors': ['ECG', 'EDA', 'EMG', 'TEMP', 'ACC'],
            'subjects': 15,
            'labels': ['Baseline', 'Stress', 'Amusement'],
            'sampling_rates': {
                'chest': '700 Hz (ECG), 32 Hz (others)',
                'wrist': '64 Hz (ACC), 4 Hz (others)'
            }
        },
        'swell_kw': {
            'name': 'SWELL-KW (SWELL Knowledge Work)',
            'description': 'Focuses on "office work" stress with data from 25 subjects',
            'sensors': ['ECG', 'Computer logging', 'Posture', 'Facial expressions'],
            'subjects': 25,
            'labels': ['No stress', 'Time pressure', 'Interruptions'],
            'sampling_rates': 'Varies by sensor'
        },
        'dreaddit': {
            'name': 'Dreaddit',
            'description': 'Text-based stress detection dataset from Reddit posts',
            'sensors': ['Text data'],
            'subjects': 'N/A (text posts)',
            'labels': ['Stressed', 'Not stressed'],
            'sampling_rates': 'N/A'
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {
        'name': dataset_name,
        'description': 'Unknown dataset',
        'sensors': [],
        'subjects': 0,
        'labels': [],
        'sampling_rates': 'Unknown'
    })

if __name__ == "__main__":
    # Example usage
    print("Dataset Loader for Stress Detection")
    print("=" * 40)
    
    # Show information about common datasets
    datasets = ['wesad', 'swell_kw', 'dreaddit']
    for dataset in datasets:
        info = get_dataset_info(dataset)
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Sensors: {', '.join(info['sensors'])}")
        print(f"  Number of subjects: {info['subjects']}")
        print(f"  Labels: {', '.join(info['labels'])}")