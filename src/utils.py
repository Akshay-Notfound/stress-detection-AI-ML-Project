"""
Utility functions for the stress detection project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def save_dataset(data: pd.DataFrame, file_path: str) -> None:
    """
    Save dataset to file.
    
    Args:
        data (pd.DataFrame): Dataset to save
        file_path (str): Path to save the dataset
    """
    try:
        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif file_path.endswith('.json'):
            data.to_json(file_path, orient='records')
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file path.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

def create_directories_if_not_exists(paths: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths (List[str]): List of directory paths to create
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def normalize_data(data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Normalize data using z-score normalization.
    
    Args:
        data (pd.DataFrame): Data to normalize
        columns (List[str], optional): Columns to normalize. If None, normalize all numeric columns.
        
    Returns:
        pd.DataFrame: Normalized data
    """
    data_copy = data.copy()
    
    if columns is None:
        columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in data_copy.columns:
            mean = data_copy[col].mean()
            std = data_copy[col].std()
            if std != 0:
                data_copy[col] = (data_copy[col] - mean) / std
            else:
                data_copy[col] = 0
    
    return data_copy

def handle_missing_values(data: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Data with missing values
        method (str): Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    data_copy = data.copy()
    
    if method == 'interpolate':
        return data_copy.interpolate()
    elif method == 'forward_fill':
        return data_copy.fillna(method='ffill')
    elif method == 'backward_fill':
        return data_copy.fillna(method='bfill')
    elif method == 'drop':
        return data_copy.dropna()
    else:
        raise ValueError("Invalid method. Choose from 'interpolate', 'forward_fill', 'backward_fill', 'drop'")