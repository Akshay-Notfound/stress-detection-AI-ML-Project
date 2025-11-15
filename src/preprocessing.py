"""
Preprocessing functions for stress detection from physiological signals.
"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Optional
import neurokit2 as nk

def bandpass_filter(signal_data: np.ndarray, lowcut: float, highcut: float, 
                   fs: float, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        signal_data (np.ndarray): Input signal
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        fs (float): Sampling frequency
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0:
        # Use lowpass filter
        b, a = signal.butter(order, high, btype='low')
    elif high >= 1:
        # Use highpass filter
        b, a = signal.butter(order, low, btype='high')
    else:
        # Use bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
    
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def preprocess_ecg(ecg_signal: np.ndarray, sampling_rate: float) -> Optional[Dict]:
    """
    Preprocess ECG signal and extract R-peaks.
    
    Args:
        ecg_signal (np.ndarray): ECG signal data
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        dict: Dictionary containing processed ECG data and R-peaks
    """
    try:
        # Clean ECG signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        r_peaks = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=sampling_rate)
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks['ECG_R_Peaks']) / sampling_rate * 1000  # in milliseconds
        
        return {
            'ecg_cleaned': ecg_cleaned,
            'r_peaks': r_peaks['ECG_R_Peaks'],
            'rr_intervals': rr_intervals
        }
    except Exception as e:
        print(f"Error preprocessing ECG: {e}")
        return None

def preprocess_eda(eda_signal: np.ndarray, sampling_rate: float) -> Optional[Dict]:
    """
    Preprocess EDA signal.
    
    Args:
        eda_signal (np.ndarray): EDA signal data
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        dict: Dictionary containing processed EDA data
    """
    try:
        # Apply low-pass filter to remove noise (typically < 1 Hz for EDA)
        eda_filtered = bandpass_filter(eda_signal, 0.01, 1.0, sampling_rate, order=4)
        
        return {
            'eda_filtered': eda_filtered
        }
    except Exception as e:
        print(f"Error preprocessing EDA: {e}")
        return None

def preprocess_accelerometer(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Optional[Dict]:
    """
    Preprocess accelerometer data.
    
    Args:
        acc_x (np.ndarray): X-axis acceleration
        acc_y (np.ndarray): Y-axis acceleration
        acc_z (np.ndarray): Z-axis acceleration
        
    Returns:
        dict: Dictionary containing processed accelerometer data
    """
    try:
        # Calculate magnitude of acceleration
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        return {
            'acc_magnitude': acc_magnitude,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z
        }
    except Exception as e:
        print(f"Error preprocessing accelerometer: {e}")
        return None

def segment_signals(data: pd.DataFrame, window_size: int, overlap: float = 0.5) -> List[pd.DataFrame]:
    """
    Segment signals into windows.
    
    Args:
        data (pd.DataFrame): Signal data
        window_size (int): Size of each window in samples
        overlap (float): Overlap between windows (0.0 to 1.0)
        
    Returns:
        List[pd.DataFrame]: List of segmented windows
    """
    try:
        step_size = int(window_size * (1 - overlap))
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            window['window_id'] = i // step_size
            windows.append(window)
            
        return windows
    except Exception as e:
        print(f"Error segmenting signals: {e}")
        return []

def remove_artifacts(data: pd.DataFrame, acc_threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove artifacts from physiological signals based on accelerometer data.
    
    Args:
        data (pd.DataFrame): Signal data with accelerometer columns
        acc_threshold (float): Threshold for detecting motion artifacts
        
    Returns:
        pd.DataFrame: Data with artifacts removed
    """
    try:
        # Check if accelerometer data is available
        acc_cols = [col for col in data.columns if 'acc' in col.lower()]
        if len(acc_cols) == 0:
            print("No accelerometer data found. Skipping artifact removal.")
            return data
            
        # Calculate magnitude of acceleration
        if len(acc_cols) >= 3:
            acc_mag = np.sqrt(
                data[acc_cols[0]]**2 + 
                data[acc_cols[1]]**2 + 
                data[acc_cols[2]]**2
            )
        else:
            acc_mag = data[acc_cols[0]].abs()
            
        # Identify periods with low movement (likely clean data)
        static_periods = acc_mag < acc_threshold
        
        # Return data during static periods
        clean_data = data[static_periods].copy()
        
        return clean_data
    except Exception as e:
        print(f"Error removing artifacts: {e}")
        return data