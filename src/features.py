"""
Feature extraction functions for stress detection from physiological signals.
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from typing import List, Dict, Optional
import pyhrv
import pyhrv.frequency_domain as fd
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl

def extract_time_domain_features(rr_intervals: np.ndarray) -> Dict:
    """
    Extract time-domain HRV features from RR intervals.
    
    Args:
        rr_intervals (np.ndarray): RR intervals in milliseconds
        
    Returns:
        dict: Time-domain HRV features
    """
    try:
        # Mean RR interval
        mean_rr = np.mean(rr_intervals)
        
        # Standard deviation of RR intervals (SDNN)
        sdnn = np.std(rr_intervals)
        
        # Root mean square of successive differences (RMSSD)
        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        
        # Number of pairs of adjacent RR intervals differing by > 50 ms (NN50)
        nn50 = np.sum(np.abs(diff_rr) > 50)
        
        # Proportion of NN50 divided by total number of RR intervals (pNN50)
        pnn50 = (nn50 / len(rr_intervals)) * 100
        
        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': nn50,
            'pnn50': pnn50
        }
    except Exception as e:
        print(f"Error extracting time-domain features: {e}")
        return {}

def extract_frequency_domain_features(rr_intervals: np.ndarray, 
                                   sampling_rate: float = 4.0) -> Dict:
    """
    Extract frequency-domain HRV features from RR intervals.
    
    Args:
        rr_intervals (np.ndarray): RR intervals in milliseconds
        sampling_rate (float): Sampling rate for interpolation (Hz)
        
    Returns:
        dict: Frequency-domain HRV features
    """
    try:
        # Interpolate RR intervals to a regular time grid
        time_intervals = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
        time_grid = np.arange(0, time_intervals[-1], 1/sampling_rate)
        interpolated_rr = np.interp(time_grid, time_intervals, rr_intervals)
        
        # Compute power spectral density using Welch's method
        freq, psd = signal.welch(interpolated_rr, fs=sampling_rate, nperseg=len(interpolated_rr)//2)
        
        # Define frequency bands
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate power in each band
        vlf_power = np.trapz(psd[(freq >= vlf_band[0]) & (freq < vlf_band[1])])
        lf_power = np.trapz(psd[(freq >= lf_band[0]) & (freq < lf_band[1])])
        hf_power = np.trapz(psd[(freq >= hf_band[0]) & (freq < hf_band[1])])
        
        # Total power
        total_power = vlf_power + lf_power + hf_power
        
        # Normalize powers
        lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        
        # LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_norm': lf_norm,
            'hf_norm': hf_norm,
            'lf_hf_ratio': lf_hf_ratio,
            'total_power': total_power
        }
    except Exception as e:
        print(f"Error extracting frequency-domain features: {e}")
        return {}

def extract_eda_features(eda_signal: np.ndarray) -> Dict:
    """
    Extract features from EDA signal.
    
    Args:
        eda_signal (np.ndarray): EDA signal data
        
    Returns:
        dict: EDA features
    """
    try:
        # Basic statistical features
        mean_eda = np.mean(eda_signal)
        std_eda = np.std(eda_signal)
        min_eda = np.min(eda_signal)
        max_eda = np.max(eda_signal)
        
        # Number of peaks (skin conductance responses)
        peaks, _ = signal.find_peaks(eda_signal, height=np.mean(eda_signal))
        num_peaks = len(peaks)
        
        # Amplitude of peaks
        peak_amplitudes = eda_signal[peaks] - np.mean(eda_signal)
        mean_peak_amp = np.mean(peak_amplitudes) if len(peak_amplitudes) > 0 else 0
        
        return {
            'mean_eda': mean_eda,
            'std_eda': std_eda,
            'min_eda': min_eda,
            'max_eda': max_eda,
            'num_scr_peaks': num_peaks,
            'mean_peak_amplitude': mean_peak_amp
        }
    except Exception as e:
        print(f"Error extracting EDA features: {e}")
        return {}

def extract_accelerometer_features(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Dict:
    """
    Extract features from accelerometer data.
    
    Args:
        acc_x (np.ndarray): X-axis acceleration
        acc_y (np.ndarray): Y-axis acceleration
        acc_z (np.ndarray): Z-axis acceleration
        
    Returns:
        dict: Accelerometer features
    """
    try:
        # Magnitude of acceleration
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Statistical features for each axis
        features = {}
        for axis_name, axis_data in [('x', acc_x), ('y', acc_y), ('z', acc_z), ('magnitude', acc_magnitude)]:
            features[f'mean_acc_{axis_name}'] = np.mean(axis_data)
            features[f'std_acc_{axis_name}'] = np.std(axis_data)
            features[f'min_acc_{axis_name}'] = np.min(axis_data)
            features[f'max_acc_{axis_name}'] = np.max(axis_data)
            features[f'rms_acc_{axis_name}'] = np.sqrt(np.mean(axis_data**2))
            
        return features
    except Exception as e:
        print(f"Error extracting accelerometer features: {e}")
        return {}

def extract_statistical_features(signal_data: np.ndarray) -> Dict:
    """
    Extract general statistical features from a signal.
    
    Args:
        signal_data (np.ndarray): Signal data
        
    Returns:
        dict: Statistical features
    """
    try:
        # Basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        median_val = np.median(signal_data)
        
        # Higher-order statistics
        skewness = stats.skew(signal_data)
        kurtosis = stats.kurtosis(signal_data)
        
        # Signal energy
        energy = np.sum(signal_data**2)
        
        # Root mean square
        rms = np.sqrt(np.mean(signal_data**2))
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'energy': energy,
            'rms': rms
        }
    except Exception as e:
        print(f"Error extracting statistical features: {e}")
        return {}

def extract_all_features(window_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all relevant features from a window of physiological data.
    
    Args:
        window_data (pd.DataFrame): Window of physiological data
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    try:
        features = {}
        
        # Extract ECG features if available
        ecg_cols = [col for col in window_data.columns if 'ecg' in col.lower()]
        if len(ecg_cols) > 0:
            # Assume the first ECG column contains the signal
            ecg_signal = window_data[ecg_cols[0]].values
            # For demonstration, we'll assume a fixed sampling rate
            sampling_rate = 256.0  # Hz
            
            # Process ECG to get RR intervals
            # In practice, you would use a proper R-peak detection algorithm here
            # For now, we'll simulate RR intervals
            rr_intervals = np.random.normal(800, 100, len(ecg_signal) // sampling_rate)
            
            time_features = extract_time_domain_features(rr_intervals)
            freq_features = extract_frequency_domain_features(rr_intervals)
            
            features.update(time_features)
            features.update(freq_features)
        
        # Extract EDA features if available
        eda_cols = [col for col in window_data.columns if 'eda' in col.lower()]
        if len(eda_cols) > 0:
            eda_signal = window_data[eda_cols[0]].values
            eda_features = extract_eda_features(eda_signal)
            features.update(eda_features)
        
        # Extract accelerometer features if available
        acc_cols = [col for col in window_data.columns if 'acc' in col.lower()]
        if len(acc_cols) >= 3:
            acc_x = window_data[acc_cols[0]].values
            acc_y = window_data[acc_cols[1]].values
            acc_z = window_data[acc_cols[2]].values
            acc_features = extract_accelerometer_features(acc_x, acc_y, acc_z)
            features.update(acc_features)
        
        # Add any label column if present
        label_cols = [col for col in window_data.columns if 'label' in col.lower() or 'stress' in col.lower()]
        if len(label_cols) > 0:
            features['label'] = window_data[label_cols[0]].iloc[0]
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        return features_df
        
    except Exception as e:
        print(f"Error extracting all features: {e}")
        return pd.DataFrame()