"""
Streamlit app for real-time stress detection demo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import io
import base64

# Import our modules (assuming they're installed)
try:
    from src.preprocessing import preprocess_ecg, preprocess_eda, segment_signals
    from src.features import extract_all_features
    from src.models import StressDetectionModel
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.warning("Required modules not available. Please install dependencies.")

def main():
    st.title("Real-Time Stress Detection System")
    st.markdown("""
    This application demonstrates a machine learning system for detecting stress 
    from physiological signals such as ECG, EDA, and accelerometer data.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the mode",
        ["Home", "Data Upload", "Real-time Simulation", "Model Information"]
    )
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Data Upload":
        show_data_upload()
    elif app_mode == "Real-time Simulation":
        show_real_time_simulation()
    elif app_mode == "Model Information":
        show_model_info()

def show_home():
    st.header("Welcome to the Stress Detection System")
    
    st.subheader("How it works")
    st.markdown("""
    1. **Data Collection**: Physiological signals are collected from wearable sensors
    2. **Preprocessing**: Signals are filtered and cleaned to remove noise
    3. **Feature Extraction**: Relevant features are extracted from the signals
    4. **Classification**: A machine learning model predicts stress levels
    5. **Visualization**: Results are displayed in an easy-to-understand format
    """)
    
    st.subheader("Supported Sensors")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**ECG (Electrocardiogram)**")
        st.markdown("Measures heart activity and heart rate variability")
    
    with col2:
        st.markdown("**EDA (Electrodermal Activity)**")
        st.markdown("Measures skin conductance related to emotional responses")
    
    with col3:
        st.markdown("**Accelerometer**")
        st.markdown("Measures physical activity and movement")
    
    st.subheader("About Stress Detection")
    st.markdown("""
    Stress detection using physiological signals is based on the body's natural 
    responses to stress. When a person is stressed, their:
    - Heart rate variability changes
    - Skin conductance increases
    - Movement patterns may change
    
    This system uses machine learning to identify these patterns and classify 
    stress levels in real-time.
    """)

def show_data_upload():
    st.header("Upload Physiological Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with physiological data", 
        type="csv"
    )
    
    if uploaded_file is not None:
        # Read the file
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Display basic info
            st.subheader("Data Overview")
            st.write(f"Dataset shape: {data.shape}")
            st.write("Columns:", list(data.columns))
            st.dataframe(data.head())
            
            # Process data if modules are available
            if MODULES_AVAILABLE:
                if st.button("Process Data and Predict Stress"):
                    with st.spinner("Processing data..."):
                        # Segment data into windows
                        windows = segment_signals(data, window_size=256, overlap=0.5)
                        
                        # Extract features for each window
                        feature_list = []
                        for window in windows:
                            features = extract_all_features(window)
                            feature_list.append(features)
                        
                        if feature_list:
                            features_df = pd.concat(feature_list, ignore_index=True)
                            
                            # For demo purposes, we'll use a dummy model
                            # In practice, you would load a trained model
                            st.subheader("Stress Prediction Results")
                            
                            # Simulate predictions
                            if 'label' in features_df.columns:
                                # If we have actual labels, show comparison
                                predictions = np.random.choice([0, 1], size=len(features_df))
                                features_df['predicted_stress'] = predictions
                                
                                # Calculate accuracy
                                accuracy = np.mean(features_df['label'] == features_df['predicted_stress'])
                                st.write(f"Accuracy: {accuracy:.2f}")
                            else:
                                # Just show predictions
                                predictions = np.random.choice([0, 1], size=len(features_df))
                                confidence = np.random.uniform(0.5, 1.0, size=len(features_df))
                                features_df['predicted_stress'] = predictions
                                features_df['confidence'] = confidence
                            
                            # Display results
                            st.dataframe(features_df[['predicted_stress', 'confidence']].head(10))
                            
                            # Visualization
                            st.subheader("Stress Level Distribution")
                            fig, ax = plt.subplots()
                            stress_counts = features_df['predicted_stress'].value_counts()
                            ax.bar(['No Stress', 'Stress'], [stress_counts.get(0, 0), stress_counts.get(1, 0)])
                            ax.set_ylabel('Number of Windows')
                            ax.set_title('Predicted Stress Levels')
                            st.pyplot(fig)
                            
                            # Show confidence distribution
                            st.subheader("Prediction Confidence")
                            fig, ax = plt.subplots()
                            ax.hist(features_df['confidence'], bins=20)
                            ax.set_xlabel('Confidence')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Prediction Confidence')
                            st.pyplot(fig)
            else:
                st.warning("Processing modules not available. Please install dependencies to process data.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

def show_real_time_simulation():
    st.header("Real-time Stress Detection Simulation")
    
    st.markdown("""
    This simulation demonstrates how the system would work with real-time data streams.
    In a real implementation, this would connect to wearable sensors.
    """)
    
    # Simulation controls
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Simulation"):
            st.session_state.simulation_running = True
    
    with col2:
        if st.button("Stop Simulation"):
            st.session_state.simulation_running = False
    
    if st.session_state.simulation_running:
        st.subheader("Live Data Stream")
        
        # Simulate incoming data
        time_points = np.arange(0, 10, 0.1)
        ecg_signal = np.sin(2 * np.pi * 1.5 * time_points) + 0.5 * np.random.normal(size=len(time_points))
        eda_signal = 5 + 2 * np.sin(2 * np.pi * 0.2 * time_points) + np.random.normal(size=len(time_points))
        
        # Plot signals
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(time_points, ecg_signal)
        ax1.set_title("Simulated ECG Signal")
        ax1.set_ylabel("Amplitude")
        
        ax2.plot(time_points, eda_signal)
        ax2.set_title("Simulated EDA Signal")
        ax2.set_ylabel("Conductance (µS)")
        ax2.set_xlabel("Time (s)")
        
        st.pyplot(fig)
        
        # Simulate stress prediction
        stress_probability = np.random.uniform(0, 1)
        stress_level = "Stress" if stress_probability > 0.5 else "No Stress"
        confidence = stress_probability if stress_probability > 0.5 else 1 - stress_probability
        
        # Display result with color coding
        st.subheader("Current Stress Prediction")
        if stress_level == "Stress":
            st.error(f"**{stress_level}** (Confidence: {confidence:.2f})")
        else:
            st.success(f"**{stress_level}** (Confidence: {confidence:.2f})")
        
        # Show important features (simulated)
        st.subheader("Key Physiological Indicators")
        indicators = pd.DataFrame({
            'Indicator': ['Heart Rate Variability', 'Skin Conductance', 'Movement Activity'],
            'Value': [np.random.uniform(20, 80), np.random.uniform(3, 10), np.random.uniform(0, 5)],
            'Unit': ['ms', 'µS', 'mg']
        })
        st.dataframe(indicators)
        
        # Refresh every few seconds
        st.rerun()

def show_model_info():
    st.header("Machine Learning Model Information")
    
    st.subheader("Model Architecture")
    st.markdown("""
    The stress detection system uses ensemble machine learning models trained on 
    physiological features to classify stress levels.
    """)
    
    st.subheader("Features Used")
    features = [
        "**Heart Rate Variability (HRV)**: SDNN, RMSSD, pNN50, LF/HF ratio",
        "**Electrodermal Activity**: Mean, standard deviation, peak count",
        "**Accelerometer Data**: Movement patterns, activity levels",
        "**Statistical Features**: Mean, standard deviation, skewness, kurtosis"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.subheader("Model Performance")
    st.markdown("""
    Typical performance metrics on benchmark datasets:
    - **Accuracy**: 85-95%
    - **Precision**: 80-90%
    - **Recall**: 80-90%
    - **F1-Score**: 82-92%
    """)
    
    # Show confusion matrix (simulated)
    st.subheader("Confusion Matrix (Example)")
    confusion_data = pd.DataFrame(
        [[85, 15], [20, 80]], 
        columns=['Predicted No Stress', 'Predicted Stress'],
        index=['Actual No Stress', 'Actual Stress']
    )
    st.dataframe(confusion_data)
    
    st.subheader("Model Interpretability")
    st.markdown("""
    Feature importance analysis shows that the most important indicators of stress are:
    1. LF/HF ratio (heart rate variability)
    2. Skin conductance levels
    3. RMSSD (heart rate variability)
    4. Movement activity
    """)

if __name__ == "__main__":
    main()