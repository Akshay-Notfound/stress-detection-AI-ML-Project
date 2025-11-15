"""
Machine learning models for stress detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import Tuple, Dict, List, Optional
import joblib

class StressDetectionModel:
    """
    A class for stress detection using various machine learning models.
    """
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the stress detection model.
        
        Args:
            model_type (str): Type of model to use ('rf', 'xgb', 'svm', 'lr')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model based on type
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            self.model = XGBClassifier(random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose from 'rf', 'xgb', 'svm', 'lr'")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training (scaling).
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled features and labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
        """
        # Prepare data
        X_scaled, y_array = self.prepare_data(X, y)
        
        # Train model
        self.model.fit(X_scaled, y_array)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        
        return metrics
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            param_grid: Dict, cv: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            param_grid (dict): Parameter grid for grid search
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and score
        """
        # Prepare data
        X_scaled, y_array = self.prepare_data(X, y)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=cv, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y_array)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'StressDetectionModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            StressDetectionModel: Loaded model
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        model = cls(model_data['model_type'])
        model.model = model_data['model']
        model.scaler = model_data['scaler']
        model.is_fitted = model_data['is_fitted']
        
        return model

def compare_models(X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Dict:
    """
    Compare multiple models for stress detection.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Comparison results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define models to compare
    models = {
        'Random Forest': StressDetectionModel('rf'),
        'XGBoost': StressDetectionModel('xgb'),
        'Logistic Regression': StressDetectionModel('lr'),
        'SVM': StressDetectionModel('svm')
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
    
    return results

def leave_one_subject_out_cv(X: pd.DataFrame, y: pd.Series, 
                           subject_ids: pd.Series, 
                           model_type: str = 'rf') -> Dict:
    """
    Perform leave-one-subject-out cross-validation.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        subject_ids (pd.Series): Subject IDs for each sample
        model_type (str): Type of model to use
        
    Returns:
        dict: Cross-validation results
    """
    subjects = subject_ids.unique()
    scores = []
    
    for subject in subjects:
        # Split data: test on current subject, train on others
        train_idx = subject_ids != subject
        test_idx = subject_ids == subject
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Skip if no test samples
        if len(y_test) == 0:
            continue
            
        # Train model
        model = StressDetectionModel(model_type)
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        scores.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    metric_keys = scores[0].keys() if scores else []
    
    for key in metric_keys:
        if key != 'confusion_matrix':
            avg_metrics[key] = np.mean([score[key] for score in scores])
        else:
            avg_metrics[key] = np.mean([np.array(score[key]) for score in scores], axis=0).tolist()
    
    return {
        'individual_scores': scores,
        'average_scores': avg_metrics
    }