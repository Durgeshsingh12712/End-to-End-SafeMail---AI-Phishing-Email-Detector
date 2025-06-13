import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report, roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from emailDetector.entity.config_entity import ModelEvaluationConfig
from emailDetector.entity.artifact_entity import ModelEvaluationArtifact
from emailDetector.logging.logger import logger
from emailDetector.exception.exception import EmailDetectionException
from emailDetector.utils.utils import load_object, save_json


class ModelEvaluation:
    """Advanced Model Evaluation Class with Comprehensive Metric and Visualization Support"""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.test_data = None
    
    def load_components(self) -> None:
        """Load Model, Vectorizer and Test Data"""
        try:
            logger.info("Loading Model Components")

            # Validate File Paths
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            if not os.path.exists(self.config.vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {self.config.vectorizer_path}")
            if not os.path.exists(self.config.test_data_path):
                raise FileNotFoundError(f"Test dat file not found: {self.config.test_data_path}")
            
            # Load Components
            self.model = load_object(file_path = self.config.model_path)
            self.vectorizer = load_object(file_path = self.config.vectorizer_path)
            self.test_data = pd.read_csv(self.config.test_data_path)

            # Validate Test Data Structure
            required_columns = ['text', 'target']
            missing_columns = [col for col in required_columns if col not in self.test_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in test data: {missing_columns}")
            
            logger.info(f"Loaded Test Data with {len(self.test_data)} samples")

        except Exception as e:
            raise EmailDetectionException(e, sys)
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and Transform Test Data."""
        try:
            logger.info("Preparing Test Data")

            # Handle Missing Values
            self.test_data['text'] = self.test_data['text'].fillna('')

            # Extract Feature and Targets
            X_test = self.test_data['text']
            y_test = self.test_data['target']

            # Transform Features
            X_test_transformed = self.vectorizer.transform(X_test)

            # Get Prediction Probabilities if available
            y_pred = self.model.predict(X_test_transformed)

            # Get Prediction and Probabilities
            y_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
            elif hasattr(self.model, 'decision_function'):
                y_pred_proba = self.model.decision_function(X_test_transformed)
            
            return y_test.values, y_pred, y_pred_proba
        
        except Exception as e:
            logger.error(f"Error Preparing Data: {str(e)}")
            raise EmailDetectionException(e, sys)
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate Basic Classification Metrics."""

        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_score_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            raise EmailDetectionException(e, sys)
        
    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray)-> Dict[str, Any]:
        """Calculate Confusion Matrix and derived Metrics"""
        try:
            cm = confusion_matrix(y_true, y_pred)

            # Handle Binary Classification
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                # Calculate Addictional metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
                
                return {
                    "confusion_matrix": {
                        "TP": int(tp),
                        "TN": int(tn),
                        "FP": int(fp),
                        "FN": int(fn)
                    },
                    "specificity": specificity,
                    "sensitivity": sensitivity,
                    "npv": npv,
                    "ppv": ppv,
                    "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                    "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0
                }
            else:
                # Multi-class confusion matrix
                return {
                    "confusion_matrix": cm.tolist(),
                    "confusion_matrix_shape": cm.shape
                }
                
        except Exception as e:
            logger.error(f"Error calculating confusion matrix metrics: {str(e)}")
            raise EmailDetectionException(e, sys)
        
    def calculate_probabilistic_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate Metric that require prediction probabilies"""

        if y_pred_proba is None:
            logger.warning("Prediction Probabilities not available. Skipping probabilistic metrics.")
            return {}
        
        try:
            metrics = {}
            # ROC AUC (for binary Classification)
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                metrics["average_precision"] = average_precision_score(y_true, y_pred_proba)
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Error Calculation Probabilistic metrics: {str(e)}")
            return {}
        
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate Detailed Classification Report."""
        try:
            # Determine Class Names
            unique_labels = sorted(np.unique(y_true))
            if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                target_names = ['safe', 'phishing']
            else:
                target_names = [f'class_{label}' for label in unique_labels]
            
            class_report = classification_report(
                y_true, y_pred,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            return class_report

        except Exception as e:
            logger.error(f"Error Generating Classifaction report : {str(e)}")
            raise EmailDetectionException(e, sys)
    
    def save_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[str]:
        """Save Confusion Matrix visualization"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)

            # Create HeatMap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['safe', 'phishing'], yticklabels=['safe', 'phishing'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Prediction Label')

            # Save plot
            plot_path = os.path.join(self.config.root_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
            plt.close()

            logger.info(f"Confusion Matrix plot saved to: {plot_path}")
            return plot_path
        
        except Exception as e:
            logger.warning(f"Could not save confusion matrix plot: {str(e)}")
            return None
        
    def evaluate_model_acceptance(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate whether the model meets acceptance criteria"""
        try:
            # Define Acceptance threshold
            min_accuracy = getattr(self.config, 'min_accuracy', 0.85)
            min_precission = getattr(self.config, 'min_precission', 0.80)
            min_recall = getattr(self.config, 'min_recall', 0.80)
            min_f1 = getattr(self.config, 'min_f1_score', 0.80)

            reasons = []

            # Check each criterion
            if metrics["accuracy"] < min_accuracy:
                reasons.append(f"Accuracy {metrics['accuracy']:.4f} < {min_accuracy}")
            if metrics['precision'] < min_precission:
                reasons.append(f"Precision {metrics['precision']:.4f} < {min_precission}")
            if metrics['recall'] < min_recall:
                reasons.append(f"Recall {metrics['recall']:.4f} < {min_recall}")
            if metrics['f1_score'] < min_f1:
                reasons.append(f"F1-Score {metrics['f1_score']:.4f} < {min_f1}")

            is_accepted = len(reasons) == 0
            rejection_reason = "; ".join(reasons) if reasons else "Model Meets all acceptance criteria"

            return is_accepted, rejection_reason
        
        except Exception as e:
            logger.error(f"Error Evaluation model Acceptance: {str(e)}")
            return False, f"Error During Evaluation: {str(e)}"
        
    
    def log_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Log a Summary of key metrics"""
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"Precision:    {metrics['precision']:.4f}")
        logger.info(f"Recall:       {metrics['recall']:.4f}")
        logger.info(f"F1-Score:     {metrics['f1_score']:.4f}")
        
        if 'specificity' in metrics:
            logger.info(f"Specificity:  {metrics['specificity']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC:      {metrics['roc_auc']:.4f}")
            
        logger.info("=" * 50)
    
    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Main Method to evaluate the model comprehensively.

        Returns:
            ModelEvaluationArtifacts: Evaluation Result and Metadata
        """
        try:
            logger.info("Starting Comprehensive Model Evaluation")

            # Load all required components
            self.load_components()

            #Prepare Data And get Predictions
            y_true, y_pred, y_pred_proba = self.prepare_data()

            # Calculate All Metrics
            basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
            cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
            prob_metrics = self.calculate_probabilistic_metrics(y_true, y_pred_proba)
            class_report = self.generate_classification_report(y_true, y_pred)

            # Combine all Metrics
            all_metrics = {
                **basic_metrics,
                **cm_metrics,
                **prob_metrics,
                "classification_report": class_report,
                "test_samples": len(y_true),
                "model_type": str(type(self.model).__name__)
            }

            # Save Confusion Matrix Plot
            plot_path = self.save_confusion_matrix_plot(y_true, y_pred)
            if plot_path:
                all_metrics["confusion_matrix_plot"] = plot_path

            # Save Metrics to Json File
            metric_file_path = Path(self.config.root_dir) / self.config.metric_file_name
            save_json(path=metric_file_path, data=all_metrics)
            logger.info(f"Metrics saved to: {metric_file_path}")

            # Log Summary
            self.log_metrics_summary(all_metrics)

            # Evaluate Model Acceptance
            is_accepted, acceptance_reason = self.evaluate_model_acceptance(all_metrics)
            logger.info(f"Model Acceptance: {'ACCEPTED' if is_accepted else 'REJECTED'}")
            logger.info(f"Reason: {acceptance_reason}")

            logger.info("Model Evaluation Completed Successfully")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                changed_accuracy=all_metrics["accuracy"],
                # s3_model_path = "",
                trained_model_path=self.config.model_path,
                evaluation_report_path=str(metric_file_path)
            )
        
        except Exception as e:
            logger.error(f"Model Evaluation Failed: {str(e)}")
            raise EmailDetectionException(e, sys)
