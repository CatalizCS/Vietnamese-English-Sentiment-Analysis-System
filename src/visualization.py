
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List
import numpy as np
from datetime import datetime
import joblib

class ModelVisualizer:
    def __init__(self, language: str):
        self.language = language
        self.plot_style = {
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }
        plt.style.use('seaborn')
        for key, value in self.plot_style.items():
            plt.rcParams[key] = value

    def generate_plots(self, output_dir: str):
        """Generate and save all visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load model metrics
        model_path = f"data/models/{self.language}_sentiment_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        metrics = model_data.get('metrics', {})

        # Generate various plots
        self._plot_performance_metrics(metrics, output_dir, timestamp)
        self._plot_confusion_matrix(metrics, output_dir, timestamp)
        self._plot_training_history(metrics, output_dir, timestamp)
        self._plot_feature_importance(metrics, output_dir, timestamp)
        
        return True

    def _plot_performance_metrics(self, metrics: Dict, output_dir: str, timestamp: str):
        """Plot key performance metrics"""
        fig, ax = plt.subplots()
        
        metrics_to_plot = {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1_score', 0)
        }
        
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        ax.set_ylim(0, 1)
        ax.set_title(f'{self.language.upper()} Model Performance Metrics')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, f'{timestamp}_performance_metrics.png'))
        plt.close()

    def _plot_confusion_matrix(self, metrics: Dict, output_dir: str, timestamp: str):
        """Plot confusion matrix heatmap"""
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots()
            
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'{self.language.upper()} Model Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            plt.savefig(os.path.join(output_dir, f'{timestamp}_confusion_matrix.png'))
            plt.close()

    def _plot_training_history(self, metrics: Dict, output_dir: str, timestamp: str):
        """Plot training history"""
        if 'training_history' in metrics:
            history = metrics['training_history']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            if 'accuracy' in history:
                ax1.plot(history['accuracy'], label='Training')
                if 'val_accuracy' in history:
                    ax1.plot(history['val_accuracy'], label='Validation')
                ax1.set_title('Model Accuracy Over Time')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
            
            # Plot loss
            if 'loss' in history:
                ax2.plot(history['loss'], label='Training')
                if 'val_loss' in history:
                    ax2.plot(history['val_loss'], label='Validation')
                ax2.set_title('Model Loss Over Time')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{timestamp}_training_history.png'))
            plt.close()

    def _plot_feature_importance(self, metrics: Dict, output_dir: str, timestamp: str):
        """Plot feature importance"""
        if 'feature_importance' in metrics and 'feature_names' in metrics:
            importance = metrics['feature_importance']
            features = metrics['feature_names']
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot top 20 features
            plt.figure(figsize=(10, 6))
            plt.title(f'Top 20 Most Important Features ({self.language.upper()})')
            plt.bar(range(20), importance[indices[:20]])
            plt.xticks(range(20), [features[i] for i in indices[:20]], rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{timestamp}_feature_importance.png'))
            plt.close()

    def visualize_prediction(self, text: str, prediction: Dict, output_file: str = None):
        """Create visualization for a single prediction"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot sentiment scores
        sentiment_scores = prediction.get('sentiment_scores', {})
        if sentiment_scores:
            ax1.bar(sentiment_scores.keys(), sentiment_scores.values())
            ax1.set_title('Sentiment Scores')
            ax1.set_ylim(0, 1)
            
        # Plot emotion scores
        emotion_scores = prediction.get('emotion_scores', {})
        if emotion_scores:
            emotions = list(emotion_scores.keys())
            scores = list(emotion_scores.values())
            ax2.bar(emotions, scores)
            ax2.set_title('Emotion Scores')
            ax2.set_ylim(0, 1)
            plt.xticks(rotation=45)
        
        plt.suptitle(f'Analysis Results for: "{text[:50]}..."')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()