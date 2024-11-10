from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Model evaluation utility with visualization capabilities.
    
    Attributes:
        language (str): Language code for the model being evaluated
    """
    def __init__(self, language: str):
        self.language = language

    def evaluate(self, y_true, y_pred, y_prob=None):
        """
        Evaluates model performance with multiple metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics and plots
        """
        results = {}
        
        # Detailed metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        results["classification_report"] = classification_report(y_true, y_pred)
        results["metrics"] = {
            "accuracy": report['accuracy'],
            "macro_f1": report['macro avg']['f1-score'],
            "weighted_f1": report['weighted avg']['f1-score']
        }
        
        # Plot metrics
        self._plot_metrics_summary(results["metrics"])

        # Basic classification metrics
        results["classification_report"] = classification_report(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm)

        if y_prob is not None:
            # ROC curve
            self.plot_roc_curve(y_true, y_prob)

        return results

    def _plot_metrics_summary(self, metrics):
        """Plots a summary of key performance metrics"""
        plt.figure(figsize=(10, 4))
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        plt.bar(metrics.keys(), metrics.values(), color=colors)
        plt.title(f'Model Performance Metrics - {self.language.upper()}')
        plt.ylim(0, 1)
        
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {self.language.upper()}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def plot_roc_curve(self, y_true, y_prob):
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.language.upper()}")
        plt.legend(loc="lower right")
        plt.show()
