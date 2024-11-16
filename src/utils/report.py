import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.model_predictor import SentimentPredictor
from ..utils.logger import Logger
import joblib


class ReportGenerator:
    """
    Generates detailed analysis reports for sentiment model performance
    """

    def __init__(self, language: str):
        self.language = language
        self.logger = Logger(__name__).logger

    def get_model_info(self):
        """Get current model information and metrics"""
        try:
            model_path = os.path.join(
                "data", "models", f"{self.language}_sentiment_model.pkl"
            )
            model_info = joblib.load(model_path)
            return model_info
        except Exception as e:
            self.logger.error(f"Error loading model info: {e}")
            return None

    def generate_metrics_plots(self, metrics, output_dir):
        """Generate performance metric visualizations"""
        plt.figure(figsize=(15, 10))

        # Confusion matrix
        plt.subplot(2, 2, 1)
        if "confusion_matrix" in metrics:
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d")
            plt.title("Confusion Matrix")

        # ROC curve
        plt.subplot(2, 2, 2)
        if "roc_curve" in metrics:
            fpr, tpr, _ = metrics["roc_curve"]
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

        # Score distribution
        plt.subplot(2, 2, 3)
        if "score_distribution" in metrics:
            sns.histplot(data=metrics["score_distribution"])
            plt.title("Score Distribution")

        # Error analysis
        plt.subplot(2, 2, 4)
        if "error_analysis" in metrics:
            errors = pd.DataFrame(metrics["error_analysis"])
            sns.barplot(x="error_type", y="count", data=errors)
            plt.title("Error Analysis")
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
        plt.close()

    def generate_html_report(self, model_info, metrics, output_file):
        """Generate HTML report with metrics and visualizations"""
        html = """
        <html>
        <head>
            <title>Sentiment Analysis Model Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { margin: 20px 0; }
                .visualization { margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f4f4f4; }
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Model Report</h1>
            <div class="metadata">
                <h2>Model Information</h2>
                <table>
                    <tr><th>Language</th><td>{language}</td></tr>
                    <tr><th>Generated</th><td>{timestamp}</td></tr>
                    <tr><th>Model Version</th><td>{version}</td></tr>
                </table>
            </div>

            <div class="metrics">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {metrics_rows}
                </table>
            </div>

            <div class="visualization">
                <h2>Visualizations</h2>
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
        </body>
        </html>
        """

        # Format metrics rows
        metrics_rows = ""
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"

        # Fill template
        report = html.format(
            language=self.language.upper(),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            version=model_info.get("version", "N/A"),
            metrics_rows=metrics_rows,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

    def generate_report(self, output_file):
        """Generate complete model analysis report"""
        try:
            # Create output directory
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Get model info and metrics
            model_info = self.get_model_info()
            if not model_info:
                raise ValueError("Could not load model information")

            metrics = model_info.get("metrics", {})

            # Generate visualizations
            self.generate_metrics_plots(metrics, output_dir)

            # Generate HTML report
            self.generate_html_report(model_info, metrics, output_file)

            self.logger.info(f"Report generated successfully at {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return False
