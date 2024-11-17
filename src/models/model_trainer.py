import os
import joblib
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    mutual_info_classif,
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import Logger
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
    balanced_accuracy_score,
    roc_auc_score,
)
import numpy as np


class EnhancedModelTrainer:
    """
    Enhanced model trainer with ensemble learning and performance monitoring.

    Attributes:
        language (str): Language code ('en' or 'vi')
        config: Configuration object containing model parameters
        logger: Logger instance for tracking training progress
    """

    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.logger = Logger(__name__).logger
        self.checkpoint_dir = os.path.join(config.DATA_DIR, "checkpoints")
        self.models_dir = os.path.join(config.DATA_DIR, "models")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.training_time = 0
        self.feature_extractor = None  # Initialize as None
        self.param_grid = {
            "rf__n_estimators": [500, 1000],  # Increased trees significantly
            "rf__max_depth": [50, 100],  # Deeper trees
            "rf__min_samples_split": [2, 3],  # More precise splits
            "rf__min_samples_leaf": [1],  # Allow smaller leaf size
            "rf__max_features": ["sqrt", "log2"],  # Add feature selection options
            "rf__class_weight": [
                "balanced",
                "balanced_subsample",
            ],  # Add subsample option
            "svm__C": [0.1, 1.0, 10.0, 100.0],  # More regularization options
            "svm__tol": [1e-4, 1e-5],  # More precise convergence
            "svm__max_iter": [5000],  # More iterations for convergence
            "svm__class_weight": ["balanced"],
            "svm__dual": [False],
            "nb__alpha": [0.01, 0.1, 0.5],  # More smoothing options
            "nb__fit_prior": [True],
            "feature_selection__k": [500, 1000],  # More features
        }

    def create_ensemble_model(self):
        """Create simplified model ensemble with core models only"""
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier

        # Simplified model configurations
        models = [
            ("rf", Pipeline([
                ("scaler", MinMaxScaler()),
                ("rf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42,
                    class_weight='balanced'
                )),
            ])),
            ("lgbm", Pipeline([
                ("scaler", MinMaxScaler()),
                ("lgbm", LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    learning_rate=0.1,
                    max_depth=10,
                    device="cpu",
                    objective='multiclass',
                    metric='multi_logloss',
                    num_class=3,
                    verbose=-1
                )),
            ])),
        ]
        return models

    def save_checkpoint(self, model, metrics, epoch=None):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.language}_checkpoint_{timestamp}.pkl"
        )

        checkpoint = {
            "model_state": model,
            "metrics": metrics,
            "epoch": epoch,
            "timestamp": timestamp,
            "language": self.language,
        }

        joblib.dump(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Keep only last 5 checkpoints
        checkpoints = sorted(
            [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith(f"{self.language}_checkpoint")
            ]
        )
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

    def save_final_model(self, model, metrics):
        """Save the final trained model with all feature extractors"""
        model_path = os.path.join(
            self.models_dir, f"{self.language}_sentiment_model.pkl"
        )

        # Ensure metrics are properly formatted
        model_metrics = {
            "models": {},
            "total_time": metrics.get("total_time", self.training_time),
        }

        # Format metrics for each model type
        if isinstance(model, dict):
            for model_name, model_obj in model.items():
                model_metrics["models"][model_name] = {
                    "best_score": metrics.get(
                        "test_score", 0.0
                    ),  # Default to test_score if available
                    "training_time": self.training_time,
                    "parameters": getattr(model_obj, "get_params", lambda: {})(),
                }

                # Add additional metrics if available
                if model_name in metrics.get("models", {}):
                    model_metrics["models"][model_name].update(
                        metrics["models"][model_name]
                    )

                # Add feature importance for RF model
                if model_name == "rf" and hasattr(model_obj, "feature_importances_"):
                    model_metrics["models"][model_name][
                        "feature_importance"
                    ] = model_obj.feature_importances_.tolist()

        model_info = {
            "model": model,
            "metrics": model_metrics,
            "feature_extractor": {
                "vectorizer": self.feature_extractor.tfidf,
                "svd": self.feature_extractor.svd,
                "scaler": self.feature_extractor.scaler,
                "word_vectorizer": self.feature_extractor.word_vectorizer,
                "char_vectorizer": self.feature_extractor.char_vectorizer,
                "feature_dims": self.feature_extractor.feature_dims,
            },
            "config": {
                "language": self.language,
                "max_features": self.config.MAX_FEATURES,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            },
        }

        joblib.dump(model_info, model_path)
        self.logger.info(f"Saved final model to {model_path}")

    def plot_training_progress(self, grid_search, X_test=None, y_test=None):
        """Visualizes the training progress and model performance"""
        plt.figure(figsize=(12, 6))

        try:
            # Check if input is a dictionary of models
            if isinstance(grid_search, dict):
                # Plot model scores
                scores = []
                model_names = []
                training_times = []

                for model_name, model_obj in grid_search.items():
                    # Get scores from either metrics or model attributes
                    score = None
                    if hasattr(model_obj, "best_score_"):
                        score = model_obj.best_score_
                    else:
                        # Try get score from model's predict method
                        try:
                            score = model_obj.score(X_test, y_test)
                        except:
                            pass

                    if score is not None:
                        scores.append(score)
                        model_names.append(model_name)
                        training_times.append(getattr(model_obj, "fit_time_", 0))

                # Plot performance comparison
                plt.subplot(1, 2, 1)
                bars = plt.bar(range(len(scores)), scores)
                plt.xticks(range(len(model_names)), model_names, rotation=45)
                plt.title("Model Performance Comparison")
                plt.xlabel("Models")
                plt.ylabel("Score")
                plt.ylim(0, 1)

                # Add score labels
                for bar, score in zip(bars, scores):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        score + 0.01,
                        f"{score:.3f}",
                        ha="center",
                    )

                # Plot training times
                plt.subplot(1, 2, 2)
                bars = plt.bar(range(len(training_times)), training_times)
                plt.xticks(range(len(model_names)), model_names, rotation=45)
                plt.title("Training Time Comparison")
                plt.xlabel("Models")
                plt.ylabel("Time (s)")

                # Add time labels
                for bar, time in zip(bars, training_times):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        time + max(training_times) * 0.02,
                        f"{time:.1f}s",
                        ha="center",
                    )

            # Handle GridSearchCV object case
            elif hasattr(grid_search, "cv_results_"):
                # ... existing GridSearchCV plotting code ...
                pass

            plt.tight_layout()

        except Exception as e:
            self.logger.error(f"Error plotting training progress: {str(e)}")
            import traceback

            print("Full error traceback:")
            print(traceback.format_exc())

    def train_with_grid_search(self, X_train, y_train):
        """Simplified training process"""
        start_time = datetime.now()
        self.logger.info("Starting model training...")

        try:
            # Basic feature extraction
            if self.feature_extractor is None:
                from src.features.feature_engineering import FeatureExtractor
                self.feature_extractor = FeatureExtractor(self.language, self.config)

            # Extract and validate features
            X_train_features = self.feature_extractor.extract_features(X_train)
            if X_train_features is None or X_train_features.shape[0] == 0:
                raise ValueError("Feature extraction failed")

            # Simple parameter grid
            self.param_grid = {
                "rf__n_estimators": [100],
                "rf__max_depth": [10],
                "lgbm__n_estimators": [100],
                "lgbm__num_leaves": [31]
            }

            # Basic cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # Train models
            models = self.create_ensemble_model()
            best_models = {}
            best_metrics = {}

            for name, pipeline in models:
                try:
                    self.logger.info(f"\nTraining {name} model...")
                    
                    # Simple grid search
                    grid_search = GridSearchCV(
                        pipeline,
                        {k: v for k, v in self.param_grid.items() if k.startswith(name)},
                        cv=cv,
                        n_jobs=-1,
                        verbose=1
                    )

                    # Train model
                    grid_search.fit(X_train_features, y_train)

                    # Save best model and metrics
                    best_models[name] = grid_search.best_estimator_
                    best_metrics[name] = {
                        "best_score": grid_search.best_score_,
                        "training_time": (datetime.now() - start_time).total_seconds()
                    }

                    self.logger.info(f"{name} Best score: {grid_search.best_score_:.4f}")

                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue

            if not best_models:
                raise ValueError("No models were successfully trained")

            # Save final model
            final_metrics = {
                "models": best_metrics,
                "total_time": (datetime.now() - start_time).total_seconds()
            }
            self.save_final_model(best_models, final_metrics)

            return best_models

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return None

    def list_checkpoints(self):
        """List all available checkpoints with better metrics handling"""
        checkpoints = []
        try:
            for file in os.listdir(self.checkpoint_dir):
                if file.startswith(f"{self.language}_checkpoint"):
                    checkpoint_path = os.path.join(self.checkpoint_dir, file)
                    info = joblib.load(checkpoint_path)
                    metrics_value = None

                    # Safely extract metrics
                    if "metrics" in info:
                        if isinstance(info["metrics"], dict):
                            metrics_value = info["metrics"].get("best_score")
                        else:
                            metrics_value = str(info["metrics"])

                    checkpoints.append(
                        {
                            "filename": file,
                            "timestamp": info.get("timestamp", "Unknown"),
                            "epoch": info.get("epoch", "Unknown"),
                            "metrics": metrics_value,
                        }
                    )
            return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            self.logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def restore_from_checkpoint(self, checkpoint_name=None):
        """Restore model from checkpoint"""
        try:
            if checkpoint_name is None:
                # Get latest checkpoint
                checkpoints = sorted(
                    [
                        f
                        for f in os.listdir(self.checkpoint_dir)
                        if f.startswith(f"{self.language}_checkpoint")
                    ]
                )
                if not checkpoints:
                    raise ValueError("No checkpoints found")
                checkpoint_name = checkpoints[-1]

            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = joblib.load(checkpoint_path)
            self.logger.info(f"Restored model from checkpoint: {checkpoint_name}")
            return checkpoint["model_state"], checkpoint["metrics"]

        except Exception as e:
            self.logger.error(f"Error restoring checkpoint: {str(e)}")
            return None, None

    def optimize_hyperparameters(self, X_train=None, y_train=None):
        """Optimize hyperparameters using cross-validation"""
        self.logger.info("Starting hyperparameter optimization...")

        try:
            # Use stored data if not provided
            if X_train is None or y_train is None:
                # Load last checkpoint to get data
                checkpoints = self.list_checkpoints()
                if not checkpoints:
                    raise ValueError("No checkpoints found and no data provided")
                checkpoint = joblib.load(
                    os.path.join(self.checkpoint_dir, checkpoints[0]["filename"])
                )
                X_train = checkpoint.get("X_train")
                y_train = checkpoint.get("y_train")

            if X_train is None or y_train is None:
                raise ValueError("No training data available")

            # Extract features
            if self.feature_extractor is None:
                from src.features.feature_engineering import FeatureExtractor

                self.feature_extractor = FeatureExtractor(self.language, self.config)

            X_train_features = self.feature_extractor.extract_features(X_train)

            # Define expanded parameter grid for optimization
            expanded_param_grid = {
                "rf__n_estimators": [300, 500, 1000],
                "rf__max_depth": [30, 50, 100],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4],
                "rf__max_features": ["sqrt", "log2", None],
                "svm__C": [0.1, 1.0, 10.0],
                "svm__tol": [1e-4, 1e-5],
                "svm__max_iter": [3000, 5000],
                "nb__alpha": [0.01, 0.1, 0.5, 1.0],
                "nb__fit_prior": [True, False],
                "feature_selection__k": [300, 500, 1000],
            }

            # Configure cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.MODEL_TRAINING_CONFIG["cv_folds"],
                shuffle=True,
                random_state=42,
            )

            # Initialize models
            models = self.create_ensemble_model()
            best_params = {}

            # Optimize each model separately
            for name, pipeline in models:
                self.logger.info(f"\nOptimizing {name} model...")

                # Get relevant parameters for this model
                model_params = {
                    k: v for k, v in expanded_param_grid.items() if k.startswith(name)
                }

                # Configure scoring
                scorers = {
                    "f1": make_scorer(f1_score, average="weighted", zero_division=1),
                    "precision": make_scorer(
                        precision_score, average="weighted", zero_division=1
                    ),
                    "recall": make_scorer(
                        recall_score, average="weighted", zero_division=1
                    ),
                    "balanced_accuracy": make_scorer(balanced_accuracy_score),
                }

                # Perform grid search
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=model_params,
                    cv=cv,
                    scoring=scorers,
                    refit="balanced_accuracy",
                    n_jobs=-1,
                    verbose=1,
                )

                # Fit with sample weights if applicable
                sample_weights = compute_sample_weight("balanced", y_train)
                fit_params = {}
                if name in ["rf", "svm"]:
                    fit_params = {f"{name}__sample_weight": sample_weights}

                grid_search.fit(X_train_features, y_train, **fit_params)

                # Store best parameters
                best_params[name] = {
                    "params": grid_search.best_params_,
                    "score": grid_search.best_score_,
                }

                self.logger.info(f"Best {name} parameters: {grid_search.best_params_}")
                self.logger.info(f"Best {name} score: {grid_search.best_score_:.4f}")

            # Save optimized parameters
            optimization_path = os.path.join(
                self.config.DATA_DIR,
                "optimization",
                f"{self.language}_optimized_params.json",
            )
            os.makedirs(os.path.dirname(optimization_path), exist_ok=True)

            with open(optimization_path, "w") as f:
                import json

                json.dump(best_params, f, indent=4)

            self.logger.info(f"Saved optimized parameters to {optimization_path}")
            return best_params

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return None
