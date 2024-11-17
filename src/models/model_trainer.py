import os
import joblib
from datetime import datetime
import warnings
import pandas as pd
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


class SVMWithProba(LinearSVC):
    """SVM with probability estimates"""

    def predict_proba(self, X):
        decision = self.decision_function(X)
        if len(decision.shape) == 1:
            decision = np.column_stack([-decision, decision])
        probs = 1 / (1 + np.exp(-decision))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


class EnhancedModelTrainer:
    """Enhanced model trainer with ensemble learning and performance monitoring."""

    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.logger = Logger(__name__).logger
        self.checkpoint_dir = os.path.join(config.DATA_DIR, "checkpoints")
        self.models_dir = os.path.join(config.DATA_DIR, "models")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.training_time = 0
        self.feature_extractor = None  # Initialize feature extractor as None
        self.param_grid = config.PARAM_GRID
        self.model_config = config.MODEL_TRAINING_CONFIG
        self.regularization_config = config.REGULARIZATION_CONFIG
        self.validation_config = config.VALIDATION_CONFIG
        self.scoring_config = config.SCORING_CONFIG

    def create_ensemble_model(self):
        """Create model ensemble with documented algorithms"""

        rf = RandomForestClassifier(
            n_estimators=self.param_grid["rf__n_estimators"][0],
            max_depth=self.param_grid["rf__max_depth"][0],
            min_samples_split=self.model_config["preprocessing"]["min_df"],
            class_weight=self.model_config["class_weight_method"],
            ccp_alpha=self.regularization_config["rf_reg"]["ccp_alpha"],
            max_samples=self.regularization_config["rf_reg"]["max_samples"],
            random_state=42,
        )

        svm = SVMWithProba(
            C=self.param_grid["svm__C"][0],
            max_iter=1000,
            class_weight="balanced",
            kernel=self.regularization_config["svm_reg"]["kernel"],
            shrinking=self.regularization_config["svm_reg"]["shrinking"],
            dual=False,
        )

        nb = MultinomialNB(alpha=self.param_grid["nb__alpha"][0], fit_prior=True)

        models = [
            ("rf", Pipeline([("scaler", MinMaxScaler()), ("rf", rf)])),
            ("svm", Pipeline([("scaler", MinMaxScaler()), ("svm", svm)])),
            ("nb", Pipeline([("scaler", MinMaxScaler()), ("nb", nb)])),
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
            "metrics": metrics,  # Save complete metrics including training history
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
        """Visualizes training progress and performance metrics"""
        try:
            # Create figure with 3 subplots
            fig = plt.figure(figsize=(20, 6))

            # 1. Model Performance Comparison
            ax1 = plt.subplot(131)
            scores = []
            names = []

            for name, model in grid_search.items():
                # Get all available scores
                train_f1 = np.mean(model.cv_results_["mean_train_f1"])
                val_f1 = np.mean(model.cv_results_["mean_test_f1"])
                test_score = None
                if X_test is not None and y_test is not None:
                    test_score = f1_score(
                        y_test, model.predict(X_test), average="weighted"
                    )

                scores.append(
                    {"train": train_f1, "validation": val_f1, "test": test_score}
                )
                names.append(name)

            # Plot grouped bar chart
            x = np.arange(len(names))
            width = 0.25

            ax1.bar(
                x - width,
                [s["train"] for s in scores],
                width,
                label="Train",
                color="skyblue",
            )
            ax1.bar(
                x,
                [s["validation"] for s in scores],
                width,
                label="Validation",
                color="lightgreen",
            )
            if all(s["test"] is not None for s in scores):
                ax1.bar(
                    x + width,
                    [s["test"] for s in scores],
                    width,
                    label="Test",
                    color="salmon",
                )

            ax1.set_ylabel("F1 Score")
            ax1.set_title("Model Performance Comparison")
            ax1.set_xticks(x)
            ax1.set_xticklabels(names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Learning Curves
            ax2 = plt.subplot(132)
            for name, model in grid_search.items():
                train_scores = model.cv_results_["mean_train_f1"]
                val_scores = model.cv_results_["mean_test_f1"]
                epochs = range(1, len(train_scores) + 1)

                ax2.plot(epochs, train_scores, "o-", label=f"{name}_train", alpha=0.7)
                ax2.plot(epochs, val_scores, "s--", label=f"{name}_val", alpha=0.7)

            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("F1 Score")
            ax2.set_title("Learning Curves")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)

            # 3. Model Performance Details
            ax3 = plt.subplot(133)
            details = []
            metrics = ["precision", "recall", "f1"]

            for name, model in grid_search.items():
                row = [name]
                for metric in metrics:
                    train_score = np.mean(model.cv_results_[f"mean_train_{metric}"])
                    val_score = np.mean(model.cv_results_[f"mean_test_{metric}"])
                    row.extend([train_score, val_score])
                details.append(row)

            # Create table
            cell_text = [
                [f"{x:.3f}" if isinstance(x, float) else x for x in row]
                for row in details
            ]
            columns = ["Model"] + sum([[f"{m}_train", f"{m}_val"] for m in metrics], [])
            table = ax3.table(cellText=cell_text, colLabels=columns, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax3.axis("off")
            ax3.set_title("Detailed Metrics")

            plt.tight_layout()
            return fig

        except Exception as e:
            self.logger.error(f"Error plotting training progress: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return None

    def _plot_learning_curves(self, grid_search):
        """Plot learning curves showing training vs validation performance"""
        if isinstance(grid_search, dict):
            for name, model in grid_search.items():
                if hasattr(model, "cv_results_"):
                    train_scores = model.cv_results_["mean_train_f1"]
                    valid_scores = model.cv_results_["mean_test_f1"]
                    iterations = range(1, len(train_scores) + 1)

                    plt.plot(
                        iterations, train_scores, "o-", label=f"{name}_train", alpha=0.8
                    )
                    plt.plot(
                        iterations, valid_scores, "s-", label=f"{name}_val", alpha=0.8
                    )

            plt.title("Learning Curves")
            plt.xlabel("Parameter Combination")
            plt.ylabel("F1 Score")
            plt.legend(loc="center right")
            plt.grid(True)

    def train_with_grid_search(self, X_train, y_train):
        """Train with documented evaluation metrics"""
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

            # Simple parameter grid with corrected keys
            self.param_grid = {
                "rf__n_estimators": [300],
                "rf__max_depth": [30],
                "svm__C": [1.0],
                "nb__alpha": [0.1],
            }

            # Update scorers to handle different model types safely
            scorers = {
                "f1": make_scorer(f1_score, average="weighted"),
                "precision": make_scorer(
                    precision_score, average="weighted", zero_division=1
                ),
                "recall": make_scorer(
                    recall_score, average="weighted", zero_division=1
                ),
            }

            # K-fold CV as documenteds
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            if isinstance(y_train, pd.Series):
                y_train = y_train.to_numpy()
            if isinstance(X_train, pd.Series):
                X_train = X_train.to_numpy()

            X_train_features = self.feature_extractor.extract_features(X_train)

            # Initialize dictionaries to store training history
            training_history = {}

            # Train models
            models = self.create_ensemble_model()
            best_models = {}
            best_metrics = {}

            # Update grid search to use decision function for SVM
            for name, pipeline in models:
                try:
                    self.logger.info(f"\nTraining {name} model...")

                    # Get training scores for each fold
                    num_epochs = 5  # Number of CV folds
                    train_scores = []
                    valid_scores = []

                    # Create cross-validation splits
                    cv = StratifiedKFold(
                        n_splits=num_epochs, shuffle=True, random_state=42
                    )

                    # Manual cross-validation loop
                    for fold, (train_idx, val_idx) in enumerate(
                        cv.split(X_train_features, y_train)
                    ):
                        # Split data - using numpy indexing
                        X_train_fold = X_train_features[train_idx]
                        X_val_fold = X_train_features[val_idx]
                        y_train_fold = y_train[train_idx]
                        y_val_fold = y_train[val_idx]

                        # Train model
                        pipeline.fit(X_train_fold, y_train_fold)

                        # Get scores using weighted F1
                        train_score = f1_score(
                            y_train_fold,
                            pipeline.predict(X_train_fold),
                            average="weighted",
                        )
                        val_score = f1_score(
                            y_val_fold, pipeline.predict(X_val_fold), average="weighted"
                        )

                        train_scores.append(train_score)
                        valid_scores.append(val_score)

                        self.logger.info(
                            f"Fold {fold+1}/{num_epochs} - "
                            f"Train: {train_score:.4f}, Val: {val_score:.4f}"
                        )

                    # Store training history
                    training_history[name] = {
                        "train_scores": train_scores,
                        "valid_scores": valid_scores,
                        "epochs": range(1, num_epochs + 1),
                    }

                    # Final training on full dataset
                    pipeline.fit(X_train_features, y_train)
                    best_models[name] = pipeline
                    best_metrics[name] = {
                        "best_score": np.max(valid_scores),
                        "training_time": (datetime.now() - start_time).total_seconds(),
                        "train_scores": train_scores,
                        "valid_scores": valid_scores,
                        "epochs": range(1, num_epochs + 1),
                    }

                    self.logger.info(
                        f"{name} Final Scores:\n"
                        f"Best validation score: {np.max(valid_scores):.4f}\n"
                        f"Final training score: {train_scores[-1]:.4f}"
                    )

                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue

            if not best_models:
                raise ValueError("No models were successfully trained")

            # Save final model
            final_metrics = {
                "models": best_metrics,
                "total_time": (datetime.now() - start_time).total_seconds(),
                "training_history": training_history,  # Include full training history
                "feature_importance": getattr(
                    best_models["rf"], "feature_importances_", None
                ),
                "validation_scores": {
                    "precision": self.scoring_config["precision_zero_division"]
                },
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
