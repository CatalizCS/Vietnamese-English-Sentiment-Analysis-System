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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import Logger


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
            "rf__n_estimators": [200, 300],  # Increased trees
            "rf__max_depth": [20, 30],  # Deeper trees
            "rf__min_samples_split": [2, 5],  # More granular splits
            "rf__min_samples_leaf": [1, 2],  # More granular leaves
            "rf__class_weight": ["balanced"],
            "svm__C": [0.1, 1.0, 10.0],  # More C values
            "svm__tol": [1e-4],  # Tighter tolerance
            "svm__max_iter": [2000],  # More iterations
            "svm__class_weight": ["balanced"],
            "svm__dual": [False],  # Added dual parameter
            "nb__alpha": [0.1, 0.5, 1.0],  # More alpha values
            "nb__fit_prior": [True, False],
            "feature_selection__k": [200, 300],  # More features
        }

    def create_ensemble_model(self):
        """Creates an enhanced ensemble of models"""
        # Initialize basic pipelines with sample weight support
        rf_pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )

        svm_pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("feature_selection", SelectKBest(chi2)),
                ("svm", LinearSVC(random_state=42)),
            ]
        )

        nb_pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("feature_selection", SelectKBest(chi2)),
                ("nb", MultinomialNB()),
            ]
        )

        models = [("rf", rf_pipeline), ("svm", svm_pipeline), ("nb", nb_pipeline)]

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
        """Enhanced training with feature extractor validation"""
        start_time = datetime.now()
        self.logger.info("Starting model training...")

        try:
            # Initialize feature extractor if not set
            if self.feature_extractor is None:
                from src.features.feature_engineering import FeatureExtractor

                self.feature_extractor = FeatureExtractor(self.language, self.config)
                if self.feature_extractor is None:
                    raise ValueError("Failed to initialize feature extractor")

            # Extract features with validation
            X_train_features = self.feature_extractor.extract_features(X_train)
            if X_train_features is None or X_train_features.shape[0] == 0:
                raise ValueError(
                    "Feature extraction failed - empty or None features returned"
                )

            self.logger.info(f"Extracted features shape: {X_train_features.shape}")

            # Create and train models with optimized parameters
            models = self.create_ensemble_model()
            best_models = {}
            best_metrics = {}

            # Optimized cross-validation
            cv = StratifiedKFold(
                n_splits=5,  # 5-fold CV for better balance
                shuffle=True,
                random_state=42,
            )

            # Calculate balanced sample weights
            sample_weights = compute_sample_weight("balanced", y_train)

            # Train each model with optimized parameters
            for name, pipeline in models:
                try:
                    self.logger.info(f"\nTraining {name} model...")
                    model_params = {
                        k: v for k, v in self.param_grid.items() if k.startswith(name)
                    }

                    # Optimized GridSearchCV
                    grid_search = GridSearchCV(
                        pipeline,
                        param_grid=model_params,
                        cv=cv,
                        scoring={
                            "f1": "f1_weighted",
                            "precision": "precision_weighted",
                            "recall": "recall_weighted",
                        },
                        refit="f1",
                        n_jobs=-1,
                        verbose=1,
                        error_score="raise",
                    )

                    # Train with sample weights
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=Warning)
                        grid_search.fit(
                            X_train_features,
                            y_train,
                            **(
                                {"sample_weight": sample_weights}
                                if name != "nb"
                                else {}
                            ),
                        )

                    # Track metrics
                    model_time = (datetime.now() - start_time).total_seconds()
                    metrics = {
                        "best_score": grid_search.best_score_,
                        "best_params": grid_search.best_params_,
                        "cv_results": grid_search.cv_results_,
                        "training_time": model_time,
                    }

                    self.save_checkpoint(grid_search.best_estimator_, metrics, name)
                    best_models[name] = grid_search.best_estimator_
                    best_metrics[name] = metrics

                    self.logger.info(f"{name} Results:")
                    self.logger.info(f"Best score: {grid_search.best_score_:.4f}")

                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue

            # Save final model
            if best_models:
                final_metrics = {
                    "models": best_metrics,
                    "total_time": (datetime.now() - start_time).total_seconds(),
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
