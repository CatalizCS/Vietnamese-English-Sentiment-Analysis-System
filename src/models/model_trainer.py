import os
import joblib
from datetime import datetime
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
        self.feature_extractor = None

    def create_ensemble_model(self):
        """Creates an enhanced ensemble of models"""
        # Initialize basic pipelines
        rf_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        svm_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_selection', SelectKBest(chi2)),
            ('svm', LinearSVC(random_state=42, class_weight='balanced'))
        ])

        nb_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_selection', SelectKBest(chi2)),
            ('nb', MultinomialNB())
        ])

        models = [
            ('rf', rf_pipeline),
            ('svm', svm_pipeline),
            ('nb', nb_pipeline)
        ]

        # Try to add XGBoost if available
        try:
            from xgboost import XGBClassifier
            xgb_pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('xgb', XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ))
            ])
            models.append(('xgb', xgb_pipeline))
            
            # Add XGBoost parameters to grid
            self.param_grid.update({
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.1],
                'xgb__n_estimators': [100, 200]
            })
            
        except ImportError:
            self.logger.warning("XGBoost not available, continuing without it")

        # Base parameter grid
        self.param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            
            'svm__C': [0.1, 1.0, 10.0],
            'svm__tol': [1e-4, 1e-3],
            'svm__max_iter': [2000],
            
            'nb__alpha': [0.1, 0.5, 1.0],
            'nb__fit_prior': [True, False],
            
            'feature_selection__k': ['all', 100, 200]
        }

        return models

    def save_checkpoint(self, model, metrics, epoch=None):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.language}_checkpoint_{timestamp}.pkl"
        )
        
        checkpoint = {
            'model_state': model,
            'metrics': metrics,
            'epoch': epoch,
            'timestamp': timestamp,
            'language': self.language
        }
        
        joblib.dump(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Keep only last 5 checkpoints
        checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) 
                            if f.startswith(f"{self.language}_checkpoint")])
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

    def save_final_model(self, model, metrics):
        """Save the final trained model with all feature extractors"""
        model_path = os.path.join(
            self.models_dir,
            f"{self.language}_sentiment_model.pkl"
        )
        
        model_info = {
            'model': model,
            'metrics': metrics,
            'feature_extractor': {
                'vectorizer': self.feature_extractor.tfidf,
                'svd': self.feature_extractor.svd,
                'scaler': self.feature_extractor.scaler,
                'word_vectorizer': self.feature_extractor.word_vectorizer,
                'char_vectorizer': self.feature_extractor.char_vectorizer,
                'feature_dims': self.feature_extractor.feature_dims
            },
            'config': {
                'language': self.language,
                'max_features': self.config.MAX_FEATURES,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        }
        
        joblib.dump(model_info, model_path)
        self.logger.info(f"Saved final model to {model_path}")

    def plot_training_progress(self, grid_search):
        """Visualizes the training progress and model performance"""
        plt.figure(figsize=(12, 6))
        
        # Plot CV scores
        cv_results = grid_search.cv_results_
        plt.subplot(1, 2, 1)
        sns.boxplot(data=cv_results['split0_test_score'])
        plt.title(f'Cross-validation Scores\n{self.language.upper()}')
        plt.ylabel('F1 Score')
        
        # Plot feature importance if available
        plt.subplot(1, 2, 2)
        if hasattr(grid_search.best_estimator_, 'named_estimators_'):
            rf_model = grid_search.best_estimator_.named_estimators_['rf']
            importances = rf_model.feature_importances_[:10]  # Top 10 features
            plt.bar(range(10), importances)
            plt.title('Top 10 Feature Importance\n(Random Forest)')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
        
        plt.tight_layout()
        plt.show()

    def train_with_grid_search(self, X_train, y_train):
        """Train models with checkpointing"""
        start_time = datetime.now()
        self.logger.info("Starting individual model training...")
        
        try:
            # Initialize feature extractor if not already initialized
            if self.feature_extractor is None:
                from src.features.feature_engineering import FeatureExtractor
                self.feature_extractor = FeatureExtractor(self.language, self.config)
                
            # Extract features first
            X_train_features = self.feature_extractor.extract_features(X_train)
            if X_train_features is None or X_train_features.shape[0] == 0:
                raise ValueError("Feature extraction failed")
                
            self.logger.info(f"Extracted features shape: {X_train_features.shape}")

            # Create ensemble models
            models = self.create_ensemble_model()
            best_models = {}
            best_metrics = {}

            # Add stratified k-fold cross validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Train each model using extracted features
            for name, pipeline in models:
                try:
                    self.logger.info(f"\nTraining {name} model...")
                    model_params = {k:v for k,v in self.param_grid.items() if k.startswith(name)}
                    
                    grid_search = GridSearchCV(
                        pipeline,
                        param_grid=model_params,
                        cv=cv,
                        scoring=['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted'],
                        refit='f1_weighted',
                        n_jobs=-1,
                        verbose=1,
                        error_score='raise'  # Added to debug failures
                    )

                    # Add sample weights for imbalanced classes
                    sample_weights = compute_sample_weight(
                        'balanced',
                        y_train
                    )

                    # Use the extracted features instead of raw text
                    grid_search.fit(X_train_features, y_train, 
                                  sample_weight=sample_weights)
                    
                    # Track individual model training time
                    model_time = (datetime.now() - start_time).total_seconds()
                    metrics = {
                        'best_score': grid_search.best_score_,
                        'best_params': grid_search.best_params_,
                        'cv_results': grid_search.cv_results_,
                        'training_time': model_time
                    }
                    
                    self.save_checkpoint(grid_search.best_estimator_, metrics, name)
                    best_models[name] = grid_search.best_estimator_
                    best_metrics[name] = metrics
                    
                    self.logger.info(f"{name} Results:")
                    self.logger.info(f"Best parameters: {grid_search.best_params_}")
                    self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue

            # Calculate total training time
            self.training_time = (datetime.now() - start_time).total_seconds()
            
            # Save final model after all training
            if best_models:
                final_metrics = {
                    'models': best_metrics,
                    'total_time': self.training_time
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
                    if 'metrics' in info:
                        if isinstance(info['metrics'], dict):
                            metrics_value = info['metrics'].get('best_score')
                        else:
                            metrics_value = str(info['metrics'])
                            
                    checkpoints.append({
                        'filename': file,
                        'timestamp': info.get('timestamp', 'Unknown'),
                        'epoch': info.get('epoch', 'Unknown'),
                        'metrics': metrics_value
                    })
            return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            self.logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def restore_from_checkpoint(self, checkpoint_name=None):
        """Restore model from checkpoint"""
        try:
            if checkpoint_name is None:
                # Get latest checkpoint
                checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) 
                                   if f.startswith(f"{self.language}_checkpoint")])
                if not checkpoints:
                    raise ValueError("No checkpoints found")
                checkpoint_name = checkpoints[-1]

            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = joblib.load(checkpoint_path)
            self.logger.info(f"Restored model from checkpoint: {checkpoint_name}")
            return checkpoint['model_state'], checkpoint['metrics']

        except Exception as e:
            self.logger.error(f"Error restoring checkpoint: {str(e)}")
            return None, None
