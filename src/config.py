import os


class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # Model parameters - Giảm độ phức tạp
    MAX_FEATURES = 10000  # tránh noise
    MIN_SAMPLES = 5  # đảm bảo tính ổn định
    MAX_LEN = 300  # giảm noise
    SVD_COMPONENTS = 100  # Add this line to define the number of SVD components
    # Language specific configs
    LANGUAGE_CONFIGS = {
        "vi": {
            "stop_words": ["và", "của", "các", "có", "được", "trong", "đã", "này"],
            "model_path": os.path.join(DATA_DIR, "models", "vi_sentiment_model.pkl"),
        },
        "en": {
            "stop_words": "english",  # Using NLTK's English stop words
            "model_path": os.path.join(DATA_DIR, "models", "en_sentiment_model.pkl"),
        },
    }

    # Expanded emotion labels
    EMOTION_LABELS = {
        "POSITIVE": {
            2: "positive",
            3: "excited",
            4: "happy",
            5: "satisfied",
            6: "impressed",
        },
        "NEGATIVE": {
            0: "negative",
            7: "angry",
            8: "disappointed",
            9: "frustrated",
            10: "worried",
        },
        "NEUTRAL": {1: "neutral", 11: "confused", 12: "uncertain", 13: "mixed"},
    }

    # Emotion mapping for conversion
    EMOTION_TO_SENTIMENT = {
        # Positive emotions -> 2
        "excited": 2,
        "happy": 2,
        "satisfied": 2,
        "impressed": 2,
        "positive": 2,
        # Negative emotions -> 0
        "angry": 0,
        "disappointed": 0,
        "frustrated": 0,
        "worried": 0,
        "negative": 0,
        # Neutral emotions -> 1
        "confused": 1,
        "uncertain": 1,
        "mixed": 1,
        "neutral": 1,
    }

    # Model saving configurations
    MODEL_SAVE_CONFIG = {
        "max_checkpoints": 5,
        "checkpoint_frequency": 1,  # Save checkpoint every N epochs
        "model_format": "pkl",
        "compression": True,
    }

    # Expanded emotion mapping
    EMOTION_MAPPING = {
        # Positive emotions (2)
        "happy": {"id": 2.1, "sentiment": 2, "vi": "vui vẻ", "emoji": "😊"},
        "excited": {"id": 2.2, "sentiment": 2, "vi": "phấn khích", "emoji": "🤗"},
        "satisfied": {"id": 2.3, "sentiment": 2, "vi": "hài lòng", "emoji": "😌"},
        "proud": {"id": 2.4, "sentiment": 2, "vi": "tự hào", "emoji": "😊"},
        # Neutral emotions (1)
        "neutral": {"id": 1.0, "sentiment": 1, "vi": "bình thường", "emoji": "😐"},
        "surprised": {"id": 1.1, "sentiment": 1, "vi": "ngạc nhiên", "emoji": "😮"},
        "confused": {"id": 1.2, "sentiment": 1, "vi": "bối rối", "emoji": "😕"},
        # Negative emotions (0)
        "sad": {"id": 0.1, "sentiment": 0, "vi": "buồn", "emoji": "😢"},
        "angry": {"id": 0.2, "sentiment": 0, "vi": "giận dữ", "emoji": "😠"},
        "disappointed": {"id": 0.3, "sentiment": 0, "vi": "thất vọng", "emoji": "😞"},
        "frustrated": {"id": 0.4, "sentiment": 0, "vi": "bực bội", "emoji": "😤"},
        "worried": {"id": 0.5, "sentiment": 0, "vi": "lo lắng", "emoji": "😟"},
    }

    # Emotion keywords for each category
    EMOTION_KEYWORDS = {
        "vi": {
            "happy": ["vui", "hạnh phúc", "thích", "tuyệt vời", "tốt", "thú vị"],
            "excited": ["phấn khích", "hào hứng", "tuyệt quá", "wow"],
            "satisfied": ["hài lòng", "thoải mái", "ổn", "được"],
            "proud": ["tự hào", "xuất sắc", "giỏi"],
            "neutral": ["bình thường", "tạm", "okay"],
            "surprised": ["ngạc nhiên", "bất ngờ", "không ngờ"],
            "confused": ["bối rối", "không hiểu", "lạ"],
            "sad": ["buồn", "khổ", "chán", "thương"],
            "angry": ["giận", "tức", "khó chịu", "ghét"],
            "disappointed": ["thất vọng", "không được", "kém"],
            "frustrated": ["bực", "khó chịu", "phiền"],
            "worried": ["lo", "sợ", "không an tâm"],
        },
        "en": {
            "happy": ["happy", "joyful", "pleased", "delighted", "good", "great"],
            "excited": ["excited", "thrilled", "eager", "enthusiastic", "wow"],
            "satisfied": ["satisfied", "content", "pleased", "okay", "fine"],
            "proud": ["proud", "accomplished", "successful", "confident"],
            "neutral": ["neutral", "indifferent", "unaffected", "okay"],
            "surprised": ["surprised", "shocked", "amazed", "astonished"],
            "confused": ["confused", "puzzled", "perplexed", "baffled"],
            "sad": ["sad", "unhappy", "down", "depressed", "miserable"],
            "angry": ["angry", "mad", "furious", "irritated", "annoyed"],
            "disappointed": ["disappointed", "dissatisfied", "unhappy", "let down"],
            "frustrated": ["frustrated", "annoyed", "irritated", "exasperated"],
            "worried": ["worried", "concerned", "anxious", "nervous", "apprehensive"],
        },
    }

    # API Configuration
    API_CONFIG = {
        "HOST": "0.0.0.0",
        "PORT": 7270,
        "WORKERS": 4,
        "TIMEOUT": 60,
        "RELOAD": True,
        "CORS_ORIGINS": ["*"],
        "MAX_REQUEST_SIZE": 1024 * 1024,  # 1MB
        "RATE_LIMIT": {"requests": 10000, "window": 60},  # seconds
    }

    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        "update_interval": 5,  # seconds
        "metrics_history": 100,  # number of historical data points to keep
        "charts": {
            "request_rate": {"window": 60},  # 1 minute window
            "response_time": {"window": 300},  # 5 minute window
            "error_rate": {"window": 300},
        },
    }

    # Metrics Configuration
    METRICS_CONFIG = {
        "collect_detailed_metrics": True,
        "metrics_retention_days": 7,
        "metrics_file": "api_metrics.json",
        "alert_thresholds": {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 1.0,  # 1 second
            "memory_usage": 0.8,  # 80% memory usage
        },
    }

    # Metrics configuration
    METRICS_CONFIG = {
        "retention_days": 7,
        "alert_thresholds": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8},
    }

    # Enhanced model training configuration
    # MODEL_TRAINING_CONFIG = {
    #     "cv_folds": 10,  # Increased from 5
    #     "class_weight_method": "balanced",
    #     "feature_selection_method": "mutual_info_classif",
    #     "sampling_strategy": "smote",
    #     "preprocessing": {
    #         "min_df": 10,  # Increased from 5
    #         "max_df": 0.85,  # Reduced from 0.9
    #         "ngram_range": (1, 2),  # Reduced from (1,3) to prevent overfitting
    #         "analyzer": ["word", "char_wb"],
    #         "strip_accents": "unicode",
    #         "binary": True,
    #         "sublinear_tf": True,
    #     },
    # }
    MODEL_TRAINING_CONFIG = {
        "cv_folds": 10,
        "class_weight_method": "balanced",
        "preprocessing": {
            "min_df": 15,  # Increased to reduce noise
            "max_df": 0.8,  # Reduced to remove very common words
            "ngram_range": (1, 3),
            "max_features": 15000  # Increased vocabulary size
        },
        "ensemble": {
            "voting": "soft",
            "weights": [0.4, 0.3, 0.3]  # RF, SVM, NB weights
        }
    }

    # Optimized parameter grid with better regularization
    PARAM_GRID = {
        # Random Forest - Improved parameters
        "rf__n_estimators": [500, 800, 1000],  # Increased values
        "rf__max_depth": [30, 50, 70],  # Adjusted range
        "rf__min_samples_split": [10, 15],  # Increased to reduce overfitting
        "rf__min_samples_leaf": [4, 8],  # Increased for better generalization
        "rf__max_features": ["sqrt", "log2"],  # Added log2 option
        "rf__bootstrap": [True],
        "rf__criterion": ["gini", "entropy"],  # Added entropy
        "rf__oob_score": [True],

        # SVM - Enhanced regularization
        "svm__C": [0.01, 0.1, 0.5],  # Adjusted for stronger regularization
        "svm__tol": [1e-4, 1e-3],
        "svm__max_iter": [2000],  # Increased iterations
        "svm__class_weight": ["balanced"],
        "svm__dual": [False],

        # Naive Bayes - Adjusted smoothing
        "nb__alpha": [0.8, 1.2, 1.5],  # Adjusted range
        "nb__fit_prior": [True, False],  # Test both options

        # Feature Selection - Optimized
        "feature_selection__k": [800, 1200],  # Adjusted feature count
        "feature_selection__score_func": ["mutual_info_classif"],
    }

    VALIDATION_CONFIG = {
        "early_stopping": {
            "patience": 5,  # Increased from 3
            "min_delta": 0.0005,  # Reduced from 0.001 for finer control
            "monitor": "val_score"
        },
        "validation_split": 0.15,  # Reduced from 0.2
        "shuffle": True,
        "random_state": 42,
    }

    # Enhanced regularization configuration
    # REGULARIZATION_CONFIG = {
    #     "rf_reg": {
    #         "ccp_alpha": 0.005,  # Reduced from 0.01 for finer pruning
    #         "max_samples": 0.7,  # Reduced from 0.8
    #     },
    #     "svm_reg": {
    #         "kernel": "linear",
    #         "shrinking": True
    #     },
    # }
    REGULARIZATION_CONFIG = {
        "rf_reg": {
            "ccp_alpha": 0.002,  # Pruning strength 
            "max_samples": 0.8,
            "max_features": "sqrt",
            "min_samples_leaf": 4
        },
        "svm_reg": {
            "C": 0.8,  # Stronger regularization
            "class_weight": "balanced",
            "dual": False
        },
        "nb_reg": {
            "alpha": 1.2,
            "fit_prior": True
        }
    }

    # Add scoring configuration
    SCORING_CONFIG = {
        "precision_zero_division": 1,  # Handle zero division in precision
        "score_weights": {"precision": 0.4, "recall": 0.4, "f1": 0.2},
    }

    # Error messages
    ERROR_MESSAGES = {
        "MODEL_NOT_FOUND": "Model not found for language {}",
        "PREPROCESSING_FAILED": "Text preprocessing failed",
        "FEATURE_EXTRACTION_FAILED": "Feature extraction failed",
        "PREDICTION_FAILED": "Prediction failed",
        "INVALID_LANGUAGE": 'Invalid language. Must be "vi" or "en"',
        "EMPTY_TEXT": "Empty text provided",
        "SERVER_ERROR": "Internal server error",
    }

    # Model information
    MODEL_INFO = {
        "vi": {
            "name": "Vietnamese Sentiment Model",
            "version": "1.0.0",
            "description": "A model for analyzing Vietnamese sentiment.",
        },
        "en": {
            "name": "English Sentiment Model",
            "version": "1.0.0",
            "description": "A model for analyzing English sentiment.",
        },
    }
