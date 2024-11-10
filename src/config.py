import os


class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # Model parameters
    MAX_FEATURES = 5000
    MIN_SAMPLES = 10
    MAX_LEN = 100

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
        'max_checkpoints': 5,
        'checkpoint_frequency': 1,  # Save checkpoint every N epochs
        'model_format': 'pkl',
        'compression': True
    }
