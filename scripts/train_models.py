
import argparse
import joblib
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureExtractor
from src.models.model_trainer import EnhancedModelTrainer
from src.utils.logger import Logger

def train_model_for_language(language: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Training model for {language}...")
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(language, config)
    model_trainer = EnhancedModelTrainer(language, config)
    
    # Load processed data
    df = data_loader.load_processed_data(language)
    X, y = data_loader.get_features_and_labels(df)
    
    # Extract features
    X_features = feature_extractor.extract_features(X)
    
    # Train model
    model = model_trainer.train_with_grid_search(X_features, y)
    
    # Save model
    model_path = config.LANGUAGE_CONFIGS[language]['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--languages', nargs='+', choices=['en', 'vi'],
                      default=['en', 'vi'],
                      help='Languages to train models for')
    args = parser.parse_args()
    
    config = Config()
    for language in args.languages:
        train_model_for_language(language, config)

if __name__ == "__main__":
    main()