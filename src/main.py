import argparse
import pandas as pd
import os
import sys
import joblib
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureExtractor
from src.models.model_trainer import EnhancedModelTrainer
from src.models.model_predictor import SentimentPredictor
from src.utils.evaluation import ModelEvaluator
from src.utils.logger import Logger
from src.utils.menu import TerminalMenu
from scripts.generate_training_data import TrainingDataGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis CLI')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'evaluate'],
                      help='Mode of operation: train, predict, or evaluate')
    parser.add_argument('--language', type=str, required=True, choices=['en', 'vi'],
                      help='Language for sentiment analysis (en/vi)')
    parser.add_argument('--input', type=str,
                      help='Input file path for prediction or evaluation')
    parser.add_argument('--output', type=str,
                      help='Output file path for saving results')
    return parser.parse_args()

def train(language: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting training for {language} language")
    
    try:
        # Initialize components
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(language, config)
        feature_extractor = FeatureExtractor(language, config)
        model_trainer = EnhancedModelTrainer(language, config)
        model_trainer.feature_extractor = feature_extractor  # Share feature extractor

        # Load and validate data
        df = data_loader.load_data(language)
        if df.empty:
            raise ValueError("No data loaded")
            
        processed_df = preprocessor.preprocess(df)
        if processed_df.empty:
            raise ValueError("No valid samples after preprocessing")
            
        # Split and extract features
        train_df, test_df = data_loader.split_data(processed_df)
        X_train, y_train = data_loader.get_features_and_labels(train_df)
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("No training samples available")
            
        # Record start time
        start_time = datetime.now()

        # Train model
        model = model_trainer.train_with_grid_search(X_train, y_train)
        # ...rest of existing code...
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        X_test, y_test = data_loader.get_features_and_labels(test_df)
        X_test_features = feature_extractor.extract_features(X_test)
        
        # Get final metrics
        final_metrics = {
            'test_score': model['rf'].score(X_test_features, y_test),
            'feature_importance': getattr(model['rf'], 'feature_importances_', None),
            'training_time': training_time
        }
        
        # Save final model with metrics
        model_trainer.save_final_model(model, final_metrics)
        
        logger.info("Training completed successfully")
        logger.info(f"Total training time: {training_time:.2f} seconds")
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Load last checkpoint if available
        checkpoint_dir = os.path.join(config.DATA_DIR, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) 
                               if f.startswith(f"{language}_checkpoint")])
            if checkpoints:
                latest_checkpoint = joblib.load(os.path.join(checkpoint_dir, checkpoints[-1]))
                logger.info(f"Loaded latest checkpoint from {checkpoints[-1]}")
                return latest_checkpoint['model_state']
        raise e

def predict(language: str, input_file: str, output_file: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting prediction for {language} language")

    # Initialize components
    predictor = SentimentPredictor(language, config)
    preprocessor = DataPreprocessor(language, config)
    feature_extractor = FeatureExtractor(language, config)

    # Load and process input data
    df = pd.read_csv(input_file)
    processed_df = preprocessor.preprocess(df)
    features = feature_extractor.extract_features(processed_df['cleaned_text'])
    
    # Make predictions
    predictions = predictor.predict(features)
    df['sentiment'] = predictions
    df.to_csv(output_file, index=False)
    
    logger.info(f"Predictions saved to {output_file}")

def evaluate(language: str, input_file: str, config: Config):
    logger = Logger(__name__).logger
    logger.info(f"Starting evaluation for {language} language")

    # Initialize components
    predictor = SentimentPredictor(language, config)
    preprocessor = DataPreprocessor(language, config)
    feature_extractor = FeatureExtractor(language, config)
    evaluator = ModelEvaluator(language)

    # Load and process test data
    df = pd.read_csv(input_file)
    processed_df = preprocessor.preprocess(df)
    features = feature_extractor.extract_features(processed_df['cleaned_text'])
    
    # Make predictions and evaluate
    predictions = predictor.predict(features)
    probabilities = predictor.predict_proba(features)
    results = evaluator.evaluate(processed_df['label'], predictions, probabilities)
    
    logger.info("Evaluation results:")
    logger.info(results['classification_report'])

def validate_paths(input_file: str = None, output_file: str = None):
    """Validate input and output paths"""
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

def test_model(language: str, config: Config):
    """Interactive model testing"""
    logger = Logger(__name__).logger
    menu = TerminalMenu()  # Move menu initialization to top
    
    try:
        # Initialize components
        predictor = SentimentPredictor(language, config)
        preprocessor = DataPreprocessor(language, config)
        feature_extractor = FeatureExtractor(language, config)
        
        while True:
            try:
                # Get test text
                test_text = menu.get_test_text()
                if test_text.lower() == 'q':
                    break
                    
                # Create DataFrame with single text
                df = pd.DataFrame({'text': [test_text]})
                
                # Process text
                processed_df = preprocessor.preprocess(df)
                if processed_df.empty:
                    menu.display_result(False, "Text preprocessing failed")
                    continue
                    
                # Extract features
                features = feature_extractor.extract_features(processed_df['cleaned_text'])
                if features is None or features.size == 0:
                    menu.display_result(False, "Feature extraction failed")
                    continue
                    
                # Get predictions
                prediction = predictor.predict(features)[0]
                probabilities = predictor.predict_proba(features)[0]
                confidence = max(probabilities)
                
                # Display results
                menu.display_sentiment_result(test_text, prediction, confidence)
                
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                menu.display_result(False, f"Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Test model error: {str(e)}")
        menu.display_result(False, f"Error: {str(e)}")

    return

def restore_model(language: str, config: Config):
    """Restore model from checkpoint"""
    logger = Logger(__name__).logger
    trainer = EnhancedModelTrainer(language, config)
    
    try:
        # List available checkpoints
        checkpoints = trainer.list_checkpoints()
        if not checkpoints:
            logger.warning("No checkpoints available")
            return None
            
        # Display available checkpoints
        print("\nAvailable checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. {cp['filename']}")
            print(f"   Timestamp: {cp['timestamp']}")
            print(f"   Epoch: {cp['epoch']}")
            print(f"   Score: {cp['metrics']:.4f if cp['metrics'] else 'N/A'}")
            
        # Get user choice
        choice = input("\nEnter checkpoint number to restore (or press Enter for latest): ")
        if choice.strip():
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_name = checkpoints[idx]['filename']
            else:
                raise ValueError("Invalid checkpoint number")
        else:
            checkpoint_name = None
            
        # Restore model
        model, metrics = trainer.restore_from_checkpoint(checkpoint_name)
        if model is not None:
            logger.info("Model restored successfully")
            logger.info(f"Model metrics: {metrics}")
            return model
            
    except Exception as e:
        logger.error(f"Error restoring model: {str(e)}")
        return None

def main():
    menu = TerminalMenu()
    config = Config()

    while True:
        menu.display_header()
        choice = menu.display_menu()

        if choice == 'q':
            break

        language = menu.get_language_choice()

        try:
            if choice == '1':
                menu.display_progress("Training new model")
                train(language, config)
                menu.display_result(True, "Model training completed successfully")

            elif choice == '2':
                input_file = menu.get_file_path("input")
                output_file = menu.get_file_path("output")
                validate_paths(input_file, output_file)
                menu.display_progress("Analyzing text")
                predict(language, input_file, output_file, config)
                menu.display_result(True, f"Results saved to {output_file}")

            elif choice == '3':
                input_file = menu.get_file_path("test data")
                validate_paths(input_file)
                menu.display_progress("Evaluating model")
                evaluate(language, input_file, config)

            elif choice == '4':
                output_file = menu.get_file_path("output")
                validate_paths(output_file=output_file)
                num_samples = menu.get_sample_count()
                menu.display_progress(f"Generating {num_samples} training samples")
                
                generator = TrainingDataGenerator(language, config, num_samples)
                generator.generate_training_data(output_file)
                
                menu.display_result(True, f"Generated {num_samples} samples successfully")

            elif choice == '5':
                menu.display_progress("Loading model metrics")
                # Add your display_metrics function call here
                evaluator = ModelEvaluator(language)
                # Display metrics logic here
                
            elif choice == '6':  # Add new test option
                menu.display_progress("Loading model for testing")
                test_model(language, config)

            elif choice == '7':  # Add new restore option
                menu.display_progress("Restoring model from checkpoint")
                model = restore_model(language, config)
                if model:
                    menu.display_result(True, "Model restored successfully")
                else:
                    menu.display_result(False, "Failed to restore model")

        except Exception as e:
            menu.display_result(False, f"Error: {str(e)}")

        menu.wait_for_user()

if __name__ == "__main__":
    main()
