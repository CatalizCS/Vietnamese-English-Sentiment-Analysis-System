import argparse
import random
import pandas as pd
import numpy as np
from sklearn.utils import resample
import sys
import os
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import Logger
from src.utils.augmentation import TextAugmenter


class TrainingDataGenerator:
    def __init__(self, language: str, config: Config, num_samples: int = 1000):
        self.language = language
        self.config = config
        self.num_samples = num_samples
        self.logger = Logger(__name__).logger
        self.data_loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(language, config)
        self.eda = TextAugmenter()

    def generate_synthetic_data(self, text: str, label: int):
        """Generate synthetic data using more natural text variations"""
        synthetic_samples = []
        max_attempts = min(4, self.num_samples)
        
        sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        sentiment = sentiment_map.get(label, 'neutral')
        
        # Mix of traditional augmentation and humanized text
        methods = [
            lambda t: self.eda.humanize_text(t, self.language, sentiment),
            self.eda.synonym_replacement,
            self.eda.random_swap,
            self.eda.random_insertion
        ]

        for method in methods[:max_attempts]:
            try:
                augmented_text = method(text)
                synthetic_samples.append({"text": augmented_text, "label": label})
            except Exception as e:
                self.logger.warning(f"Error in augmentation: {str(e)}")
                continue

        return synthetic_samples

    def balance_dataset(self, df: pd.DataFrame, target_col: str = "label"):
        """Balance dataset using upsampling"""
        self.logger.info("Balancing dataset...")

        # Get class distribution
        class_counts = df[target_col].value_counts()
        max_size = class_counts.max()

        # Balance each class
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[target_col] == label]
            if len(class_df) < max_size:
                upsampled = resample(
                    class_df, replace=True, n_samples=max_size, random_state=42
                )
                balanced_dfs.append(upsampled)
            else:
                balanced_dfs.append(class_df)

        return pd.concat(balanced_dfs)

    def generate_training_data(self, output_path: str):
        """Main method to generate and save training data with exactly two columns"""
        self.logger.info(f"Generating {self.num_samples} training samples for {self.language}...")
        
        # Create output directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate synthetic data with topic-based generation
        synthetic_data = []
        topics = {
            'product_review': 0.3,
            'food_review': 0.25,
            'service_review': 0.25,
            'movie_review': 0.2
        }
        
        # Calculate samples per topic
        for topic, weight in topics.items():
            count = int(self.num_samples * weight)
            comments = self.eda.generate_topic_comments(
                topic, 
                count=count,
                language=self.language
            )
            synthetic_data.extend(comments)
        
        # Convert to DataFrame with only required columns
        df = pd.DataFrame(synthetic_data)[['text', 'label']]
        
        # Validate and clean data
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)
        
        # Remove any rows with missing values
        df = df.dropna().reset_index(drop=True)
        
        # Save the data
        df.to_csv(output_path, index=False)
        self.logger.info(f"Generated and saved {len(df)} samples to {output_path}")
        
        # Log statistics
        self.logger.info("\n=== Generation Results ===")
        self.logger.info(f"Total samples: {len(df)}")
        self.logger.info("\nClass distribution:")
        self.logger.info(df['label'].value_counts())

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for sentiment analysis"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "vi"],
        help="Language to generate data for (en/vi)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for generated training data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    args = parser.parse_args()

    config = Config()
    generator = TrainingDataGenerator(args.language, config, args.num_samples)
    generator.generate_training_data(args.output)


if __name__ == "__main__":
    main()
