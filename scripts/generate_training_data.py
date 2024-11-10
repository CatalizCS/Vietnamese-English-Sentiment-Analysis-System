import argparse
import random
import pandas as pd
from sklearn.utils import resample
import sys
import os

from src.utils.templates import CommentTemplates

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

    def generate_natural_variation(self, text: str, label: int):
        """Tạo biến thể tự nhiên cho văn bản"""
        templates = CommentTemplates()
        
        # Thêm opening/closing ngẫu nhiên
        if random.random() < 0.7:  # 70% chance
            text = f"{random.choice(templates.natural_expressions['opening'])}, {text}"
        if random.random() < 0.5:  # 50% chance
            text = f"{text}. {random.choice(templates.natural_expressions['closing'])}"
            
        # Thêm emoji phù hợp
        sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[label]
        if random.random() < 0.8:  # 80% chance
            emojis = templates.emojis[sentiment]
            emoji_count = random.randint(1, 3)
            text = f"{text} {''.join(random.sample(emojis, emoji_count))}"
            
        return text

    def generate_synthetic_data(self, text: str, label: int):
        synthetic_samples = []
        text_variations = set()
        
        templates = CommentTemplates()
        sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[label]
        
        for _ in range(4):
            try:
                # Thêm xác suất để sinh bình luận tương tác
                if random.random() < 0.2:  # 20% chance for interaction comments
                    if label == 0:  # negative
                        interaction_type = random.choice(['argument', 'trolling'])
                        sub_type = 'aggressive' if interaction_type == 'argument' else None
                    elif label == 2:  # positive
                        interaction_type = 'support'
                        sub_type = random.choice(['agreement', 'praise'])
                    else:  # neutral
                        interaction_type = random.choice(['argument', 'support', 'trolling'])
                        sub_type = 'dismissive' if interaction_type == 'argument' else None
                    
                    augmented_text = templates.generate_interaction_comment(interaction_type, sub_type)
                else:
                    # Xác suất để chọn độ dài khác nhau
                    if random.random() < 0.3:
                        augmented_text = templates.generate_varied_length_comment(sentiment, 'general')
                    else:
                        augmented_text = self.eda.humanize_text(text, self.language, sentiment)
                
                augmented_text = self.generate_natural_variation(augmented_text, label)
                
                if augmented_text not in text_variations and len(augmented_text.split()) >= 3:
                    text_variations.add(augmented_text)
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
        
        # Define topics and sentiment ratios
        topics = {
            'product_review': 0.3,
            'food_review': 0.25,
            'service_review': 0.25,
            'movie_review': 0.2
        }
        
        sentiment_ratios = {
            'negative': 0.25,  # 25% negative
            'neutral': 0.25,   # 25% neutral
            'positive': 0.50   # 50% positive
        }
        
        # Calculate samples per topic and sentiment
        for topic, topic_weight in topics.items():
            topic_samples = int(self.num_samples * topic_weight)
            
            for sentiment, sent_ratio in sentiment_ratios.items():
                sent_samples = int(topic_samples * sent_ratio)
                sentiment_label = {'negative': 0, 'neutral': 1, 'positive': 2}[sentiment]
                
                comments = self.eda.generate_topic_comments(
                    topic, 
                    count=sent_samples,
                    language=self.language,
                    sentiment=sentiment_label
                )
                synthetic_data.extend(comments)
        
        # Convert to DataFrame with only required columns
        df = pd.DataFrame(synthetic_data)[['text', 'label']]
        
        # Validate and clean data
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)
        
        # Remove any rows with missing values
        df = df.dropna().reset_index(drop=True)
        
        # Thêm kiểm tra trùng lặp
        df.drop_duplicates(subset=['text'], keep='first', inplace=True)
        
        # Thêm random shuffling để tăng tính ngẫu nhiên
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
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
