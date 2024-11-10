import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.utils.logger import Logger  # Updated import

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(__name__).logger

    def load_data(self, language: str) -> pd.DataFrame:
        """Load data with strict two-column enforcement"""
        file_path = os.path.join(
            self.config.DATA_DIR, "raw", f"{language}_social_media.csv"
        )
        
        try:
            # First attempt with comma separator
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                usecols=[0, 1],  # Only read first two columns
                names=['text', 'label'],  # Force column names
                header=0  # Skip header row
            )
            return self._validate_dataframe(df)
        except Exception as e:
            self.logger.warning(f"Initial load failed: {str(e)}")
            try:
                # Second attempt with flexible parsing
                df = pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    sep=None,
                    engine='python',
                    usecols=[0, 1],  # Only read first two columns
                    names=['text', 'label'],  # Force column names
                    header=0  # Skip header row
                )
                return self._validate_dataframe(df)
            except Exception as e:
                self.logger.error(f"Failed to load data: {str(e)}")
                return pd.DataFrame(columns=['text', 'label'])

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced DataFrame validation"""
        required_cols = ['text', 'label']
        
        try:
            # Ensure required columns exist
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns. Found: {df.columns}")
                return pd.DataFrame(columns=required_cols)
            
            # Clean text data
            df['text'] = df['text'].astype(str).str.strip()
            
            # Convert and validate labels
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            
            # Keep only valid sentiment labels
            mask = df['label'].isin([0, 1, 2])
            df = df[mask].reset_index(drop=True)
            
            if len(df) == 0:
                self.logger.warning("No valid data after validation")
                return pd.DataFrame(columns=required_cols)
            
            self.logger.info(f"Valid samples after validation: {len(df)}")
            return df[required_cols]
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrame: {str(e)}")
            return pd.DataFrame(columns=required_cols)

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size=test_size, random_state=42)

    def load_processed_data(self, language: str) -> pd.DataFrame:
        file_path = os.path.join(
            self.config.DATA_DIR, "processed", f"{language}_processed_data.csv"
        )
        return pd.read_csv(file_path)

    def get_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Get features and labels with NaN validation"""
        # Ensure cleaned_text exists
        if 'cleaned_text' not in df.columns:
            self.logger.error("Column 'cleaned_text' not found in DataFrame")
            raise ValueError("Missing 'cleaned_text' column")
            
        # Convert label to numeric and handle NaN
        labels = pd.to_numeric(df['label'], errors='coerce')
        valid_mask = labels.notna() & labels.isin([0, 1, 2])
        
        if not valid_mask.any():
            self.logger.error("No valid labels found after validation")
            raise ValueError("No valid labels in dataset")
            
        # Filter both features and labels
        features = df['cleaned_text'][valid_mask]
        labels = labels[valid_mask].astype(int)
        
        self.logger.info(f"Using {len(features)} valid samples after NaN removal")
        
        return features, labels
    
    def load_manual_data(self, language: str) -> pd.DataFrame:
        """Load manually labeled data for a given language"""
        file_path = os.path.join(
            self.config.DATA_DIR, 
            "raw", 
            f"{language}_manual_labeled.csv"
        )
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded manual data for {language}")
            return df
        except FileNotFoundError:
            self.logger.warning(f"No manual data found for {language} at {file_path}")
            return pd.DataFrame(columns=['text', 'label'])
