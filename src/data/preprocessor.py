import pandas as pd
from src.utils.logger import Logger
from src.features.text_cleaner import TextCleaner


class DataPreprocessor:
    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.logger = Logger(__name__).logger
        self.text_cleaner = TextCleaner(language, config)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with enhanced validation"""
        self.logger.info(f"Preprocessing {self.language} data...")

        try:
            # Ensure we have a copy and data is not empty
            if df is None or df.empty:
                raise ValueError("Empty input data")
            df = df.copy()

            # Validate text column
            if "text" not in df.columns:
                raise ValueError("No 'text' column found")

            # Convert text to string and clean
            df["text"] = df["text"].astype(str).fillna("")
            df["cleaned_text"] = df["text"].apply(self.text_cleaner.clean_text)

            # Remove invalid texts
            df = df[df["cleaned_text"].str.strip().str.len() > 3].copy()

            # Validate label column if it exists
            if "label" in df.columns:
                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
                df = df[df["label"].isin([0, 1, 2])].copy()

            elif "sentiment" in df.columns:
                df["label"] = df["sentiment"].apply(self._convert_to_basic_sentiment)
                df = df.drop(columns=["sentiment"])
            else:
                df["label"] = 1  # Neutral sentiment

            df = df.reset_index(drop=True)
            if len(df) == 0:
                raise ValueError("No valid samples after preprocessing")

            self.logger.info(f"Preprocessed {len(df)} valid samples")
            return df

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=["text", "cleaned_text", "label"])

    def _convert_to_basic_sentiment(self, label):
        """Convert detailed emotion label to basic sentiment"""
        if label in self.config.EMOTION_LABELS["POSITIVE"].keys():
            return 2  # positive
        elif label in self.config.EMOTION_LABELS["NEGATIVE"].keys():
            return 0  # negative
        else:
            return 1  # neutral

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")
