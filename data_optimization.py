
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

def optimize_data_sources():
    """
    Optimize data sources using various techniques:
    1. Combine multiple data sources
    2. Handle class imbalance using SMOTE
    3. Remove duplicates
    4. Handle missing values
    5. Normalize text data
    """
    
    def load_and_preprocess_data(file_paths):
        dataframes = []
        for path in file_paths:
            df = pd.read_csv(path)
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    
    def apply_smote(X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def remove_duplicates(df):
        return df.drop_duplicates()
    
    def handle_missing_values(df):
        return df.dropna()
    
    def normalize_text(text):
        # Add your text normalization logic here
        return text.lower().strip()
    
    # Main optimization process
    file_paths = [
        'path_to_vietnamese_data.csv',
        'path_to_english_data.csv'
    ]
    
    # Load and combine data
    combined_data = load_and_preprocess_data(file_paths)
    
    # Remove duplicates and handle missing values
    combined_data = remove_duplicates(combined_data)
    combined_data = handle_missing_values(combined_data)
    
    # Normalize text data
    combined_data['text'] = combined_data['text'].apply(normalize_text)
    
    # Handle class imbalance
    X = combined_data['text']
    y = combined_data['label']
    X_resampled, y_resampled = apply_smote(X, y)
    
    # Create final balanced dataset
    balanced_data = pd.DataFrame({
        'text': X_resampled,
        'label': y_resampled
    })
    
    return balanced_data

if __name__ == "__main__":
    optimized_data = optimize_data_sources()
    print("Data optimization completed.")
    print(f"Final dataset shape: {optimized_data.shape}")
    print("Class distribution:", Counter(optimized_data['label']))