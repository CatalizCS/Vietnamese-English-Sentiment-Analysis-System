import pytest
import pandas as pd
from src.data.preprocessor import DataPreprocessor
from src.config import Config

@pytest.fixture
def preprocessor():
    config = Config()
    return DataPreprocessor('vi', config)

def test_text_cleaning():
    config = Config()
    preprocessor = DataPreprocessor('vi', config)
    
    test_data = pd.DataFrame({
        'text': ['Đây là một bài test!!!', 'This is a test!!!'],
        'label': ['positive', 'negative']
    })
    
    processed_df = preprocessor.preprocess(test_data)
    assert len(processed_df) == 2
    assert 'cleaned_text' in processed_df.columns
    assert processed_df['label'].dtype == 'int64'