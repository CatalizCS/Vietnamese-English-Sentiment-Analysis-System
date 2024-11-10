# Vietnamese-English Sentiment Analysis System

A robust machine learning system for sentiment analysis supporting both Vietnamese and English text, built with advanced NLP techniques and ensemble learning.

## Key Features

- **Multilingual Support**: Vietnamese and English language processing
- **Advanced Text Processing**:
  - Intelligent text cleaning
  - Stop word removal
  - Language-specific tokenization
- **Feature Engineering**:
  - TF-IDF vectorization
  - SVD dimensionality reduction
  - Statistical feature extraction
- **Ensemble Learning**:
  - Random Forest
  - Linear SVC
  - Naive Bayes
- **Data Augmentation**:
  - Synonym replacement
  - Random swap
  - Random deletion
  - Random insertion

## Directory Structure
```
sentiment_analysis/
├── data/
│ ├── raw/ # Raw input data
│ ├── processed/ # Processed data
│ └── models/ # Trained model files
├── src/
│ ├── config.py # Configuration settings
│ ├── main.py # Main application entry point
│ ├── data/ # Data handling modules
│ ├── features/ # Feature engineering
│ ├── models/ # Model training and prediction
│ └── utils/ # Utility functions
├── scripts/
│ ├── generate_training_data.py
│ └── train_models.py
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment_analysis.git
cd sentiment_analysis
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Data Generation
   Generate training data for both languages:

```bash
python scripts/generate_training_data.py --language vi --output data/processed/vi_processed_data.csv
python scripts/generate_training_data.py --language en --output data/processed/en_processed_data.csv
```

2. Model Training Train models for both languages:

```bash
python scripts/train_models.py --language vi --output models/vi_model.pkl
python scripts/train_models.py --language en --output models/en_model.pkl
```

3. Sentiment Analysis
   The main application supports three modes:

   - **Training Mode**: Train the model with new data.
   - **Prediction Mode**: Predict sentiment for new input text.
   - **Evaluation Mode**: Evaluate the model performance on test data.

4. Input Data Format
   The input CSV files should have the following format:

   ```csv
   text,label
   "This is a positive review",positive
   "This is a negative review",negative
   ```

## Model Performance

Current model performance metrics:

| Language   | Accuracy | F1-Score | Precision | Recall |
| ---------- | -------- | -------- | --------- | ------ |
| English    | 0.85     | 0.84     | 0.83      | 0.85   |
| Vietnamese | 0.82     | 0.81     | 0.80      | 0.82   |

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- underthesea (for Vietnamese)
- nltk
- textaugment
- seaborn
- matplotlib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details
