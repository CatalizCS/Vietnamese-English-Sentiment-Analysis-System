from setuptools import setup, find_packages
import subprocess
import sys
import nltk
import os

def setup_environment():
    """Initialize required data and install dependencies"""
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {str(e)}")
        return False

    print("\nDownloading NLTK data...")
    try:
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        return False

    print("\nCreating required directories...")
    dirs = [
        'data',
        'data/raw_data',
        'data/processed_data',
        'data/models',
        'data/checkpoints',
        'data/metrics',
        'data/lexicons',
        'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    print("\nSetup completed successfully!")
    return True

if __name__ == "__main__":
    setup_environment()

setup(
    name="sentiment_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "underthesea",  # For Vietnamese text processing
        "matplotlib",
        "seaborn",
        "joblib",
        "pytest",
    ],
    author="CatalizCS",
    description="Sentiment Analysis for Vietnamese and English Social Media Data",
    python_requires=">=3.7",
)
