from setuptools import setup, find_packages

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
    author="Your Name",
    description="Sentiment Analysis for Vietnamese and English Social Media Data",
    python_requires=">=3.7",
)
