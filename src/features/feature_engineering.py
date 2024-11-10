from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import joblib
import re


class FeatureExtractor:
    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.tfidf = None
        self.svd = None
        self.scaler = None
        self.feature_dims = None
        self.vocabulary = None
        self.n_features = None
        self.min_components = 2
        self.is_fitted = False  # Move is_fitted initialization before method calls
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.positive_words = self._load_word_list('positive')
        self.negative_words = self._load_word_list('negative')
        self._initialize_extractors()  # Try to load first
        if not self.is_fitted:  # Only initialize base if loading failed
            self._initialize_base_extractors()

    def _initialize_base_extractors(self):
        """Initialize basic extractors regardless of model existence"""
        self.tfidf = TfidfVectorizer(
            max_features=min(self.config.MAX_FEATURES, 1000),
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
        
        # Initialize word and character vectorizers
        self.word_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=500,
            min_df=1,
            max_df=1.0
        )
        
        self.svd = None  # Will be initialized during feature extraction
        self.scaler = MinMaxScaler()

    def _initialize_extractors(self):
        """Load pretrained extractors if available"""
        model_path = os.path.join(
            self.config.DATA_DIR, "models", f"{self.language}_sentiment_model.pkl"
        )

        if os.path.exists(model_path):
            try:
                model_info = joblib.load(model_path)
                if "feature_extractor" in model_info:
                    # Load all vectorizers and transformers
                    self.tfidf = model_info["feature_extractor"]["vectorizer"]
                    self.svd = model_info["feature_extractor"]["svd"]
                    self.scaler = model_info["feature_extractor"]["scaler"]
                    self.word_vectorizer = model_info["feature_extractor"].get("word_vectorizer")
                    self.char_vectorizer = model_info["feature_extractor"].get("char_vectorizer")
                    self.feature_dims = model_info["feature_extractor"].get("feature_dims")
                    self.vocabulary = self.tfidf.vocabulary_
                    
                    # Initialize vectorizers if they don't exist in saved model
                    if self.word_vectorizer is None:
                        self._initialize_text_vectorizers()
                    
                    self.is_fitted = True
                    print(f"Loaded feature extractor with {self.feature_dims} dimensions")
                    return True
            except Exception as e:
                print(f"Error loading pretrained extractors: {str(e)}")
                self._initialize_text_vectorizers()
        return False

    def _initialize_text_vectorizers(self):
        """Initialize word and character vectorizers"""
        self.word_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=500,
            min_df=1,
            max_df=1.0
        )

    def _load_sentiment_lexicon(self):
        """Load sentiment lexicon based on language"""
        try:
            lexicon_path = os.path.join(
                self.config.DATA_DIR,
                "lexicons",
                f"{self.language}_sentiment_lexicon.txt"
            )
            if os.path.exists(lexicon_path):
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f)
            return set()
        except Exception:
            return set()

    def _load_word_list(self, category):
        """Load positive/negative word lists"""
        try:
            path = os.path.join(
                self.config.DATA_DIR,
                "lexicons",
                f"{self.language}_{category}_words.txt"
            )
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f)
            return set()
        except Exception:
            return set()

    def extract_features(self, texts):
        try:
            # Validate input
            if texts is None or len(texts) == 0:
                raise ValueError("Empty input texts provided")

            # Convert input
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, pd.Series):
                texts = texts.tolist()
                
            # Clean and validate texts
            valid_texts = [str(t).strip() for t in texts if str(t).strip()]
            if not valid_texts:
                raise ValueError("No valid text content after cleaning")

            # Extract features based on fit status
            if not self.is_fitted:
                # Training phase - fit and transform
                word_features = self.word_vectorizer.fit_transform(valid_texts).toarray()
                char_features = self.char_vectorizer.fit_transform(valid_texts).toarray()
                tfidf_features = self.tfidf.fit_transform(valid_texts).toarray()
            else:
                # Prediction phase - transform only
                word_features = self.word_vectorizer.transform(valid_texts).toarray()
                char_features = self.char_vectorizer.transform(valid_texts).toarray()
                tfidf_features = self.tfidf.transform(valid_texts).toarray()

            # Process with SVD
            if self.svd is None:
                n_components = min(95, tfidf_features.shape[1] - 1, len(valid_texts) - 1)
                self.svd = TruncatedSVD(n_components=max(2, n_components))
                svd_features = self.svd.fit_transform(tfidf_features)
            else:
                svd_features = self.svd.transform(tfidf_features)

            # Scale features
            scaled_features = (self.scaler.fit_transform(svd_features) 
                             if not self.is_fitted 
                             else self.scaler.transform(svd_features))

            # Get statistical features
            stat_features = self._extract_statistical_features(valid_texts)

            # Combine all features
            combined_features = np.hstack([
                word_features,
                char_features,
                scaled_features,
                stat_features
            ])

            # Update dimensions if needed
            if not self.is_fitted:
                self.feature_dims = combined_features.shape[1]
                self.is_fitted = True
            elif self.feature_dims:
                # Ensure consistent dimensions
                if combined_features.shape[1] < self.feature_dims:
                    padding = np.zeros((combined_features.shape[0], 
                                     self.feature_dims - combined_features.shape[1]))
                    combined_features = np.hstack([combined_features, padding])
                else:
                    combined_features = combined_features[:, :self.feature_dims]

            return combined_features

        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            print(f"Debug info - Input size: {len(texts)}")
            print(f"Debug info - Features shape: {combined_features.shape if 'combined_features' in locals() else 'N/A'}")
            print(f"Debug info - Is fitted: {self.is_fitted}")
            print(f"Debug info - Vectorizers: word={self.word_vectorizer is not None}, char={self.char_vectorizer is not None}")
            raise

    def _extract_and_scale_features(self, tfidf_features, texts):
        # Calculate optimal number of components
        max_components = min(
            100,  # Maximum desired components
            tfidf_features.shape[1],  # Available features
            len(texts) - 1,  # Number of samples - 1
            self.config.MAX_FEATURES,  # Config limit
        )

        if self.svd.n_components != max_components:
            self.svd = TruncatedSVD(n_components=max_components)

        # Reduce dimensionality with SVD
        n_components = min(100, tfidf_features.shape[1], len(texts) - 1)
        self.svd.n_components = n_components
        svd_features = self.svd.fit_transform(tfidf_features)

        # Scale SVD features to [0,1] range
        svd_features = self.scaler.fit_transform(svd_features)

        # Get statistical features
        stat_features = self._extract_statistical_features(texts)

        # Combine features
        return np.hstack([svd_features, stat_features])

    def _extract_statistical_features(self, texts):
        """Enhanced statistical features for better accuracy"""
        features = []
        for text in texts:
            text = str(text)
            words = text.split()
            
            # Basic features
            length = len(text)
            word_count = len(text.split())
            avg_word_length = length / max(word_count, 1)

            # Additional features
            unique_chars = len(set(text))
            digit_count = sum(c.isdigit() for c in text)
            upper_count = sum(c.isupper() for c in text)
            space_count = sum(c.isspace() for c in text)
            special_chars = sum(not c.isalnum() for c in text)

            # New advanced features
            sentiment_words = len([w for w in words if w in self.sentiment_lexicon])
            exclamation_count = text.count('!')
            question_count = text.count('?')
            emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
            
            # Word patterns
            caps_word_count = len([w for w in words if w.isupper()])
            word_length_variance = np.var([len(w) for w in words]) if words else 0
            
            # Sentiment patterns
            positive_words = len([w for w in words if w in self.positive_words])
            negative_words = len([w for w in words if w in self.negative_words])
            
            features.append([
                length,
                word_count,
                avg_word_length,
                unique_chars,
                digit_count,
                upper_count,
                space_count,
                special_chars,
                sentiment_words / max(len(words), 1),
                exclamation_count,
                question_count,
                emoji_count,
                caps_word_count / max(len(words), 1),
                word_length_variance,
                positive_words / max(len(words), 1),
                negative_words / max(len(words), 1)
            ])

        return np.array(features, dtype=np.float32)
