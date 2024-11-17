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
        try:
            self.language = language
            self.config = config
            self.is_fitted = False

            # Initialize all required attributes
            self.tfidf = None
            self.svd = None
            self.scaler = None
            self.feature_dims = None
            self.vocabulary = None
            self.n_features = None
            self.min_components = 2
            self.word_vectorizer = None
            self.char_vectorizer = None

            # Load lexicons and initialize vectorizers
            self.sentiment_lexicon = self._load_sentiment_lexicon()
            self.positive_words = self._load_word_list("positive")
            self.negative_words = self._load_word_list("negative")

            # Try loading existing extractors first
            if not self._initialize_extractors():
                self._initialize_base_extractors()

        except Exception as e:
            print(f"Error initializing FeatureExtractor: {str(e)}")
            raise

    def _initialize_base_extractors(self):
        """Initialize feature extractors with documented parameters"""
        
        # TF-IDF vectorizer 
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True
        )

        # SVD for dimensionality reduction
        self.svd = TruncatedSVD(n_components=None)

        # Word and character level vectorizers 
        self.word_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2,4), 
            max_features=500
        )

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
                    self.word_vectorizer = model_info["feature_extractor"].get(
                        "word_vectorizer"
                    )
                    self.char_vectorizer = model_info["feature_extractor"].get(
                        "char_vectorizer"
                    )
                    self.feature_dims = model_info["feature_extractor"].get(
                        "feature_dims"
                    )
                    self.vocabulary = self.tfidf.vocabulary_

                    # Initialize vectorizers if they don't exist in saved model
                    if self.word_vectorizer is None:
                        self._initialize_text_vectorizers()

                    self.is_fitted = True
                    print(
                        f"Loaded feature extractor with {self.feature_dims} dimensions"
                    )
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
            lowercase=True,
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=500, min_df=1, max_df=1.0
        )

    def _load_sentiment_lexicon(self):
        """Load sentiment lexicon based on language"""
        try:
            lexicon_path = os.path.join(
                self.config.DATA_DIR,
                "lexicons",
                f"{self.language}_sentiment_lexicon.txt",
            )
            if os.path.exists(lexicon_path):
                with open(lexicon_path, "r", encoding="utf-8") as f:
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
                f"{self.language}_{category}_words.txt",
            )
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return set(line.strip() for line in f)
            return set()
        except Exception:
            return set()

    def extract_features(self, texts):
        """Enhanced feature extraction with proper array handling"""
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
                word_features = self.word_vectorizer.fit_transform(
                    valid_texts
                ).toarray()
                char_features = self.char_vectorizer.fit_transform(
                    valid_texts
                ).toarray()
                tfidf_features = self.tfidf.fit_transform(valid_texts).toarray()
            else:
                # Prediction phase - transform only
                word_features = self.word_vectorizer.transform(valid_texts).toarray()
                char_features = self.char_vectorizer.transform(valid_texts).toarray()
                tfidf_features = self.tfidf.transform(valid_texts).toarray()

            # Add new linguistic features
            linguistic_features = self._extract_linguistic_features(valid_texts)

            # Add emotion lexicon features
            emotion_features = self._extract_emotion_features(valid_texts)

            # Add semantic features if available
            semantic_features = self._extract_semantic_features(valid_texts)

            # Debugging: Print shapes of individual feature arrays
            print(f"Word features shape: {word_features.shape}")
            print(f"Char features shape: {char_features.shape}")
            print(f"Tfidf features shape: {tfidf_features.shape}")
            print(f"Linguistic features shape: {linguistic_features.shape}")
            print(f"Emotion features shape: {emotion_features.shape}")
            if semantic_features is not None:
                print(f"Semantic features shape: {semantic_features.shape}")

            # Ensure all feature arrays are 2D and have consistent sample size
            features_list = [
                word_features,
                char_features,
                tfidf_features,
                linguistic_features,
                emotion_features,
            ]

            if semantic_features is not None:
                features_list.append(semantic_features)

            num_samples = len(valid_texts)
            for i, feat in enumerate(features_list):
                # Check if feature array is None
                if feat is None:
                    raise ValueError(f"Feature array at index {i} is None")

                # Ensure feature arrays are 2D
                if feat.ndim == 1:
                    feat = feat.reshape(-1, 1)
                    features_list[i] = feat

                # Check if the number of samples matches
                if feat.shape[0] != num_samples:
                    raise ValueError(
                        f"Feature array at index {i} has inconsistent number of samples. Expected {num_samples}, got {feat.shape[0]}"
                    )

            # Combine all features
            all_features = np.hstack(features_list)
            print(f"All features shape after hstack: {all_features.shape}")

            # Scale features
            if not self.is_fitted:
                self.scaler.fit(all_features)
                self.is_fitted = True

            scaled_features = self.scaler.transform(all_features)

            return scaled_features

        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            print(f"Debug info - Input size: {len(texts)}")
            print(f"Debug info - Features shape: {[f.shape for f in features_list if f is not None]}")
            print(
                f"Debug info - Is fitted: {self.is_fitted}")
            print(
                f"Debug info - Vectorizers: word={self.word_vectorizer is not None}, char={self.char_vectorizer is not None}"
            )
            raise

    def _extract_and_scale_features(self, tfidf_features, texts):
        """Extract and scale features using SVD"""
        n_components = min(
            tfidf_features.shape[1]-1,
            len(texts)-1,
            self.config.MAX_FEATURES
        )
        self.svd.n_components = n_components
        
        svd_features = self.svd.fit_transform(tfidf_features)
        scaled_features = self.scaler.fit_transform(svd_features)
        
        return scaled_features

    def _extract_statistical_features(self, texts):
        """Enhanced statistical features for better sentiment detection"""
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
            exclamation_count = text.count("!")
            question_count = text.count("?")
            emoji_count = len(re.findall(r"[\U0001F300-\U0001F9FF]", text))

            # Word patterns
            caps_word_count = len([w for w in words if w.isupper()])
            word_length_variance = np.var([len(w) for w in words]) if words else 0

            # Sentiment patterns
            positive_words = len([w for w in words if w in self.positive_words])
            negative_words = len([w for w in words if w in self.negative_words])

            # Additional sentiment-specific features
            exclamation_sequences = len(re.findall(r"!+", text))
            question_sequences = len(re.findall(r"\?+", text))
            uppercase_words = sum(1 for word in words if word.isupper())
            word_count = len(words)

            # Emotional pattern features
            positive_emoticons = len(re.findall(r"[:;]-?[\)pP]", text))
            negative_emoticons = len(re.findall(r"[:;]-?[\(]", text))
            emoji_pattern = len(re.findall(r"[\U0001F600-\U0001F64F]", text))

            # Sentiment word ratios
            positive_ratio = sum(
                1 for word in words if word in self.positive_words
            ) / max(word_count, 1)
            negative_ratio = sum(
                1 for word in words if word in self.negative_words
            ) / max(word_count, 1)

            features.append(
                [
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
                    negative_words / max(len(words), 1),
                    exclamation_sequences,
                    question_sequences,
                    uppercase_words / max(word_count, 1),
                    positive_emoticons,
                    negative_emoticons,
                    emoji_pattern,
                    positive_ratio,
                    negative_ratio,
                ]
            )

        return np.array(features, dtype=np.float32)

    def _extract_linguistic_features(self, texts):
        """Extract linguistic features from texts"""
        features = []
        for text in texts:
            text = str(text)
            words = text.split()
            
            # Syntactic features
            sentence_count = len([s for s in text.split('.') if len(s.strip()) > 0])
            avg_words_per_sentence = len(words) / max(sentence_count, 1)
            avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
            
            # Basic features
            punctuation_ratio = sum(c in '.,!?;:' for c in text) / max(len(text), 1)
            capital_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
            
            # Stop words ratio
            stop_words = self.config.LANGUAGE_CONFIGS[self.language]["stop_words"]
            if isinstance(stop_words, str) and stop_words == "english":
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
            else:
                stop_words = set(stop_words)
            stop_word_ratio = sum(w.lower() in stop_words for w in words) / max(len(words), 1)
            
            # Combine basic features
            feature_vector = [
                sentence_count,
                avg_words_per_sentence,
                avg_word_length,
                punctuation_ratio,
                capital_ratio,
                stop_word_ratio
            ]
            
            features.append(feature_vector)
            
        return np.array(features, dtype=np.float32)

    def _extract_emotion_features(self, texts):
        """Extract emotion-based features from texts"""
        features = []
        for text in texts:
            text = str(text).lower()
            words = text.split()
            
            # Get emotion keywords for current language
            emotion_keywords = self.config.EMOTION_KEYWORDS.get(self.language, {})
            
            # Calculate emotion scores
            emotion_scores = []
            for emotion, keywords in sorted(emotion_keywords.items()):  # Sort for consistent order
                score = sum(1 for word in words if word in keywords)
                emotion_scores.append(score / max(len(words), 1))
            
            # Additional emotion indicators
            exclamation_ratio = text.count('!') / max(len(text), 1)
            question_ratio = text.count('?') / max(len(text), 1)
            emoji_ratio = len(re.findall(r'[\U0001F300-\U0001F9FF]', text)) / max(len(text), 1)
            
            # Combine all emotion features
            feature_vector = [
                *emotion_scores,
                exclamation_ratio,
                question_ratio,
                emoji_ratio
            ]
            features.append(feature_vector)
            
        return np.array(features, dtype=np.float32)

    def _extract_semantic_features(self, texts):
        """Extract semantic features from texts if available"""
        if not hasattr(self, 'word_vectors'):
            return None
            
        vector_size = getattr(self.word_vectors, 'vector_size', 100)
        features = []
        
        for text in texts:
            words = str(text).lower().split()
            word_vectors = []
            
            for word in words:
                try:
                    if word in self.word_vectors:
                        word_vectors.append(self.word_vectors[word])
                except:
                    continue
                    
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(vector_size)
                
            features.append(avg_vector)
            
        return np.array(features, dtype=np.float32)
