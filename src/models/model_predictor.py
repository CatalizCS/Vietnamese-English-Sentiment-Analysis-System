import os
import joblib
import numpy as np
from scipy import stats
from src.features.feature_engineering import FeatureExtractor
from src.config import Config
from src.utils.logger import Logger


class SentimentPredictor:
    def __init__(self, language: str, config: Config):
        self.language = language
        self.config = config
        self.logger = Logger(__name__).logger
        self.model_path = os.path.join(
            config.DATA_DIR, "models", f"{language}_sentiment_model.pkl"
        )
        self.feature_extractor = FeatureExtractor(language, config)
        self.models = self._load_model()
        self.feature_dims = (
            self.feature_extractor.feature_dims
        )  # Ensure feature_dims is set

    def _load_model(self):
        """Load trained model from file"""
        model_path = os.path.join(
            self.config.DATA_DIR, "models", f"{self.language}_sentiment_model.pkl"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found for {self.language} at {model_path}"
            )

        model_info = joblib.load(model_path)

        if "model" in model_info:
            model = model_info["model"]
            if isinstance(model, dict):
                return {k: v for k, v in model.items() if hasattr(v, "predict")}
            elif hasattr(model, "predict"):
                return {"base": model}

        raise ValueError("No valid models found in model file")

    def predict(self, features):
        """Ensemble prediction using voting with dimension validation"""
        if not self.models:
            raise ValueError("No models available")

        if not isinstance(features, np.ndarray):
            features = np.array(features)

        # Validate feature dimensions
        if self.feature_dims:
            if features.shape[1] != self.feature_dims:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {self.feature_dims}, got {features.shape[1]}"
                )

        predictions = []
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict"):
                    pred = model.predict(features)
                    predictions.append(pred)
            except Exception as e:
                self.logger.error(f"Error in {name} prediction: {str(e)}")
                continue

        if not predictions:
            raise RuntimeError("All models failed to predict")

        # Use majority voting
        stacked_preds = np.vstack(predictions)
        modes, _ = stats.mode(stacked_preds, axis=0)
        return modes.flatten()

    def predict_proba(self, features):
        """Get probability estimates from available models"""
        if not self.models:
            raise ValueError("No models loaded")

        probas = []
        for name, model in self.models.items():
            try:
                # Handle models that don't have predict_proba
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)
                elif hasattr(model, "decision_function"):
                    # Convert decision function to probabilities
                    dec = model.decision_function(features)
                    proba = 1 / (1 + np.exp(-dec))
                    # Normalize to get probabilities
                    proba = proba / proba.sum(axis=1, keepdims=True)
                else:
                    continue
                probas.append(proba)
            except Exception as e:
                self.logger.error(f"Error in {name} probability estimation: {str(e)}")
                continue

        if probas:
            # Average probabilities from all models
            return np.mean(probas, axis=0)

        raise RuntimeError("No models could provide probability estimates")

    def predict_detailed_emotion(self, features):
        """Predict detailed emotions based on sentiment probabilities"""
        sentiments = self.predict(features)
        probabilities = self.predict_proba(features)

        predictions = []
        for sent, probs in zip(sentiments, probabilities):
            # Map sentiment to emotion category
            if sent == 2:
                emotions = self.config.EMOTION_LABELS["POSITIVE"]
            elif sent == 0:
                emotions = self.config.EMOTION_LABELS["NEGATIVE"]
            else:
                emotions = self.config.EMOTION_LABELS["NEUTRAL"]

            # Select random emotion from category with confidence
            emotion = np.random.choice(list(emotions.values()))
            confidence = np.max(probs)

            predictions.append(
                {
                    "sentiment": sent,
                    "sentiment_confidence": confidence,
                    "detailed_emotion": emotion,
                    "emotion_confidence": confidence
                    * 0.8,  # Slightly lower confidence for detailed emotion
                }
            )

        return predictions

    def predict_emotion(self, features, text: str = None):
        """Predict detailed emotion with confidence scores"""
        try:
            # Get base sentiment prediction
            sentiment = self.predict(features)[0]
            probabilities = self.predict_proba(features)[0]
            base_confidence = max(probabilities)

            # Initialize emotion detector if text is provided
            emotion_scores = {}

            if text:
                text = text.lower()
                # Get emotion keywords for the current language
                keywords = self.config.EMOTION_KEYWORDS.get(self.language, {})

                # Calculate emotion scores based on keyword presence
                for emotion, words in keywords.items():
                    score = sum(1 for word in words if word in text)
                    multiplier = 1.0

                    # Adjust score based on punctuation and capitalization
                    if "!" in text:
                        multiplier *= 1.2
                    if text.isupper():
                        multiplier *= 1.3

                    emotion_scores[emotion] = score * multiplier

                # Normalize scores
                total = sum(emotion_scores.values()) or 1
                emotion_scores = {k: v / total for k, v in emotion_scores.items()}

            # Get emotion mapping for the predicted sentiment
            valid_emotions = {
                emotion: details
                for emotion, details in self.config.EMOTION_MAPPING.items()
                if details["sentiment"] == sentiment
            }

            # Select most likely emotion
            if emotion_scores:
                # Filter emotions by base sentiment
                valid_scores = {
                    emotion: score
                    for emotion, score in emotion_scores.items()
                    if emotion in valid_emotions
                }

                if valid_scores:
                    predicted_emotion = max(valid_scores.items(), key=lambda x: x[1])[0]
                else:
                    # Fallback to default emotion for sentiment
                    predicted_emotion = next(iter(valid_emotions))
            else:
                # Fallback if no text analysis
                predicted_emotion = next(iter(valid_emotions))

            # Get emotion details
            emotion_info = self.config.EMOTION_MAPPING[predicted_emotion]

            return {
                "sentiment": sentiment,
                "sentiment_confidence": base_confidence,
                "emotion": predicted_emotion,
                "emotion_vi": emotion_info["vi"],
                "emotion_emoji": emotion_info["emoji"],
                "emotion_confidence": base_confidence
                * 0.8,  # Slightly lower confidence for detailed emotion
                "emotion_scores": emotion_scores if emotion_scores else None,
            }

        except Exception as e:
            self.logger.error(f"Error in emotion prediction: {str(e)}")
            return None

    async def predict_async(self, text: str):
        try:
            self.logger.info(f"Received text for prediction: {text}")
            processed_text = self.feature_extractor.preprocess_text(text)
            self.logger.info(f"Processed text: {processed_text}")
            features = self.feature_extractor.extract_features([processed_text])
            self.logger.info(f"Extracted features: {features.shape}")
            prediction = self.predict(features)
            self.logger.info(f"Prediction result: {prediction}")
            return prediction[0]
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
