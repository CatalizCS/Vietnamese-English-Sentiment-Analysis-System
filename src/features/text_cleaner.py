import re
from underthesea import word_tokenize  # For Vietnamese
from nltk.tokenize import word_tokenize as en_tokenize
from nltk.corpus import stopwords


class TextCleaner:
    """Text cleaning and preprocessing class"""
    
    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.stop_words = self._get_stop_words()

    def _get_stop_words(self):
        return self.config.LANGUAGE_CONFIGS[self.language]["stop_words"]

    def clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove HTML
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special chars and digits 
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()

        # Tokenize based on language
        if self.language == "vi":
            tokens = word_tokenize(text)
        else:
            tokens = en_tokenize(text)

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        return " ".join(tokens)
