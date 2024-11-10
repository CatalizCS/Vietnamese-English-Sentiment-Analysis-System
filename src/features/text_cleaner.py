import re
from underthesea import word_tokenize  # For Vietnamese
from nltk.tokenize import word_tokenize as en_tokenize
from nltk.corpus import stopwords


class TextCleaner:
    def __init__(self, language: str, config):
        self.language = language
        self.config = config
        self.stop_words = self._get_stop_words()

    def _get_stop_words(self):
        return self.config.LANGUAGE_CONFIGS[self.language]["stop_words"]

    def clean_text(self, text: str) -> str:
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove special characters and digits
        text = re.sub(r"[^\w\s]", "", text)

        # Convert to lowercase
        text = text.lower()

        # Tokenization based on language
        if self.language == "vi":
            tokens = word_tokenize(text)
        else:
            tokens = en_tokenize(text)

        # Remove stop words
        if self.language == "en":
            stop_words = set(stopwords.words(self.stop_words))
        else:
            stop_words = set(self.stop_words)

        tokens = [t for t in tokens if t not in stop_words]

        return " ".join(tokens)
