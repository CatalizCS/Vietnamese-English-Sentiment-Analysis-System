import random
import pandas as pd
import numpy as np
from typing import List, Dict
import nltk
from nltk.corpus import wordnet

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class DataAugmentor:
    def __init__(self, language: str):
        self.language = language
        # Load language-specific augmentation resources
        self.synonyms = self._load_synonyms()
        self.templates = self._load_templates()

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load word synonyms for the specified language"""
        try:
            if self.language == "en":
                # Use WordNet for English
                synonyms = {}
                for synset in wordnet.all_synsets():
                    word = synset.lemmas()[0].name()
                    synonyms[word] = [l.name() for l in synset.lemmas()[1:]]
                return synonyms
            else:
                # Load Vietnamese synonyms from file or resource
                synonyms_file = f"data/resources/{self.language}_synonyms.txt"
                synonyms = {}
                try:
                    with open(synonyms_file, "r", encoding="utf-8") as f:
                        for line in f:
                            words = line.strip().split(",")
                            synonyms[words[0]] = words[1:]
                except FileNotFoundError:
                    return {}  # Return empty dict if file not found
                return synonyms
        except Exception:
            return {}

    def _load_templates(self) -> List[str]:
        """Load text templates for data augmentation"""
        if self.language == "vi":
            return [
                "Tôi thấy {text}",
                "Theo tôi thì {text}",
                "Tôi nghĩ {text}",
                "Tôi cảm thấy {text}",
                "{text} theo ý kiến của tôi",
            ]
        else:
            return [
                "I think {text}",
                "In my opinion {text}",
                "I feel {text}",
                "I believe {text}",
                "{text} in my view",
            ]

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n random words with synonyms"""
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word in self.synonyms]))

        n = min(n, len(random_word_list))
        for _ in range(n):
            if not random_word_list:
                break
            random_word = random.choice(random_word_list)
            random_synonym = random.choice(
                self.synonyms.get(random_word, [random_word])
            )
            random_idx = random.choice(
                [i for i, word in enumerate(new_words) if word == random_word]
            )
            new_words[random_idx] = random_synonym
            random_word_list.remove(random_word)

        return " ".join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random synonyms into random positions"""
        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            if not self.synonyms:
                break
            random_word = random.choice(list(self.synonyms.keys()))
            random_synonym = random.choice(self.synonyms[random_word])
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, random_synonym)

        return " ".join(new_words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words"""
        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            if len(new_words) < 2:
                break
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return " ".join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text

        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)

        if not new_words:
            rand_int = random.randint(0, len(words) - 1)
            new_words.append(words[rand_int])

        return " ".join(new_words)

    def template_based(self, text: str) -> str:
        """Apply random template to the text"""
        template = random.choice(self.templates)
        return template.format(text=text)

    def augment_text(self, text: str, methods: List[str] = None) -> List[str]:
        """Apply multiple augmentation methods to generate variations"""
        if methods is None:
            methods = ["synonym", "insert", "swap", "delete", "template"]

        augmented = []
        for method in methods:
            try:
                if method == "synonym":
                    augmented.append(self.synonym_replacement(text))
                elif method == "insert":
                    augmented.append(self.random_insertion(text))
                elif method == "swap":
                    augmented.append(self.random_swap(text))
                elif method == "delete":
                    augmented.append(self.random_deletion(text))
                elif method == "template":
                    augmented.append(self.template_based(text))
            except Exception:
                continue

        return augmented

    def augment_data(self, df: pd.DataFrame, n_per_sample: int = 2) -> pd.DataFrame:
        """Augment entire dataset"""
        augmented_data = []

        for _, row in df.iterrows():
            text = row["text"]
            label = row["label"]

            # Original sample
            augmented_data.append({"text": text, "label": label})

            # Generate augmented samples
            augmented_texts = self.augment_text(text)
            for aug_text in augmented_texts[:n_per_sample]:
                augmented_data.append({"text": aug_text, "label": label})

        return pd.DataFrame(augmented_data)
