import random
import re
from typing import List, Dict


class TextAugmenter:
    def __init__(self):
        self.common_typos = {
            "a": "aq",
            "b": "vb",
            "c": "xc",
            "d": "sd",
            "e": "er",
            "h": "gh",
            "i": "ui",
            "k": "jk",
            "m": "nm",
            "n": "bn",
            "o": "io",
            "p": "op",
            "r": "er",
            "s": "as",
            "t": "rt",
            "u": "yu",
            "v": "cv",
            "w": "qw",
            "y": "ty",
        }

        self.vi_variations = {
            "vâng": ["vang", "vânggg", "vâg", "uk"],
            "không": ["khong", "hông", "ko", "k", "khg"],
            "rất": ["rat", "rất là", "rất chi là"],
            "tốt": ["tot", "tốt", "tuyệt"],
            "được": ["dc", "đc", "được"],
            "biết": ["bít", "bik", "biết"],
            "thích": ["thik", "thích", "thíchhh"],
            "quá": ["qá", "quá trời", "quớ"],
        }

        self.en_variations = {
            "yes": ["yep", "yeah", "yup", "yas"],
            "no": ["nope", "nah", "noway"],
            "very": ["rlly", "rly", "super", "v"],
            "good": ["gud", "noice", "gr8"],
            "thanks": ["thx", "tks", "ty"],
            "please": ["pls", "plz", "plss"],
            "love": ["luv", "luvv", "lovee"],
            "cool": ["kewl", "noice", "lit"],
        }

        self.emojis = {
            "positive": ["😊", "😄", "👍", "❤️", "🥰", "💯", "✨"],
            "negative": ["😢", "😞", "👎", "😠", "😕", "💔", "😫"],
            "neutral": ["🤔", "😐", "💭", "👀", "🆗", "💁"],
        }

        from src.utils.templates import CommentTemplates

        self.templates = CommentTemplates()

    def _apply_typos(self, word: str) -> str:
        if len(word) < 4 or random.random() > 0.3:  # 30% chance for typo
            return word

        pos = random.randint(1, len(word) - 1)
        chars = list(word)
        if chars[pos] in self.common_typos:
            chars[pos] = random.choice(list(self.common_typos[chars[pos]]))
        return "".join(chars)

    def _get_variation(self, word: str, language: str) -> str:
        variations = self.vi_variations if language == "vi" else self.en_variations
        return variations.get(word.lower(), [word])[0]

    def _add_emojis(self, text: str, sentiment: str = "neutral") -> str:
        if random.random() > 0.3:  # 30% chance to add emoji
            return text
        emoji = random.choice(self.emojis[sentiment])
        position = random.choice(["prefix", "suffix"])
        return f"{emoji} {text}" if position == "prefix" else f"{text} {emoji}"

    def humanize_text(
        self, text: str, language: str = "vi", sentiment: str = "neutral"
    ) -> str:
        """Create more natural text variations"""
        words = text.split()
        result = []

        for word in words:
            # Apply variations
            if random.random() < 0.3:  # 30% chance for informal variations
                word = self._get_variation(word, language)
            elif random.random() < 0.2:  # 20% chance for typos
                word = self._apply_typos(word)

            # Random repeated letters (excitement/emphasis)
            if random.random() < 0.1:  # 10% chance
                if word[-1] in "aeiouydg":
                    word = word + word[-1] * random.randint(1, 3)

            result.append(word)

        text = " ".join(result)

        # Add emojis based on sentiment
        text = self._add_emojis(text, sentiment)

        # Random punctuation variations
        if random.random() < 0.2:  # 20% chance
            text = text + random.choice(["!!!", "..", "?!", "!"])

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization that works for both English and Vietnamese"""
        return text.split()

    def synonym_replacement(self, text: str) -> str:
        """Simple word variation by character substitution"""
        words = self._tokenize(text)
        if len(words) < 2:
            return text

        n = max(1, int(len(words) * 0.1))  # Replace 10% of words
        indexes = random.sample(range(len(words)), min(n, len(words)))

        for i in indexes:
            word = words[i]
            if len(word) > 3:
                # Simple character substitution instead of using WordNet
                pos = random.randint(1, len(word) - 2)
                chars = list(word)
                chars[pos] = random.choice(
                    [c for c in "abcdefghijklmnopqrstuvwxyz" if c != chars[pos]]
                )
                words[i] = "".join(chars)

        return " ".join(words)

    def random_swap(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        n = max(1, int(len(words) * 0.1))  # Swap 10% of words
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def random_deletion(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        keep_prob = 0.9  # Keep 90% of words
        words = [word for word in words if random.random() < keep_prob]

        return " ".join(words) if words else text

    def random_insertion(self, text: str) -> str:
        words = text.split()
        n = max(1, int(len(words) * 0.1))  # Insert 10% new words

        for _ in range(n):
            if not words:
                break
            # Insert a random word from the text at a random position
            word_to_insert = random.choice(words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word_to_insert)

        return " ".join(words)

    def _add_expression(
        self, comment: str, sentiment: str, language: str = "vi"
    ) -> str:
        """Thêm biểu thức cảm xúc vào bình luận"""
        if random.random() > 0.3:  # 30% chance
            return comment

        expressions = (
            self.templates.vi_expressions
            if language == "vi"
            else self.templates.en_expressions
        )
        expr = random.choice(expressions[sentiment])

        position = random.choice(["prefix", "suffix"])
        if position == "prefix":
            return f"{expr}, {comment.lower()}"
        return f"{comment}, {expr}"

    def _apply_internet_slang(self, text: str, language: str) -> str:
        """Apply internet slang to text"""
        if random.random() > 0.4:  # 40% chance to use internet slang
            return text

        words = text.split()
        result = []

        for word in words:
            if random.random() < 0.3:  # 30% chance per word
                slang_word = self.templates.get_internet_term(word, language)
                result.append(slang_word)
            else:
                result.append(word)

        return " ".join(result)

    def _randomly_combine_expressions(self, text: str, sentiment: str, language: str) -> str:
        """Randomly combine multiple expressions and slang terms"""
        if random.random() > 0.3:
            return text
            
        num_expressions = random.randint(1, 3)  # Add 1-3 expressions
        expressions = []
        
        for _ in range(num_expressions):
            expr_type = random.choice(['slang', 'expression', 'internet_term'])
            if expr_type == 'slang':
                category = random.choice(list(self.templates.vi_slangs[sentiment].keys()))
                expr = self.templates.get_random_slang(sentiment, category, language)
            elif expr_type == 'expression':
                expr = random.choice(self.templates.vi_expressions[sentiment])
            else:
                word = random.choice(list(self.templates.vi_slangs['internet_terms'].keys()))
                expr = self.templates.get_internet_term(word, language)
                
            if expr:
                expressions.append(expr)
                
        if expressions:
            position = random.choice(['prefix', 'both', 'suffix'])
            if position == 'prefix':
                return f"{' '.join(expressions)} {text}"
            elif position == 'suffix':
                return f"{text} {' '.join(expressions)}"
            else:
                return f"{expressions[0]} {text} {' '.join(expressions[1:])}"
        return text

    def _add_random_punctuation(self, text: str, sentiment: str) -> str:
        """Add random punctuation based on sentiment"""
        if random.random() > 0.4:
            return text
            
        punct_patterns = {
            'positive': ['!!!', '!?!', '...!!!', '!!! <3'],
            'negative': ['...', '!?', '(?)', '>.<'],
            'neutral': ['...', '.', '...?', '(?!)']
        }
        
        num_puncts = random.randint(1, 3)
        puncts = [random.choice(punct_patterns[sentiment]) for _ in range(num_puncts)]
        return f"{text}{''.join(puncts)}"

    def generate_realistic_comment(
        self, topic: str, sentiment: str, language: str = "vi"
    ) -> str:
        """Generate a realistic comment for a given topic and sentiment"""
        templates = (
            self.templates.vi_templates
            if language == "vi"
            else self.templates.en_templates
        )
        fillers = (
            self.templates.vi_fillers if language == "vi" else self.templates.en_fillers
        )

        if topic not in templates:
            topic = list(templates.keys())[0]  # default to first topic

        topic_templates = templates[topic][sentiment]
        template = random.choice(topic_templates)

        # Generate fillers for the template
        filled_comment = template
        for placeholder in re.findall(r"\{(\w+)\}", template):
            if placeholder in fillers:
                filler = random.choice(fillers[placeholder])
                filled_comment = filled_comment.replace(f"{{{placeholder}}}", filler)
            else:
                # Generic fillers for undefined placeholders
                filled_comment = filled_comment.replace(f"{{{placeholder}}}", "")

        # Add variations and humanization
        filled_comment = self.humanize_text(filled_comment, language, sentiment)

        # Add slang and internet speech variations
        if random.random() < 0.4:  # 40% chance to use slang
            filled_comment = self._apply_internet_slang(filled_comment, language)

            # Add random slang intensifier
            if random.random() < 0.3:
                intensifier = self.templates.get_random_intensifier(language)
                filled_comment = f"{intensifier} {filled_comment}"

        # Add expressions and emojis
        filled_comment = self._add_expression(filled_comment, sentiment, language)
        filled_comment = self._add_emojis(filled_comment, sentiment)

        # Random variations in punctuation and emphasis
        if random.random() < 0.3:
            filled_comment = filled_comment + random.choice(["!!!", "..", "?!", "!!"])

        # Randomize comment structure
        if random.random() < 0.3:
            # Add random topic-specific context
            contexts = {
                'product_review': ['Mới mua', 'Dùng được 1 tuần', 'Đặt trên shopee'],
                'food_review': ['Ghé quán hôm qua', 'Đi ăn với bạn', 'Oder mang về'],
                'movie_review': ['Xem buổi chiều', 'Ra rạp coi', 'Xem trên Netflix'],
                'service_review': ['Làm dịch vụ hôm qua', 'Mới trải nghiệm', 'Book lịch']
            }
            if topic in contexts:
                context = random.choice(contexts[topic])
                filled_comment = f"{context}, {filled_comment}"

        # Add random expressions and slang combinations
        filled_comment = self._randomly_combine_expressions(filled_comment, sentiment, language)
        
        # Add varied punctuation
        filled_comment = self._add_random_punctuation(filled_comment, sentiment)

        return filled_comment

    def generate_topic_comment(self, topic: str, sentiment: str, language: str = 'vi') -> str:
        """Generate a comment for a specific topic and sentiment"""
        if not topic.endswith('_review'):
            topic = f"{topic}_review"
            
        return self.generate_realistic_comment(topic, sentiment, language)

    def generate_topic_comments(self, topic: str, count: int = 10, language: str = 'vi', sentiment: int = None) -> list:
        """Generate comments for a specific topic with optional sentiment"""
        comments = []
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        for _ in range(count):
            # If sentiment is provided, use it, otherwise randomly choose
            if sentiment is not None:
                sent = sentiment_map[sentiment]
            else:
                sent = random.choice(['negative', 'neutral', 'positive'])
                
            text = self.generate_topic_comment(topic, sent, language)
            label = {'negative': 0, 'neutral': 1, 'positive': 2}[sent]
            
            comments.append({
                'text': text,
                'label': label
            })
            
        return comments
