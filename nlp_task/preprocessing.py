# preprocessing.py
import spacy
import re
from sklearn.base import BaseEstimator, TransformerMixin

# Load spaCy model once
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Regex patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        text = URL_PATTERN.sub('', text)
        text = MENTION_PATTERN.sub('', text)
        text = HASHTAG_PATTERN.sub('', text)
        text = EMOJI_PATTERN.sub('', text)
        return text

    def transform(self, X):
        processed = []
        for doc in nlp.pipe([self.clean_text(text) for text in X], batch_size=1000, n_process=-1):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.like_num and token.is_alpha
            ]
            processed.append(" ".join(tokens))
        return processed
