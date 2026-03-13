"""
Edge-Device NLP Preprocessor
Lightweight text cleaning & feature extraction — no external downloads needed.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# Minimal stopword list (no NLTK needed)
# ─────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "how", "when", "where", "why", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each", "few",
    "more", "most", "other", "some", "such", "than", "too", "very", "just",
    "because", "as", "until", "while", "although", "though", "since", "if"
}

# Spam trigger patterns
SPAM_PATTERNS = [
    r'\b(free|win|winner|won|prize|claim|reward)\b',
    r'\b(click here|click now|click below)\b',
    r'\b(buy now|order now|shop now|act now)\b',
    r'\b(limited time|limited offer|expires?)\b',
    r'\b(100%|guaranteed|money back)\b',
    r'\b(earn \$|make money|extra income|work from home)\b',
    r'\b(nigerian?|inheritance|million dollar)\b',
    r'\b(verify your account|confirm your (identity|account|email))\b',
    r'\b(password|username|login|sign.?in)\b.*\b(required|needed|update)\b',
    r'(http[s]?://\S+)',          # URLs
    r'(\$\d+[\.,]?\d*)',          # Dollar amounts
    r'([A-Z]{4,})',               # Excessive caps
    r'(!{2,})',                   # Multiple exclamations
    r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',  # Phone numbers
]

# Toxic/hate speech patterns
TOXIC_PATTERNS = [
    r'\b(hate|kill|die|murder|attack|destroy)\b',
    r'\b(idiot|stupid|moron|dumb|loser|worthless)\b',
    r'\b(racist?|sexist?|bigot|prejudice)\b',
    r'\b(threat|threaten|violence|violent)\b',
    r'\b(ugly|disgusting|horrible|terrible|awful)\b.*\b(you|your|him|her|them)\b',
]

# Obfuscation patterns (l33t speak etc.)
OBFUSCATION_MAP = {
    '@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's',
    '5': 's', '7': 't', '4': 'a', '+': 't', '!': 'i'
}


class TextPreprocessor:
    """Lightweight text preprocessor optimised for edge devices."""

    def __init__(self, remove_stopwords: bool = True,
                 normalize_unicode: bool = True,
                 decode_obfuscation: bool = True):
        self.remove_stopwords = remove_stopwords
        self.normalize_unicode = normalize_unicode
        self.decode_obfuscation = decode_obfuscation
        self._spam_re = [re.compile(p, re.IGNORECASE) for p in SPAM_PATTERNS]
        self._toxic_re = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    # ── Core cleaning ─────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Full cleaning pipeline."""
        if not text or not isinstance(text, str):
            return ""
        text = self._normalize_unicode(text)
        text = self._decode_obfuscation(text)
        text = self._remove_html(text)
        text = self._expand_contractions(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        # Keep apostrophes inside words, remove other punct
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def get_features(self, text: str) -> Dict:
        """Extract hand-crafted features for edge inference."""
        raw = text
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)

        return {
            # Length features
            "char_count": len(raw),
            "word_count": len(tokens),
            "avg_word_len": (sum(len(t) for t in tokens) / max(len(tokens), 1)),
            "sentence_count": len(re.split(r'[.!?]+', raw)),

            # Lexical features
            "unique_word_ratio": len(set(tokens)) / max(len(tokens), 1),
            "digit_ratio": sum(c.isdigit() for c in raw) / max(len(raw), 1),
            "upper_ratio": sum(c.isupper() for c in raw) / max(len(raw), 1),
            "punct_ratio": sum(c in string.punctuation for c in raw) / max(len(raw), 1),

            # Spam signals
            "exclamation_count": raw.count('!'),
            "question_count": raw.count('?'),
            "url_count": len(re.findall(r'https?://\S+', raw)),
            "email_count": len(re.findall(r'\S+@\S+\.\S+', raw)),
            "dollar_count": len(re.findall(r'\$\d+', raw)),
            "all_caps_word_count": len(re.findall(r'\b[A-Z]{3,}\b', raw)),
            "spam_pattern_hits": self._count_pattern_hits(raw, self._spam_re),

            # Toxic signals
            "toxic_pattern_hits": self._count_pattern_hits(raw, self._toxic_re),
            "has_profanity_proxy": int(bool(re.search(
                r'\b(f+u+c+k+|s+h+i+t+|b+i+t+c+h+|a+s+s+h+o+l+e+)\b',
                raw, re.I))),

            # Structural features
            "has_url": int(bool(re.search(r'https?://', raw))),
            "has_email": int(bool(re.search(r'\S+@\S+\.\S+', raw))),
            "has_phone": int(bool(re.search(
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', raw))),
            "repeated_chars": len(re.findall(r'(.)\1{2,}', raw)),
            "line_count": raw.count('\n') + 1,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _normalize_unicode(self, text: str) -> str:
        if not self.normalize_unicode:
            return text
        return unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore').decode('ascii')

    def _decode_obfuscation(self, text: str) -> str:
        if not self.decode_obfuscation:
            return text
        result = []
        for ch in text:
            result.append(OBFUSCATION_MAP.get(ch, ch))
        return ''.join(result)

    def _remove_html(self, text: str) -> str:
        return re.sub(r'<[^>]+>', ' ', text)

    def _expand_contractions(self, text: str) -> str:
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for k, v in contractions.items():
            text = text.replace(k, v)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text)

    def _count_pattern_hits(self, text: str, patterns: list) -> int:
        return sum(1 for p in patterns if p.search(text))

    def features_to_vector(self, features: Dict) -> List[float]:
        """Ordered list for ML model input."""
        keys = [
            "char_count", "word_count", "avg_word_len", "sentence_count",
            "unique_word_ratio", "digit_ratio", "upper_ratio", "punct_ratio",
            "exclamation_count", "question_count", "url_count", "email_count",
            "dollar_count", "all_caps_word_count", "spam_pattern_hits",
            "toxic_pattern_hits", "has_profanity_proxy",
            "has_url", "has_email", "has_phone",
            "repeated_chars", "line_count"
        ]
        return [float(features.get(k, 0)) for k in keys]
