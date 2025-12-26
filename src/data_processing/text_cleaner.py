"""
Text Cleaning Utilities
"""
import re
import string
from typing import List
class TextCleaner:
    def __init__(self):
        self.stopwords = self._get_stopwords()
    def _get_stopwords(self) -> set:
        return {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
            'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
            'being', 'below', 'between', 'both', 'but', 'by', 'the', 'is', 'it'
        }
    def remove_urls(self, text: str) -> str:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    def clean_text(self, text: str, lowercase: bool = True) -> str:
        if not text:
            return ""
        text = self.remove_urls(text)
        if lowercase:
            text = text.lower()
        text = self.normalize_whitespace(text)
        return text
