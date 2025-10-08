"""
A Simple NLP class for processing text data
"""

from typing import List, Iterable, Optional

class SimpleNLP:
    """
    Minimal NLP utilities (no regex, no external libs).
    Features:
      1) Remove stop words
      2) Document -> sentences
      3) Sentence -> word tokens

    Design choices:
      - Unicode-aware via str.isalnum().
      - Keeps internal apostrophes/hyphens inside tokens (e.g., "don't", "rock-n-roll").
      - Simple sentence splitter on ., !, ?, … with a small abbreviation skip list.
    """

    DEFAULT_STOPWORDS = {
        "a","an","the","and","or","but","if","then","else","for","nor","so","yet",
        "of","in","on","at","to","from","by","with","about","as","into","like","through",
        "is","am","are","was","were","be","been","being",
        "this","that","these","those","it","its","itself",
        "i","me","my","myself","we","our","ours","ourselves",
        "you","your","yours","yourself","yourselves",
        "he","him","his","himself","she","her","hers","herself",
        "they","them","their","theirs","themselves",
        "what","which","who","whom","whose","where","when","why","how",
        "do","does","did","doing","done",
        "have","has","had","having",
        "not","no","yes","too","very","can","could","should","would","will","shall",
        "also","just","only","than","such","more","most","much","many",
    }

    # A tiny set; add more abbreviations as needed.
    ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "e.g.", "i.e.", "etc."}

    def __init__(self,
                 stopwords: Optional[Iterable[str]] = None,
                 *,
                 lowercase: bool = True,
                 keep_contractions: bool = True):
        self.lowercase = lowercase
        self.keep_contractions = keep_contractions
        self.stopwords = set((w.lower() if lowercase else w) for w in (stopwords or self.DEFAULT_STOPWORDS))


    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove tokens that appear in the stopword set.
        """

        if self.lowercase:
            return [t for t in tokens if t not in self.stopwords]
        else:
            # If case-insensitively
            return [t for t in tokens if t.lower() not in self.stopwords]
