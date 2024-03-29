import os
import re
import string
import logging
from collections import Counter, OrderedDict
from typing import List, Dict

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import (
    EnglishStemmer,
    FrenchStemmer,
    SpanishStemmer,
    ArabicStemmer,
    PortugueseStemmer
)

from langdetect import detect

logging.getLogger().setLevel(logging.INFO)

nltk.data.path.append('/nltk_data')

# Download the nltk packages if required
if not os.path.isdir("/nltk_data/corpora"):
    nltk.download("stopwords")
if not os.path.isdir("/nltk_data/tokenizers"):
    nltk.download("punkt")


class NGramsGenerator:
    """
    Class to generate n-grams from the texts
    """
    def __init__(self,
        max_ngrams_items: int=10,
        generate_unigrams: bool=True,
        generate_bigrams: bool=True,
        generate_trigrams: bool=True,
        enable_stopwords: bool=True,
        enable_stemming: bool=False,
        enable_case_sensitive: bool=True,
        enable_end_of_sentence: bool=True
    ):
        self.stopwords_ar = stopwords.words("arabic")
        self.stopwords_en = stopwords.words("english")
        self.stopwords_fr = stopwords.words("french")
        self.stopwords_es = stopwords.words("spanish")
        self.stopwords_pt = stopwords.words("portuguese")

        stemmer_ar = ArabicStemmer()
        stemmer_en = EnglishStemmer()
        stemmer_fr = FrenchStemmer()
        stemmer_es = SpanishStemmer()
        stemmer_pt = PortugueseStemmer()

        self.max_ngrams_items = max_ngrams_items
        self.generate_unigrams = generate_unigrams
        self.generate_bigrams = generate_bigrams
        self.generate_trigrams = generate_trigrams
        self.enable_stopwords = enable_stopwords
        self.enable_stemming = enable_stemming
        self.enable_case_sensitive = enable_case_sensitive
        self.enable_end_of_sentence = enable_end_of_sentence

        self.allowed_languages = ["ar", "en", "fr", "es", "pt"]

        self.language_mapper = {
            "ar": "arabic",
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "pt": "portuguese"
        }

        self.language_fn_mapper = {
            "ar": {
                "stopwords": self.stopwords_ar,
                "stemmer": stemmer_ar
            },
            "en": {
                "stopwords": self.stopwords_en,
                "stemmer": stemmer_en
            },
            "fr": {
                "stopwords": self.stopwords_fr,
                "stemmer": stemmer_fr
            },
            "es": {
                "stopwords": self.stopwords_es,
                "stemmer": stemmer_es
            },
            "pt": {
                "stopwords": self.stopwords_pt,
                "stemmer": stemmer_pt
            }
        }

        self.fn_stemmer = lambda tokens, lang: [self.language_fn_mapper[lang]["stemmer"].stem(w) for w in tokens]

    def detect_language(self, entry: str)->str:
        """
        Detects language of the text
        """
        try:
            lang = detect(entry)
            if lang not in self.allowed_languages:
                logging.warning("%s not found in allowed languages. Using english(en) instead.", lang)
                return "en"
            return lang
        except Exception as exc:
            logging.warning("%s Using english(en) instead.", str(exc))
        return "en"

    def handle_currency(self, entry_lst: List)-> List:
        """
        Handles currency
        """
        is_prev_token_merged = 0
        cur = ["$", "£", "€", "¥", "₹", "रु", "₣"]
        processed_lst = []
        try:
            for i, token in enumerate(entry_lst):
                if is_prev_token_merged:
                    is_prev_token_merged = 0
                    continue
                if token in cur and entry_lst[i+1].isnumeric():
                    processed_lst.append(token + entry_lst[i+1])
                    is_prev_token_merged = 1
                else:
                    processed_lst.append(token)
                    is_prev_token_merged = 0
            return processed_lst
        except IndexError:
            return entry_lst


    def get_punctuations(self)-> Dict:
        """
        Returns commonly used punctuations
        """
        return string.punctuation

    def clean_extra_punctuations(self, text):
        """
        Clean the not so common extra punctuations
        """
        pattern = re.compile("[^a-zA-Z0-9_?.!'£$£€¥₹रु₣ ]")
        return re.sub(pattern, "", text)

    def clean_sentence_level(
        self,
        entry: str,
        language: str,
        return_tokens: bool=True
    ):
        """
        Cleans the texts based on input parameters at sentence level
        """
        entry = entry.strip()
        entry = self.clean_extra_punctuations(entry)
        sent_tokens = sent_tokenize(entry, language=self.language_mapper.get(language, "english"))
        entry_tokens_lst = [word_tokenize(sent, language=self.language_mapper.get(language, "english")) for sent in sent_tokens]

        entry_tokens_lst = [self.handle_currency(entry_tokens) for entry_tokens in entry_tokens_lst]
        entry_tokens_lst = [[w for w in entry_tokens if w not in self.get_punctuations()] for entry_tokens in entry_tokens_lst] # Removes the punctuation from the sentence

        if self.enable_stopwords:
            entry_tokens_lst = [[w for w in entry_tokens if w.lower() not in self.language_fn_mapper[language]["stopwords"]] for entry_tokens in entry_tokens_lst]
        if self.enable_stemming:
            entry_tokens_lst = [self.fn_stemmer(entry_tokens, language) for entry_tokens in entry_tokens_lst]
        if not self.enable_case_sensitive:
            entry_tokens_lst = [[w.lower() for w in entry_tokens] for entry_tokens in entry_tokens_lst]

        if return_tokens:
            return entry_tokens_lst
        return " ".join(entry_tokens_lst)

    def clean_entry(
        self,
        entry: str,
        language: str,
        return_tokens: bool=True
    ):
        """
        Cleans the texts based on input parameters
        """
        entry = entry.strip()
        entry = self.clean_extra_punctuations(entry)
        entry_tokens = word_tokenize(entry, language=self.language_mapper.get(language, "english"))
        entry_tokens = self.handle_currency(entry_tokens)
        entry_tokens = [w for w in entry_tokens if w not in self.get_punctuations()] # Removes the punctuation from the sentence

        if self.enable_stopwords:
            entry_tokens = [w for w in entry_tokens if w.lower() not in self.language_fn_mapper[language]["stopwords"]]
        if self.enable_stemming:
            entry_tokens = self.fn_stemmer(entry_tokens, language)
        if not self.enable_case_sensitive:
            entry_tokens = [w.lower() for w in entry_tokens]

        if return_tokens:
            return entry_tokens
        return " ".join(entry_tokens)


    def get_ngrams(
        self,
        entries: List[str],
        n: int=1
    ):
        """
        Calculates the n-grams
        """
        if self.enable_end_of_sentence:
            ngrams_op = [[ngrams(entry_tokens, n) for entry_tokens in entry] for entry in entries]
            ngrams_lst = [[list(x) for x in n] for n in ngrams_op]
            # Flatten the list of list
            ngrams_flat_lst = [item for sublist in ngrams_lst for item in sublist]
            ngrams_flat_lst = [item for sublist in ngrams_flat_lst for item in sublist]
        else:
            ngrams_op = [ngrams(entry_tokens, n) for entry_tokens in entries]
            ngrams_lst = [list(x) for x in ngrams_op]
            # Flatten the list of list
            ngrams_flat_lst = [item for sublist in ngrams_lst for item in sublist]
        return Counter(ngrams_flat_lst).most_common(self.max_ngrams_items)

    def __call__(
        self,
        entries: List[str]
    )->Dict[str, Dict]:
        """
        Main handler to generate n-grams
        """
        ngrams = {}
        processed_entries = []
        for entry in entries:
            if entry.strip() == "":
                continue
            detected_language = self.detect_language(entry)
            processed_entries.append(
                self.clean_sentence_level(entry, language=detected_language)
                    if self.enable_end_of_sentence else
                    self.clean_entry(entry, language=detected_language)
            )

        if self.generate_unigrams:
            unigrams = self.get_ngrams(processed_entries, n=1)
            ngrams["unigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(unigrams).items()
            })

        if self.generate_bigrams:
            bigrams = self.get_ngrams(processed_entries, n=2)
            ngrams["bigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(bigrams).items()
            })

        if self.generate_trigrams:
            trigrams = self.get_ngrams(processed_entries, n=3)
            ngrams["trigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(trigrams).items()
            })

        return ngrams
