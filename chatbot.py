import csv
import re

import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQChatbot:
    """Retrieval-based FAQ bot: NLTK preprocessing + TF-IDF + cosine similarity."""

    _tokenizer = RegexpTokenizer(r"[a-z0-9]+")   # NLTK tokenizer, needs no extra data download
    _stemmer = PorterStemmer()                    # NLTK stemmer ("scholarships" -> "scholarship")

    def __init__(self, csv_path, threshold=0.18):
        self.data = self.load_faqs(csv_path)
        self.questions = self.data["question"].tolist()
        self.answers = self.data["answer"].tolist()
        self.threshold = threshold
        # Tokenization, stop-word removal and stemming happen in self.tokenize.
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2)
        )
        self.question_matrix = self.vectorizer.fit_transform(self.questions)

    @staticmethod
    def load_faqs(csv_path):
        """Read the FAQ CSV and fail with a clear message if it is malformed."""
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows or [c.strip().lower() for c in rows[0]] != ["question", "answer"]:
            raise ValueError(f"{csv_path}: the header must be exactly: question,answer")
        cleaned = []
        for line_no, row in enumerate(rows[1:], start=2):
            if not any(cell.strip() for cell in row):
                continue  # skip blank lines
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(
                    f"{csv_path}, line {line_no}: expected exactly 2 non-empty fields (question, answer) "
                    f"but found {len(row)}. If an answer contains a comma, wrap the whole answer in double quotes."
                )
            cleaned.append((row[0].strip(), row[1].strip()))
        if not cleaned:
            raise ValueError(f"{csv_path}: no FAQ rows found")
        return pd.DataFrame(cleaned, columns=["question", "answer"])

    @classmethod
    def tokenize(cls, text):
        tokens = cls._tokenizer.tokenize(str(text).lower())
        return [cls._stemmer.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS]

    @staticmethod
    def clean(text):
        return re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower()).strip()

    def answer(self, user_question):
        user_vector = self.vectorizer.transform([self.clean(user_question)])
        scores = cosine_similarity(user_vector, self.question_matrix).flatten()
        best_index = int(scores.argmax())
        score = float(scores[best_index])
        matched = score >= self.threshold
        return {
            "matched": matched,
            "answer": self.answers[best_index] if matched else None,
            "matched_question": self.questions[best_index] if matched else None,
            "score": score,
        }
