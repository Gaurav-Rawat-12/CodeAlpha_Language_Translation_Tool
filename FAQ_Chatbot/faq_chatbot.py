"""
FAQ Chatbot — Task 2
=====================
Pipeline:
  1. Collect FAQs from a CSV (question/answer pairs)
  2. Preprocess text with NLTK (tokenize, clean, remove stopwords, lemmatize)
  3. Match user questions to the most similar FAQ via TF-IDF + cosine similarity
  4. Display the best-matching answer
  5. Optional: simple command-line chat UI (see chat_loop / __main__)

Works out of the box with the Bitext Gen AI Chatbot Customer Support Dataset
(columns: flags, instruction, category, intent, response) but also auto-detects
plain "question"/"answer" style CSVs.
"""

import re
import string
import sys
from typing import Optional

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data() -> None:
    """Download the small NLTK corpora needed for preprocessing (no-op if cached)."""
    lookups = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for pkg, find_path in lookups.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # older/newer NLTK versions only need one of the two tagger names


ensure_nltk_data()
_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

_WORDNET_POS_MAP = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS lemmatize() expects."""
    return _WORDNET_POS_MAP.get(treebank_tag[0], wordnet.NOUN)


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, tokenize, drop stopwords, and POS-aware lemmatize.

    POS-aware lemmatization matters here: without it, "shipping" (noun) and
    "ship" (verb), or "internationally" (adverb) and "international" (adjective),
    are treated as unrelated tokens and TF-IDF misses the match. Tagging each
    token's part of speech before lemmatizing collapses these back together.
    """
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tag the FULL token list first (including stopwords) so the tagger has
    # real sentence context, then drop stopwords afterwards. Tagging after
    # stopword removal starves the tagger of context and degrades accuracy.
    all_tokens = word_tokenize(text)
    tagged = pos_tag(all_tokens)
    lemmas = [
        _LEMMATIZER.lemmatize(tok, _wordnet_pos(tag))
        for tok, tag in tagged
        if tok.isalpha() and tok not in _STOP_WORDS
    ]
    return " ".join(lemmas)


class FAQChatbot:
    """Loads a FAQ CSV and answers free-text questions via TF-IDF cosine similarity."""

    # Candidate column names, checked in order, for auto-detection.
    QUESTION_CANDIDATES = ["instruction", "question", "query", "text", "prompt"]
    ANSWER_CANDIDATES = ["response", "answer", "reply", "output"]

    def __init__(self, csv_path: str, question_col: Optional[str] = None,
                 answer_col: Optional[str] = None, min_similarity: float = 0.2):
        self.min_similarity = min_similarity
        self.df = pd.read_csv(csv_path)
        self.question_col, self.answer_col = self._resolve_columns(question_col, answer_col)

        self.df = self.df.dropna(subset=[self.question_col, self.answer_col]).reset_index(drop=True)
        self.df["_clean_question"] = self.df[self.question_col].apply(preprocess)

        # Drop rows that preprocess down to nothing (e.g. pure punctuation/numbers).
        self.df = self.df[self.df["_clean_question"].str.len() > 0].reset_index(drop=True)

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_clean_question"])

        print(f"Loaded {len(self.df)} FAQs  |  questions: '{self.question_col}'  "
              f"|  answers: '{self.answer_col}'")

    def _resolve_columns(self, question_col: Optional[str], answer_col: Optional[str]):
        cols_lower = {c.lower(): c for c in self.df.columns}

        if question_col is None:
            question_col = next((cols_lower[c] for c in self.QUESTION_CANDIDATES if c in cols_lower), None)
        if answer_col is None:
            answer_col = next((cols_lower[c] for c in self.ANSWER_CANDIDATES if c in cols_lower), None)

        if question_col is None or answer_col is None:
            raise ValueError(
                f"Couldn't auto-detect question/answer columns from {list(self.df.columns)}. "
                "Pass question_col=... and answer_col=... explicitly when creating FAQChatbot."
            )
        return question_col, answer_col

    def get_response(self, user_query: str):
        """Return (answer, similarity_score, matched_question) for the closest FAQ."""
        clean_query = preprocess(user_query)
        if not clean_query:
            return "Could you rephrase that as a full question?", 0.0, None

        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < self.min_similarity:
            return (
                "I'm not confident I have an answer for that. Could you rephrase, "
                "or ask something else?",
                best_score,
                None,
            )

        answer = self.df.iloc[best_idx][self.answer_col]
        matched_question = self.df.iloc[best_idx][self.question_col]
        return answer, best_score, matched_question


def chat_loop(bot: FAQChatbot, show_debug: bool = False) -> None:
    """Simple command-line chat UI (the 'optional' UI requirement)."""
    print("\nFAQ Chatbot ready! Ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue

        answer, score, matched_q = bot.get_response(user_input)
        print(f"Bot: {answer}")
        if show_debug and matched_q is not None:
            print(f"     (matched FAQ: \"{matched_q}\"  |  similarity: {score:.2f})")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "faqs.csv"
    bot = FAQChatbot(csv_path)
    chat_loop(bot, show_debug=True)
