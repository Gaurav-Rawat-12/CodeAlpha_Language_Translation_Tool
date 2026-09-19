# Language Translation Tool & FAQ Chatbot

Two mini-projects: a text translation tool (Task 1) and an FAQ-answering chatbot (Task 2).

## Contents
- [Repository Structure](#repository-structure)
- [Task 1: Language Translation Tool](#task-1-language-translation-tool)
- [Task 2: FAQ Chatbot](#task-2-faq-chatbot)

---

## Repository Structure
```
├── Translator_app/
│   ├── translator.py
│   └── requirements.txt
└── <your Task 2 folder>/        (adjust to match your repo — e.g. FAQ_Chatbot_app/)
    ├── faq_chatbot.py
    ├── sample_faqs.csv
    └── requirements.txt
```
Run each project's setup commands from inside its own folder.

---

## Task 1: Language Translation Tool

A Streamlit web app: enter text, pick a source and target language, and get the translation back — with optional audio playback.

**What it does**
- Text area for input, with source-language (including auto-detect) and target-language dropdowns
- Translates the text via Google Translate, through the `deep-translator` library
- Displays the translated text
- *Optional requirement:* text-to-speech — plays back the translation as audio using `gTTS`, with a graceful fallback message if a language isn't supported for audio

**Supported languages:** English, Hindi, Spanish, French, German, Russian, Japanese, Chinese (Simplified) — plus auto-detect for the source language

**Files**

| File | Purpose |
|---|---|
| `translator.py` | Streamlit app — UI, translation call, text-to-speech playback |
| `requirements.txt` | Python dependencies |

**Setup**
```bash
pip install -r requirements.txt
streamlit run translator.py
```

**Tech stack:** Python · Streamlit · deep-translator (Google Translate) · gTTS (text-to-speech)

---

## Task 2: FAQ Chatbot

A chatbot that answers free-text questions by matching them to the closest FAQ using NLP preprocessing and cosine similarity.

**How it works**
1. **Collect FAQs** — loads question/answer pairs from a CSV. Columns are auto-detected: `instruction`/`response` (the schema used by the [Bitext Gen AI Chatbot Customer Support Dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)) or generic `question`/`answer`.
2. **Preprocess** — NLTK cleans each question: lowercasing, punctuation removal, tokenization, stopword removal, and POS-aware lemmatization (each word's part of speech is tagged first, so e.g. "shipping" and "ship" reduce to the same form instead of being treated as unrelated words).
3. **Match** — FAQ questions are vectorized with TF-IDF; a user's question is compared against all of them using cosine similarity, and the closest match is selected.
4. **Respond** — returns the paired answer, or a "not confident, please rephrase" message if nothing clears a similarity threshold.
5. **Chat UI** *(optional requirement)* — a command-line loop for asking questions interactively.

**Files**

| File | Purpose |
|---|---|
| `faq_chatbot.py` | Preprocessing, TF-IDF/cosine-similarity matching engine, CLI chat loop |
| `sample_faqs.csv` | Small example FAQ set for testing the pipeline |
| `requirements.txt` | Python dependencies |

**Setup**
```bash
pip install -r requirements.txt
python faq_chatbot.py sample_faqs.csv   # try it with the sample data
python faq_chatbot.py your_faqs.csv     # or your own FAQ CSV (e.g. the Bitext dataset)
```

**Tech stack:** Python · NLTK · scikit-learn (TF-IDF + cosine similarity) · pandas

**Known limitation:** matching is word-overlap based, so it works best when a question shares vocabulary with the FAQ wording. A heavily reworded paraphrase with no shared words (e.g. "get my money back" vs. "refund policy") may not match as well — swapping in sentence embeddings (e.g. `sentence-transformers`) would address this.
