# Language Translation Tool

A simple **Streamlit web application** for translating text between multiple languages using Google Translate, with optional text-to-speech audio.

## Features

* Translate text between 8 languages
* Automatic source-language detection
* Simple Streamlit interface
* Text-to-speech for translated text
* Error handling for unsupported audio languages

## Supported Languages

* English
* Hindi
* Spanish
* French
* German
* Russian
* Japanese
* Chinese (Simplified)

## Tech Stack

* Python
* Streamlit
* `deep-translator`
* `gTTS`

## Project Structure

```text
Translator_app/
├── translator.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run translator.py
```

The application will open in your browser.

## How It Works

```text
Enter Text
    ↓
Select Source & Target Language
    ↓
Google Translator
    ↓
Translated Text
    ↓
Text-to-Speech Audio
```

## Note

Translation is provided through Google Translate via the `deep-translator` library. Text-to-speech availability may vary by language.
