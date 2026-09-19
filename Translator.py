import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

st.set_page_config(page_title="Language Translation Tool", layout="centered")

st.title("Language Translation Tool")
st.caption("Language Translation using Streamlit & Google Translator")

# Language mapping
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
}

col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("Source Language", options=["auto"] + list(LANGUAGES.keys()))
with col2:
    target_lang = st.selectbox("Target Language", options=list(LANGUAGES.keys()), index=1)

user_text = st.text_area("Enter text to translate:", height=130)

if st.button("Translate", type="primary", use_container_width=True):
    if user_text.strip():
        try:
            src = "auto" if source_lang == "auto" else LANGUAGES[source_lang]
            tgt = LANGUAGES[target_lang]
            
            translated_text = GoogleTranslator(source=src, target=tgt).translate(user_text)
            
            st.subheader("Translated Output:")
            st.success(translated_text)
            
            # Optional requirement: Text-to-Speech
            try:
                tts = gTTS(text=translated_text, lang=tgt)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.audio(audio_buffer, format="audio/mp3")
            except Exception:
                st.info("Audio pronunciation not supported for this language.")
                
        except Exception as e:
            st.error(f"Translation error: {e}")
    else:
        st.warning("Please enter text to translate.")
