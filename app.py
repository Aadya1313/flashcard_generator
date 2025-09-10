import streamlit as st
import os
from main import fetch_wikipedia_intro, classify_subject_gpt, create_flashcard
import re

OUTPUT_DIR = "images/output_flashcards/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


st.title("FACTZY - Facts made Easy")
st.markdown("<span style='font-size:18px;'>Your effortless flashcard companion — turns complex web content into bite-sized flashcards — making smart revision easy, fast, and fun.</span>", unsafe_allow_html=True)

topic = st.text_input("Enter a topic:")

if st.button("Generate Flashcards") and topic.strip():
    with st.spinner("Fetching and generating flashcards..."):
        text = fetch_wikipedia_intro(topic)
        if not text.strip():
            st.error("No content found for this topic.")
        else:
            # Clean Wikipedia reference markers
            cleaned_text = re.sub(r'\[\d+(?:\]\[\d+)*\]', '', text)
            cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
            subject = classify_subject_gpt(cleaned_text)
            sentences = re.split(r'(?<=[.!?]) +', cleaned_text.strip())
            flashcard_chunks = [chunk.strip() for chunk in sentences if chunk.strip()]
            if not flashcard_chunks:
                flashcard_chunks = [cleaned_text.strip()]
            st.subheader("Flashcards")
            for i, chunk in enumerate(flashcard_chunks):
                if chunk:
                    flashcard_path = os.path.join(OUTPUT_DIR, f"{subject}_flashcard_{i+1}_from_web.png")
                    create_flashcard(chunk, flashcard_path)
                    st.image(flashcard_path, caption=f"Flashcard {i+1}", use_container_width=True)